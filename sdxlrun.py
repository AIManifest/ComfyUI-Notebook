import os
import xformers
import xformers.ops
from comfy import model_management
import folder_paths
from enum import Enum
import torch
import cuda_malloc
from comfy.model_management import VRAMState
from IPython.display import clear_output
from pytorch_lightning import seed_everything
from PIL import Image
import numpy as np
from einops import rearrange
from comfy import model_management, sd
from torchvision.transforms.functional import to_pil_image
from comfy import latent_formats
from comfy.latent_formats import SDXL
from IPython.display import display, clear_output
import io
import nodes
import comfy
import importlib
import latent_preview
import gc
import random
from comfy_extras import nodes_clip_sdxl
import time
from ipywidgets import Image, Layout, VBox
from io import BytesIO
from PIL import Image as pilimage
from PIL.PngImagePlugin import PngInfo
import json


def get_device_memory():
    total_memory = torch.cuda.get_device_properties(0).total_memory
    total_memory_gb = total_memory / (1024 ** 3)
    reserved_memory = torch.cuda.memory_reserved(0)
    reserved_memory_gb = reserved_memory / (1024 ** 3)
    allocated_memory = torch.cuda.memory_allocated(0)
    allocated_memory_gb = allocated_memory / (1024 ** 3)
    free_memory = total_memory - allocated_memory
    free_memory_gb = free_memory / (1024 ** 3)

    print(f"Total memory: {total_memory_gb:.2f} GB")
    print(f"Reserved memory: {reserved_memory_gb:.2f} GB")
    print(f"Allocated memory: {allocated_memory_gb:.2f} GB")
    print(f"Free memory: {free_memory_gb:.2f} GB")

get_device_memory()

def load_lora(model, clip, lora_name, strength_model, strength_clip):
    loaded_lora = None
    if strength_model == 0 and strength_clip == 0:
        return (model, clip)

    lora_path = folder_paths.get_full_path("loras", lora_name)
    lora = None
    if loaded_lora is not None:
            if loaded_lora[0] == lora_path:
                lora = loaded_lora[1]
            else:
                temp = loaded_lora
                loaded_lora = None
                del temp

    if lora is None:
        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
        loaded_lora = (lora_path, lora)

    model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)
    del model
    del clip
    return (model_lora, clip_lora)

def loadsdxl(sdxl_args):
    start = time.time()
    loader = nodes.CheckpointLoaderSimple()
    out = loader.load_checkpoint(
            sdxl_args.ckpt_name,
            output_vae=True,
            output_clip=True,
            )
    
    model, clip, vae, clipvision = out
    
    clear_output(wait=True)
    
    get_device_memory()
    
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    
    if sdxl_args.lora_name != None:
        lora = load_lora(model, clip, sdxl_args.lora_name, sdxl_args.strength_model, sdxl_args.strength_clip)
        old_model, old_clip, old_out = model, clip, out
        model, clip = lora
        del old_model
        del old_clip
        del old_out
        out = (model, clip, vae, clipvision)
    end = time.time()
    print(f'model loaded in {end-start:.02f} seconds')
    return out

def runsdxl(sdxl_args, out):
    model, clip, vae, _ = out

    if sdxl_args.stop_at_last_layer != None:
        clip = clip.clone()
        clip.clip_layer(sdxl_args.stop_at_last_layer)
        
    tokens = clip.tokenize(sdxl_args.prompt)
    tokens["l"] = clip.tokenize(sdxl_args.prompt)["l"]
    if len(tokens["l"]) != len(tokens["g"]):
        empty = clip.tokenize("")
        while len(tokens["l"]) < len(tokens["g"]):
            tokens["l"] += empty["l"]
        while len(tokens["l"]) > len(tokens["g"]):
            tokens["g"] += empty["g"]
    pcond, ppooled = clip.encode_from_tokens(tokens, return_pooled=True)
    
    positive = [[pcond, {"pooled_output": ppooled, "width": sdxl_args.width, "height": sdxl_args.height, "crop_w": sdxl_args.crop_w, "crop_h": sdxl_args.crop_h, "target_width": sdxl_args.target_width, "target_height": sdxl_args.target_height}]]
    tokens = clip.tokenize(sdxl_args.negativeprompt)
    tokens["l"] = clip.tokenize(sdxl_args.negativeprompt)["l"]
    if len(tokens["l"]) != len(tokens["g"]):
        empty = clip.tokenize("")
        while len(tokens["l"]) < len(tokens["g"]):
            tokens["l"] += empty["l"]
        while len(tokens["l"]) > len(tokens["g"]):
            tokens["g"] += empty["g"]
    ncond, npooled = clip.encode_from_tokens(tokens, return_pooled=True)
    
    negative = [[ncond, {"pooled_output": npooled, "width": sdxl_args.width, "height": sdxl_args.height, "crop_w": sdxl_args.crop_w, "crop_h": sdxl_args.crop_h, "target_width": sdxl_args.target_width, "target_height": sdxl_args.target_height}]]
    
    latentempty = nodes.EmptyLatentImage()
    latent = latentempty.generate(sdxl_args.imagewidth, sdxl_args.imageheight, sdxl_args.batch_size)
    latent = latent[0]

    force_full_denoise = sdxl_args.force_full_denoise
    disable_noise = sdxl_args.disable_noise
    
    device = comfy.model_management.get_torch_device()
    latent_image = latent["samples"]
    
    if disable_noise:
        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device=sdxl_args.noisedevice)
    else:
        batch_inds = latent["batch_index"] if "batch_index" in latent else None
        noise = comfy.sample.prepare_noise(latent_image, sdxl_args.seed, batch_inds)
    
    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]
    
    preview_format = "PNG"
    if preview_format not in ["JPEG", "PNG"]:
        preview_format = "JPEG"
    
    class LatentFormat:
        def process_in(self, latent):
            return latent * self.scale_factor
    
        def process_out(self, latent):
            return latent / self.scale_factor
    latent_format = SDXL()
    use_preview = sdxl_args.use_preview
    if use_preview:
        previewer = latent_preview.Latent2RGBPreviewer(latent_format.latent_rgb_factors)#get_previewer(device, model.model.latent_format)
    else:
        previewer = latent_preview.get_previewer(device, model.model.latent_format)
    pbar = comfy.utils.ProgressBar(sdxl_args.steps)
    
    image_widget = Image()
    vbox = VBox([image_widget], layout=Layout(width="256px"))
    display(vbox)

    def callback(step, x0, x, total_steps):
        preview_bytes = None
        if previewer:
            preview_bytes = previewer.decode_latent_to_preview_image(preview_format, x0)
            if use_preview:
                new_bytes = preview_bytes[1]
                display_bytes = BytesIO()
                new_bytes.save(display_bytes, format='PNG')
                image_data = display_bytes.getvalue()
                image_widget.value = image_data
        pbar.update_absolute(step + 1, total_steps, preview_bytes)

    samples = comfy.sample.sample(sdxl_args, 
                                  model, 
                                  noise, 
                                  sdxl_args.steps, 
                                  sdxl_args.cfg, 
                                  sdxl_args.sampler_name, 
                                  sdxl_args.scheduler, 
                                  positive, 
                                  negative, 
                                  latent_image, 
                                  denoise=sdxl_args.denoise, 
                                  disable_noise=sdxl_args.disable_noise, 
                                  start_step=sdxl_args.start_step, 
                                  last_step=sdxl_args.last_step, 
                                  force_full_denoise=sdxl_args.force_full_denoise, 
                                  noise_mask=noise_mask, 
                                  callback=callback, 
                                  seed=sdxl_args.seed)
    samplez = latent.copy()
    samplez["samples"] = samples
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    get_device_memory()
    
    model_management.unload_model(model)
    
    
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    get_device_memory()
    
    try:
        model_management.unload_model(model)
    except:
        model_management.unload_model(refinermodel)
    
    samples=samples.cpu()
    
    vae_decode_method = sdxl_args.vae_decode_method
    if vae_decode_method == "normal":
        image = vae.decode(samples)
    else:
        image = vae.decode_tiled(samples)
    vaeimage = rearrange(image, 'b h w c -> b c h w')

    output_folder = sdxl_args.output_folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
        
    count = len(os.listdir(output_folder))
    
    for im in vaeimage:
        im = to_pil_image(im)
        new_im = im
        bytes_image = BytesIO()
        new_im.save(bytes_image, format='PNG')
        image_data = bytes_image.getvalue()
        image_widget = Image(value=image_data, format='png')
        vbox = VBox([image_widget], layout=Layout(width="512px"))
        display(vbox)

        if sdxl_args.save_base_image:
            if not sdxl_args.disable_metadata:
                metadata = PngInfo()

                if sdxl_args is not None:
                    for key, value in sdxl_args.__dict__.items():
                        metadata.add_text(key, json.dumps(value))
            im.save(os.path.join(output_folder, f'{sdxl_args.saveprefix}_{count+1:05d}_.png'), pnginfo=metadata, compress_level=4)
        count+=1
    try:
        model_management.unload_model(refinermodel)
    except:
        model_management.unload_model(model)
    get_device_memory()
    return model, samples
