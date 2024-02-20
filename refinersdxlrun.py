import nodes
import comfy
import importlib
import latent_preview
from pytorch_lightning import seed_everything
import gc
import torch
import os
import cuda_malloc
from comfy import model_management
from enum import Enum
from comfy.model_management import vram_state, VRAMState
import torch
from IPython.display import clear_output
from comfy import model_management
from PIL import Image
import numpy as np
from einops import rearrange
from comfy import model_management, sd
from torchvision.transforms.functional import to_pil_image
from sdxlrun import get_device_memory
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
from pytorch_lightning import seed_everything
from ipywidgets import Image, Layout, VBox
from io import BytesIO
from PIL import Image as pilimage
from PIL.PngImagePlugin import PngInfo
import json

def loadsdxlrefiner(sdxl_args):
    refiner = nodes.CheckpointLoaderSimple()
    refined_out = refiner.load_checkpoint(
            sdxl_args.refinerckpt_name,
            output_vae=True,
            output_clip=True,
            )
    clear_output()
    
    get_device_memory()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    
    return refined_out
    

def runsdxlrefiner(sdxl_args, samples, model, refined_out):
    refinermodel, refinerclip, refinervae = refined_out
    
    model_management.soft_empty_cache()
    get_device_memory()
    
    if sdxl_args.stop_at_last_layer != None:
        refinerclip = refinerclip.clone()
        refinerclip.clip_layer(sdxl_args.stop_at_last_layer)
    ptokens = refinerclip.tokenize(sdxl_args.prompt)
    pcond, ppooled = refinerclip.encode_from_tokens(ptokens, return_pooled=True)
    
    positive = [[pcond, {"pooled_output": ppooled, "aesthetic_score": sdxl_args.ascore, "width": sdxl_args.refinerwidth,"height": sdxl_args.refinerheight}]]
    
    tokens = refinerclip.tokenize(sdxl_args.negativeprompt)
    cond, pooled = refinerclip.encode_from_tokens(tokens, return_pooled=True)
    
    negative = [[cond, {"pooled_output": pooled, "aesthetic_score": sdxl_args.ascore, "width": sdxl_args.refinerwidth,"height": sdxl_args.refinerheight}]]
    
    device = comfy.model_management.get_torch_device()
    
    latent_image = samples
    if sdxl_args.refinerdisable_noise:
        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device=sdxl_args.noisedevice)
        
    else:
        batch_inds = None#latent["batch_index"] if "batch_index" in latent else None
        noise = comfy.sample.prepare_noise(latent_image, sdxl_args.seed, batch_inds)
    
    noise_mask = None
    # latent = None
    # if "noise_mask" in latent:
    #     noise_mask = latent["noise_mask"]
    
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
    pbar = comfy.utils.ProgressBar(sdxl_args.refiner_steps)
    
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

    latent_image = samples
    
    refinedsamples = comfy.sample.sample(sdxl_args, 
                                         refinermodel, 
                                         noise, 
                                         sdxl_args.refiner_steps, 
                                         sdxl_args.cfg, 
                                         sdxl_args.sampler_name, 
                                         sdxl_args.scheduler, 
                                         positive, 
                                         negative, 
                                         latent_image, 
                                         denoise=sdxl_args.denoise, 
                                         disable_noise=sdxl_args.refinerdisable_noise, 
                                         start_step=sdxl_args.refiner_start_step, 
                                         last_step=sdxl_args.refiner_last_step, 
                                         force_full_denoise=sdxl_args.refinerforce_full_denoise, 
                                         noise_mask=noise_mask, 
                                         callback=callback, 
                                         seed=sdxl_args.seed)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    
    get_device_memory()
    
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    get_device_memory()
    
    vae_decode_method = sdxl_args.vae_decode_method
    if vae_decode_method == "normal":
        image = refinervae.decode(refinedsamples)
    else:
        image = refinervae.decode_tiled(refinedsamples)
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

        if not sdxl_args.disable_metadata:
            metadata = PngInfo()

            if sdxl_args is not None:
                for key, value in sdxl_args.__dict__.items():
                    metadata.add_text(key, json.dumps(str(value)))
        im.save(os.path.join(output_folder, f'{sdxl_args.saveprefix}_{count+1:05d}_refined.png'), pnginfo=metadata, compress_level=4)
        count+=1

    get_device_memory()
    return refinermodel, refinedsamples
