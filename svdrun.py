import os
import cv2
import xformers
import xformers.ops
from comfy import model_management
import folder_paths
from enum import Enum
from tqdm.auto import tqdm
import torch
import cuda_malloc
from comfy.model_management import VRAMState
from IPython.display import clear_output
from pytorch_lightning import seed_everything
from PIL import Image as pil_image
from PIL import ImageOps
import numpy as np
from einops import rearrange
from comfy import model_management, sd
from torchvision.transforms.functional import to_pil_image
from comfy import latent_formats
from comfy.latent_formats import SDXL
from IPython.display import display, clear_output, HTML
import io
import nodes
import comfy
import importlib
import latent_preview
import gc
import random
import base64
from comfy_extras import nodes_clip_sdxl
import time
from ipywidgets import Image, Layout, VBox
from io import BytesIO
from PIL import Image as pilimage
from PIL.PngImagePlugin import PngInfo
import json
from comfy_extras.nodes_canny import canny
from comfy_extras.nodes_video_model import ImageOnlyCheckpointLoader, SVD_img2vid_Conditioning, VideoLinearCFGGuidance
from comfy_extras.nodes_images import SaveAnimatedWEBP
from ComfyUI_VideoHelperSuite.videohelpersuite.nodes import VideoCombine
from ComfyUI_Frame_Interpolation import FILM_VFI
from custom_nodes.comfy_controlnet_preprocessors.nodes.util import common_annotator_call, img_np_to_tensor, skip_v1
from custom_nodes.comfy_controlnet_preprocessors.v1 import midas, leres
from custom_nodes.comfy_controlnet_preprocessors.v11 import zoe, normalbae
import numpy as np
from iprogress import iprogress
from natsort import natsorted

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

def apply_controlnet(positive, negative, control_net, image, strength, start_percent, end_percent):
        if strength == 0:
            return (positive, negative)

        control_hint = image.movedim(-1,1)
        cnets = {}

        out = []
        for conditioning in [positive, negative]:
            c = []
            for t in conditioning:
                d = t[1].copy()

                prev_cnet = d.get('control', None)
                if prev_cnet in cnets:
                    c_net = cnets[prev_cnet]
                else:
                    c_net = control_net.copy().set_cond_hint(control_hint, strength, (1.0 - start_percent, 1.0 - end_percent))
                    c_net.set_previous_controlnet(prev_cnet)
                    cnets[prev_cnet] = c_net

                d['control'] = c_net
                d['control_apply_to_uncond'] = False
                n = [t[0], d]
                c.append(n)
            out.append(c)
        return (out[0], out[1])

def load_image(image_path):
        # image_path = folder_paths.get_annotated_filepath(image)
        i = pil_image.open(image_path)
        i = ImageOps.exif_transpose(i)
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        if 'A' in i.getbands():
            mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
            mask = 1. - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
        return (image, mask)

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
    
    model, clip, vae = out
    
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

def create_video(image_folder, fps, video_name):
    ext = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
    images = [img for img in natsorted(os.listdir(image_folder)) if os.path.splitext(img)[1] in ext]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    video = cv2.VideoWriter(os.path.join(image_folder, video_name), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width,height))
    for image in iprogress(images, desc="creating video", colour="sunset"):
        video.write(cv2.imread(os.path.join(image_folder, image)))
    cv2.destroyAllWindows()
    video.release()

import os
import cv2
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
from PIL import Image as pil_image
from PIL import ImageOps
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
from comfy_extras.nodes_canny import canny
from custom_nodes.comfy_controlnet_preprocessors.nodes.util import common_annotator_call, img_np_to_tensor, skip_v1
from custom_nodes.comfy_controlnet_preprocessors.v1 import midas, leres
from custom_nodes.comfy_controlnet_preprocessors.v11 import zoe, normalbae
import numpy as np
from iprogress import iprogress
from natsort import natsorted

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

def apply_controlnet(positive, negative, control_net, image, strength, start_percent, end_percent):
        if strength == 0:
            return (positive, negative)

        control_hint = image.movedim(-1,1)
        cnets = {}

        out = []
        for conditioning in [positive, negative]:
            c = []
            for t in conditioning:
                d = t[1].copy()

                prev_cnet = d.get('control', None)
                if prev_cnet in cnets:
                    c_net = cnets[prev_cnet]
                else:
                    c_net = control_net.copy().set_cond_hint(control_hint, strength, (1.0 - start_percent, 1.0 - end_percent))
                    c_net.set_previous_controlnet(prev_cnet)
                    cnets[prev_cnet] = c_net

                d['control'] = c_net
                d['control_apply_to_uncond'] = False
                n = [t[0], d]
                c.append(n)
            out.append(c)
        return (out[0], out[1])

def load_image(image_path):
        # image_path = folder_paths.get_annotated_filepath(image)
        i = pil_image.open(image_path)
        i = ImageOps.exif_transpose(i)
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        if 'A' in i.getbands():
            mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
            mask = 1. - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
        return (image, mask)

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
    
    model, clip, vae = out
    
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
        out = (model, clip, vae)
    end = time.time()
    print(f'model loaded in {end-start:.02f} seconds')
    return out

def create_video(image_folder, fps, video_name):
    ext = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
    images = [img for img in natsorted(os.listdir(image_folder)) if os.path.splitext(img)[1] in ext]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    video = cv2.VideoWriter(os.path.join(image_folder, video_name), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width,height))
    for image in iprogress(images, desc="creating video", colour="sunset"):
        video.write(cv2.imread(os.path.join(image_folder, image)))
    cv2.destroyAllWindows()
    video.release()

def runsvd(sdxl_args, out, refiner_out, control_net):
    svd_conditioner = SVD_img2vid_Conditioning()
    svd_guidance = VideoLinearCFGGuidance()
    svd_saver = SaveAnimatedWEBP()
    svd_ckpt_name = sdxl_args.svd_ckpt_name
    svd_model, svd_clipvision, svd_vae = sdxl_args.svd_loaded#svd_loader.load_checkpoint(svd_ckpt_name, output_vae=True, output_clip=True)
    svd_ckpt_name = sdxl_args.svd_ckpt_name
    svd_min_cfg = sdxl_args.svd_min_cfg
    svd_width = sdxl_args.svd_width
    svd_height = sdxl_args.svd_height
    svd_video_frames = sdxl_args.svd_video_frames
    svd_motion_bucket_id = sdxl_args.svd_motion_bucket_id
    svd_fps = sdxl_args. svd_fps
    svd_augmentation_level = sdxl_args.svd_augmentation_level

    # clear_output(wait=True)

    model, clip, vae = out

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
    if sdxl_args.is_controlnet:
        image, image_mask = load_image(sdxl_args.controlnet_image)
        if "canny" in sdxl_args.controlnet_name:
            output = canny(image.movedim(-1, 1), sdxl_args.controlnet_low_threshold, sdxl_args.controlnet_high_threshold)
            img_out = output[1].repeat(1, 3, 1, 1).movedim(1, -1)
        elif "depth" in sdxl_args.controlnet_name:
            np_detected_map = common_annotator_call(zoe.ZoeDetector(), image)
            img_out = img_np_to_tensor(np_detected_map)
        positive, negative = apply_controlnet(positive, negative, control_net[0], img_out, sdxl_args.controlnet_strength, sdxl_args.controlnet_start_percent, sdxl_args.controlnet_end_percent)

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

    output_folder = sdxl_args.output_folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    count = len(os.listdir(output_folder))
    
    preview_save_path = os.path.join(sdxl_args.output_folder, f'{sdxl_args.saveprefix}_{count+1:05d}')
    if not os.path.exists(preview_save_path):
        os.makedirs(preview_save_path, exist_ok=True)

    def callback(step, x0, x, total_steps):
        preview_bytes = None
        idx = len(os.listdir(preview_save_path))
        if previewer:
            preview_bytes = previewer.decode_latent_to_preview_image(preview_format, x0)
            if use_preview:
                new_bytes = preview_bytes[1]
                preview_save = os.path.join(preview_save_path, f'preview_{idx+1:05d}.png')
                new_bytes.save(preview_save)
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
    
    use_refiner = True
    if use_refiner:
        refinermodel, refinerclip, refinervae = refiner_out

        ptokens = refinerclip.tokenize(sdxl_args.prompt)
        pcond, ppooled = refinerclip.encode_from_tokens(ptokens, return_pooled=True)
        
        refiner_positive = [[pcond, {"pooled_output": ppooled, "aesthetic_score": sdxl_args.ascore, "width": sdxl_args.refinerwidth,"height": sdxl_args.refinerheight}]]
        
        tokens = refinerclip.tokenize(sdxl_args.negativeprompt)
        cond, pooled = refinerclip.encode_from_tokens(tokens, return_pooled=True)
        
        refiner_negative = [[cond, {"pooled_output": pooled, "aesthetic_score": sdxl_args.ascore, "width": sdxl_args.refinerwidth,"height": sdxl_args.refinerheight}]]

        refinernoise = comfy.sample.prepare_noise(samples, sdxl_args.seed, batch_inds)
    
        refiner_steps = sdxl_args.refiner_steps
        refiner_start_step = sdxl_args.last_step
        refiner_last_step = sdxl_args.refiner_last_step
        refinersamples = comfy.sample.sample(sdxl_args,
                                      refinermodel, 
                                      refinernoise, 
                                      refiner_steps, 
                                      sdxl_args.cfg, 
                                      sdxl_args.sampler_name, 
                                      sdxl_args.scheduler, 
                                      refiner_positive, 
                                      refiner_negative, 
                                      samples, 
                                      denoise=sdxl_args.denoise, 
                                      disable_noise=sdxl_args.refinerdisable_noise, 
                                      start_step=refiner_start_step, 
                                      last_step=refiner_last_step, 
                                      force_full_denoise=sdxl_args.refinerforce_full_denoise, 
                                      noise_mask=noise_mask, 
                                      callback=callback, 
                                      seed=sdxl_args.seed)
        del refinermodel
        del refinerclip
        del refinervae
        
        try:
            del refinerout
        except:
            pass

        old_samples = samples
        samples = refinersamples
    
    samples=samples.cpu()

    if sdxl_args.vae_path:
        print(f"Loading {sdxl_args.vae_path}")
        sd = comfy.utils.load_torch_file(sdxl_args.vae_path)
        vae = comfy.sd.VAE(sd=sd)
    
    vae_decode_method = sdxl_args.vae_decode_method
    if vae_decode_method == "normal":
        image = vae.decode(samples)
    else:
        image = vae.decode_tiled(samples)
    vaeimage = rearrange(image, 'b h w c -> b c h w')

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
        if sdxl_args.create_video_preview:
            create_video(preview_save_path, 5, f'{sdxl_args.saveprefix}_{count+1:05d}_.mp4')
        
        # svd_image, _ = load_image(os.path.join(output_folder, f'{sdxl_args.saveprefix}_{count+1:05d}_.png'))
        del out
        del refiner_out

        i = ImageOps.exif_transpose(im)
        svd_image = i.convert("RGB")
        svd_image = np.array(svd_image).astype(np.float32) / 255.0
        svd_image = torch.from_numpy(svd_image)[None,]
        latent = svd_image#svd_vae.encode(svd_image)
        new_svd_model = svd_guidance.patch(svd_model, svd_min_cfg)[0]
        svd_positive, svd_negative, svd_latent = svd_conditioner.encode(svd_clipvision, latent, svd_vae, svd_width, svd_height, svd_video_frames, svd_motion_bucket_id, svd_fps, svd_augmentation_level)
        svd_latent = svd_latent["samples"]
        svd_noise = comfy.sample.prepare_noise(svd_latent, sdxl_args.seed, batch_inds)
        svd_sampler = sdxl_args.svd_sampler
        svd_scheduler = sdxl_args.svd_scheduler
        
        svd_samples = comfy.sample.sample(sdxl_args,
                                      new_svd_model, 
                                      svd_noise, 
                                      sdxl_args.steps, 
                                      sdxl_args.cfg, 
                                      svd_sampler, 
                                      svd_scheduler,
                                      svd_positive, 
                                      svd_negative, 
                                      svd_latent, 
                                      denoise=sdxl_args.denoise, 
                                      disable_noise=sdxl_args.refinerdisable_noise, 
                                      start_step=sdxl_args.start_step, 
                                      last_step=sdxl_args.last_step, 
                                      force_full_denoise=sdxl_args.refinerforce_full_denoise, 
                                      noise_mask=noise_mask, 
                                      callback=callback, 
                                      seed=sdxl_args.seed)
        
        count+=1
        images = svd_vae.decode(svd_samples)
        # images = rearrange(images, 'b h w c -> b c h w')
        svd_fps_out = sdxl_args.svd_fps_out
        svd_filename_prefix = sdxl_args.saveprefix
        svd_lossless = sdxl_args.svd_lossless
        svd_quality = sdxl_args.svd_quality
        svd_method = sdxl_args.svd_method
        svd_num_frames = sdxl_args.svd_num_frames
        svd_multiplier = sdxl_args.svd_multiplier
        svd_clear_cache_after_n_frames = sdxl_args.svd_clear_cache_after_n_frames
        film_name = sdxl_args.film_name
        
        film_vfi = FILM_VFI()
        videocombine = VideoCombine()
        
        vfi_images = film_vfi.vfi(film_name,
                                images,
                                clear_cache_after_n_frames=svd_clear_cache_after_n_frames,
                                multiplier=svd_multiplier,
                                optional_interpolation_states = None)
        
        samples = videocombine.combine_video(vfi_images[0],
                                            sdxl_args.svd_fps_out,
                                            sdxl_args.svd_loop_count,
                                            filename_prefix=sdxl_args.saveprefix,
                                            format=sdxl_args.svd_format,
                                            pingpong=False,
                                            save_output=True,
                                            prompt=None,
                                            extra_pnginfo=None,
                                            audio=None,
                                            unique_id=None,
                                            manual_format_widgets=None,
                                            batch_manager=None)
        
        video_display_list = os.listdir(os.path.join(os.path.dirname(__file__), "output"))
        video_display_path = os.path.join(os.path.dirname(__file__), f"output/{video_display_list[-1]}")
        
        def file_to_base64(path):
            with open(path, "rb") as file:
                encoded = base64.b64encode(file.read()).decode("utf-8")
            return encoded
        
        video_base64 = file_to_base64(video_display_path)
        video_html = f"""
        <video width="640" height="480" controls>
          <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
          Your browser does not support the video tag.
        </video>
        """
        display(HTML(video_html))

        # animated_webp = svd_saver.save_images(images, svd_fps, svd_filename_prefix, svd_lossless, svd_quality, svd_method, num_frames=svd_num_frames, prompt=None, extra_pnginfo=None)
    get_device_memory()

    return model, samples

def batch_runsvd(sdxl_args):
    svd_conditioner = SVD_img2vid_Conditioning()
    svd_guidance = VideoLinearCFGGuidance()
    svd_saver = SaveAnimatedWEBP()
    svd_model, svd_clipvision, svd_vae = sdxl_args.svd_loaded
    clear_output(wait=True)
    svd_ckpt_name = sdxl_args.svd_ckpt_name
    svd_min_cfg = sdxl_args.svd_min_cfg
    svd_width = sdxl_args.svd_width
    svd_height = sdxl_args.svd_height
    svd_video_frames = sdxl_args.svd_video_frames
    svd_motion_bucket_id = sdxl_args.svd_motion_bucket_id
    svd_fps = sdxl_args. svd_fps
    svd_augmentation_level = sdxl_args.svd_augmentation_level
    svd_sampler = sdxl_args.svd_sampler
    svd_scheduler = sdxl_args.svd_scheduler

    # clear_output(wait=True)
        
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

    output_folder = sdxl_args.output_folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    count = len(os.listdir(output_folder))
    
    preview_save_path = os.path.join(sdxl_args.output_folder, f'{sdxl_args.saveprefix}_{count+1:05d}')
    if not os.path.exists(preview_save_path):
        os.makedirs(preview_save_path, exist_ok=True)

    batch_folder = sorted([f for f in os.listdir(sdxl_args.init_image_folder_path_for_svd) if f.lower().endswith(('.png', '.jpeg', '.jpg'))])

    preview_save_path = os.path.join(sdxl_args.output_folder, f'{sdxl_args.saveprefix}_{count+1:05d}')
    if not os.path.exists(preview_save_path):
        os.makedirs(preview_save_path, exist_ok=True)

    def callback(step, x0, x, total_steps):
        preview_bytes = None
        idx = len(os.listdir(preview_save_path))
        if previewer:
            preview_bytes = previewer.decode_latent_to_preview_image(preview_format, x0)
            if use_preview:
                new_bytes = preview_bytes[1]
                preview_save = os.path.join(preview_save_path, f'preview_{idx+1:05d}.png')
                new_bytes.save(preview_save)
                display_bytes = BytesIO()
                new_bytes.save(display_bytes, format='PNG')
                image_data = display_bytes.getvalue()
                image_widget.value = image_data
        pbar.update_absolute(step + 1, total_steps, preview_bytes)

    frame_index = 0
    svd_pbar = tqdm(total=len(batch_folder),desc='rendering')
    while True:
        frame_name = batch_folder[frame_index]
        frame_path = os.path.join(sdxl_args.init_image_folder_path_for_svd, frame_name)
        frame_index += 1
        print(f'rendering frame: {frame_name}')

        # Load and process the frame
        latent, svd_mask = load_image(frame_path)
        new_svd_model = svd_guidance.patch(svd_model, svd_min_cfg)[0]
        svd_positive, svd_negative, svd_latent = svd_conditioner.encode(svd_clipvision, latent, svd_vae, svd_width, svd_height, svd_video_frames, svd_motion_bucket_id, svd_fps, svd_augmentation_level)
        svd_latent = svd_latent["samples"]

        noise_mask = None
        batch_inds = None
        svd_noise = comfy.sample.prepare_noise(svd_latent, sdxl_args.seed, batch_inds)
        
        svd_samples = comfy.sample.sample(sdxl_args,
                                      new_svd_model, 
                                      svd_noise, 
                                      sdxl_args.steps, 
                                      sdxl_args.cfg, 
                                      svd_sampler, 
                                      svd_scheduler,
                                      svd_positive, 
                                      svd_negative, 
                                      svd_latent, 
                                      denoise=sdxl_args.denoise, 
                                      disable_noise=sdxl_args.refinerdisable_noise, 
                                      start_step=sdxl_args.start_step, 
                                      last_step=sdxl_args.last_step, 
                                      force_full_denoise=sdxl_args.refinerforce_full_denoise, 
                                      noise_mask=noise_mask, 
                                      callback=callback, 
                                      seed=sdxl_args.seed)
        
        count+=1
        images = svd_vae.decode(svd_samples)
        # images = rearrange(images, 'b h w c -> b c h w')
        svd_fps_out = sdxl_args.svd_fps_out
        svd_filename_prefix = sdxl_args.saveprefix
        svd_lossless = sdxl_args.svd_lossless
        svd_quality = sdxl_args.svd_quality
        svd_method = sdxl_args.svd_method
        svd_num_frames = sdxl_args.svd_num_frames
        svd_multiplier = sdxl_args.svd_multiplier
        svd_clear_cache_after_n_frames = sdxl_args.svd_clear_cache_after_n_frames
        film_name = sdxl_args.film_name
        
        film_vfi = FILM_VFI()
        videocombine = VideoCombine()
        
        vfi_images = film_vfi.vfi(film_name,
                                images,
                                clear_cache_after_n_frames=svd_clear_cache_after_n_frames,
                                multiplier=svd_multiplier,
                                optional_interpolation_states = None)
        
        samples = videocombine.combine_video(vfi_images[0],
                                            sdxl_args.svd_fps_out,
                                            sdxl_args.svd_loop_count,
                                            filename_prefix=sdxl_args.saveprefix,
                                            format=sdxl_args.svd_format,
                                            pingpong=False,
                                            save_output=True,
                                            prompt=None,
                                            extra_pnginfo=None,
                                            audio=None,
                                            unique_id=None,
                                            manual_format_widgets=None,
                                            batch_manager=None)
        # animated_webp = svd_saver.save_images(images, svd_fps, svd_filename_prefix, svd_lossless, svd_quality, svd_method, num_frames=svd_num_frames, prompt=None, extra_pnginfo=None)
        get_device_memory()
        svd_pbar.update()
        gc.collect()
        torch.cuda.empty_cache()

    return model, samples
