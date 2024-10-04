import os
import gc
import io
import cv2
import time
import json
import torch
import nodes
import comfy
import base64
import random
import xformers
import importlib
import cuda_malloc
import xformers.ops
import folder_paths
import latent_preview
import numpy as np
from enum import Enum
from io import BytesIO
from PIL import ImageOps
from tqdm.auto import tqdm
from einops import rearrange
from natsort import natsorted
from iprogress import iprogress
from colors import process_image
from comfy import latent_formats
from PIL import Image as pilimage
from comfy import model_management
from PIL import Image as pil_image
from comfy.latent_formats import SDXL
from PIL.PngImagePlugin import PngInfo
from comfy import model_management, sd
from ipywidgets import Image, Layout, VBox
from comfy_extras.nodes_canny import Canny
from ComfyUI_Frame_Interpolation import FILM_VFI
# from custom_nodes.comfy_controlnet_preprocessors.nodes.util import common_annotator_call, img_np_to_tensor, skip_v1
# from custom_nodes.comfy_controlnet_preprocessors.v1 import midas, leres
# from custom_nodes.comfy_controlnet_preprocessors.v11 import zoe, normalbae
from comfy_extras.nodes_images import SaveAnimatedWEBP
from IPython.display import display, clear_output, HTML
from torchvision.transforms.functional import to_pil_image
from ComfyUI_VideoHelperSuite.videohelpersuite.nodes import VideoCombine
from comfy_extras.nodes_upscale_model import UpscaleModelLoader, ImageUpscaleWithModel
from comfy_extras import nodes_flux, nodes_custom_sampler, nodes_model_advanced, nodes_sd3, nodes_clip_sdxl
from comfy_extras.nodes_video_model import ImageOnlyCheckpointLoader, SVD_img2vid_Conditioning, VideoLinearCFGGuidance


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

# # Load the comfy models
# comfy_upscaler = UpscaleModelLoader()
# loaded_upscaler = comfy_upscaler.load_model("4xLexicaHAT.pth")
# upscaler = ImageUpscaleWithModel()
# sd = comfy.utils.load_torch_file("/workspace/ComfyUI-Notebook/models/vae/sdxl_vae.safetensors")
# vae = comfy.sd.VAE(sd=sd)
# sharpened_upscaler = comfy_upscaler.load_model("OmniSR_X4_DF2K_epoch994.pth")

# Define a function to upscale an image
def upscale_image(image_path,loaded_upscaler):
        # image_path = folder_paths.get_annotated_filepath(image)
    i = ImageOps.exif_transpose(image_path)
    image = i.convert("RGB")
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image)[None,]
    if 'A' in i.getbands():
        mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
        mask = 1. - torch.from_numpy(mask)
    else:
        mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")

    with torch.inference_mode():
        upscale_model = loaded_upscaler[0]
        samples = image
        vaeimage = upscaler.upscale(upscale_model, samples)
        vaeimage = rearrange(vaeimage[0], 'b h w c -> b c h w')
        vaeimage = vaeimage.squeeze(0)
        vaeimage = to_pil_image(vaeimage)
        return vaeimage

def upscale_samples(sdxl_args, video_path, output_folder):
    force_resize = sdxl_args.force_resize
    sharpen = sdxl_args.sharpen
    # Video file path
    # Output folder path
    os.makedirs(output_folder, exist_ok=True)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if the video opened successfully
    if not cap.isOpened():
        print("Error opening video file")
        sys.exit(1)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    total_pixels = int(frame_width*frame_height)
    
    max_allowed_pixels = 1024*704
    
    if total_pixels>max_allowed_pixels or force_resize:
        optimized_framewidth = frame_width - 256
        optimized_frameheight = frame_height - 256
    
    final_frame_width = frame_width*4
    final_frame_height = frame_height*4
    
    # Output video path
    output_video_path = os.path.join(output_folder, f"upscaled_{os.path.basename(video_path)}")
    print(output_video_path)
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (final_frame_width, final_frame_height))
    
    # Process each frame
    for frame_num in tqdm(range(frame_count)):
        ret, frame = cap.read()
        if total_pixels>max_allowed_pixels or force_resize:
            frame = cv2.resize(frame, (optimized_framewidth, optimized_frameheight), interpolation=cv2.INTER_LANCZOS4)
        if not ret:
            break
    
        # Convert the frame to a PIL image
        pil_frame = pil_image.fromarray(frame)
    
        # Upscale the frame
        upscaled_frame = upscale_image(pil_frame,loaded_upscaler)
        upscaled_frame = upscaled_frame.convert("RGB")  # This removes the alpha channel
    
        # Convert the upscaled frame back to a numpy array
        upscaled_frame_np = cv2.cvtColor(np.array(upscaled_frame), cv2.COLOR_RGB2BGR)
        
        cv2_image = cv2.cvtColor(upscaled_frame_np, cv2.COLOR_RGB2BGR)
        cv2_image = cv2_image.astype(np.uint8)
        if total_pixels>max_allowed_pixels or force_resize:
            cv2_image = cv2.resize(cv2_image, (final_frame_width, final_frame_height), interpolation=cv2.INTER_LANCZOS4)
        if sharpen:
            print('sharpening')
            cv2_image = cv2.resize(cv2_image,(optimized_framewidth, optimized_frameheight), interpolation=cv2.INTER_LANCZOS4)
            pil_frame = pil_image.fromarray(cv2_image)
            upscaled_image = upscale_image(pil_frame,sharpened_upscaler)
            
            # Upscale the frame
            upscaled_frame = upscaled_image.convert("RGB")  # This removes the alpha channel
    
            # Convert the upscaled frame back to a numpy array
            upscaled_frame_np = cv2.cvtColor(np.array(upscaled_frame), cv2.COLOR_RGB2BGR)
    
            cv2_image = cv2.cvtColor(upscaled_frame_np, cv2.COLOR_RGB2BGR)
            cv2_image = cv2_image.astype(np.uint8)
            cv2_image = cv2.resize(cv2_image,(final_frame_width, final_frame_height), interpolation=cv2.INTER_LANCZOS4)
        # Write the upscaled frame to the output video
        out.write(np.array(cv2_image))
    
    # Release resources
    cap.release()
    out.release()
    return output_video_path


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

    #Lora Loading
    if sdxl_args.lora_name is not None:
        if isinstance(sdxl_args.lora_name, dict):
            strength_model_clip = None
            # If it's a dictionary, iterate through the items and load each one
            for lora_item, strength_model_clip in sdxl_args.lora_name.items():
                print(f'Multiple Loras detected, loading {lora_item}')
                # Get the strengths for the current lora_name or use defaults
                strength_model = strength_model_clip
                strength_clip = strength_model_clip
                print(f'running lora with strength_model: {strength_model}, strength_clip: {strength_clip}')    
                model, clip = load_lora(model, clip, lora_item, strength_model, strength_clip)
        elif isinstance(sdxl_args.lora_name, str):
            # If it's a string, load only that string
            model, clip = load_lora(model, clip, sdxl_args.lora_name, sdxl_args.strength_model, sdxl_args.strength_clip)

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

    if sdxl_args.imageheight > sdxl_args.imagewidth:
            svd_height = 1024
            svd_width = 576
    else:
        svd_height = 576
        svd_width = 1024

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

    outputimages = []
    for im in vaeimage:
        im = to_pil_image(im)
        new_im = im
        bytes_image = BytesIO()
        new_im.save(bytes_image, format='PNG')
        image_data = bytes_image.getvalue()
        image_widget = Image(value=image_data, format='png')
        vbox = VBox([image_widget], layout=Layout(width="512px"))
        im.resize((svd_width, svd_height), pilimage.Resampling.LANCZOS)
        display(vbox)

        image_widget1 = Image()
        vbox1 = VBox([image_widget1], layout=Layout(width="256px"))
        display(vbox1)
        display_bytes1 = BytesIO()
        new_im.save(display_bytes1, format='PNG')
        image_data1 = display_bytes1.getvalue()
        image_widget1.value = image_data1
        outputimages.append(new_im)

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
        if sdxl_args.use_init_image:
            print('RUNNING SVD WITH IMAGE INIT')
            new_image, new_mask = load_image(sdxl_args.svd_init_image)
            latent = vae.encode(new_image)
        new_svd_model = svd_guidance.patch(svd_model, svd_min_cfg)[0]
        svd_positive, svd_negative, svd_latent = svd_conditioner.encode(svd_clipvision, latent, svd_vae, svd_width, svd_height, svd_video_frames, svd_motion_bucket_id, svd_fps, svd_augmentation_level)
        svd_latent = svd_latent["samples"]
        svd_noise = comfy.sample.prepare_noise(svd_latent, sdxl_args.seed, batch_inds)
        svd_sampler = sdxl_args.svd_sampler
        svd_scheduler = sdxl_args.svd_scheduler
        
        svd_samples = comfy.sample.sample(sdxl_args,
                                      new_svd_model, 
                                      svd_noise, 
                                      sdxl_args.svd_steps, 
                                      sdxl_args.svd_cfg, 
                                      svd_sampler, 
                                      svd_scheduler,
                                      svd_positive, 
                                      svd_negative, 
                                      svd_latent, 
                                      denoise=sdxl_args.denoise, 
                                      disable_noise=sdxl_args.refinerdisable_noise, 
                                      start_step=sdxl_args.svd_start_step, 
                                      last_step=sdxl_args.svd_last_step, 
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
                                            manual_format_widgets={'pix_fmt': 'yuv420p', 'crf': 17, 'save_metadata': True},
                                            batch_manager=None)
        
        video_display_list = os.listdir(os.path.join(os.path.dirname(__file__), "output"))
        video_display_path = os.path.join(os.path.dirname(__file__), f"output/{video_display_list[-1]}")
        if sdxl_args.upscaled_output:
            video_display_path = upscale_samples(sdxl_args, video_display_path, output_folder)
        
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
        gc.collect()
        torch.cuda.empty_cache()

        # animated_webp = svd_saver.save_images(images, svd_fps, svd_filename_prefix, svd_lossless, svd_quality, svd_method, num_frames=svd_num_frames, prompt=None, extra_pnginfo=None)
    get_device_memory()

    return model, samples, outputimages[0]

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
                # new_bytes.save(preview_save)
                display_bytes = BytesIO()
                new_bytes.save(display_bytes, format='PNG')
                image_data = display_bytes.getvalue()
                image_widget.value = image_data
        pbar.update_absolute(step + 1, total_steps, preview_bytes)

    frame_index = 0
    svd_pbar = tqdm(total=len(batch_folder),desc='rendering')
    while True:
        if frame_index >= len(batch_folder):
            print("No more images in the folder.")
            break
        else:
            frame_name = batch_folder[frame_index]
            frame_path = os.path.join(sdxl_args.init_image_folder_path_for_svd, frame_name)
            frame_index += 1
            print(f'rendering frame: {frame_name}')

        image_info = cv2.imread(frame_path)
        imageheight, imagewidth, _ = image_info.shape

        if imageheight > imagewidth:
            svd_height = 1024
            svd_width = 576
        else:
            svd_height = 576
            svd_height = 1024
            

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
                                      sdxl_args.svd_steps, 
                                      sdxl_args.svd_cfg, 
                                      svd_sampler, 
                                      svd_scheduler,
                                      svd_positive, 
                                      svd_negative, 
                                      svd_latent, 
                                      denoise=sdxl_args.denoise, 
                                      disable_noise=sdxl_args.refinerdisable_noise, 
                                      start_step=sdxl_args.svd_start_step, 
                                      last_step=sdxl_args.svd_last_step, 
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

    return new_svd_model, svd_samples, images, vfi_images[0]
    
def animate_svd(sdxl_args):
    #Initiate Vars For SVD
    svd_conditioner = SVD_img2vid_Conditioning()
    svd_guidance = VideoLinearCFGGuidance()
    svd_saver = SaveAnimatedWEBP()
    svd_model, svd_clipvision, svd_vae = sdxl_args.svd_loaded
    clear_output(wait=True)
    # svd_ckpt_name = sdxl_args.svd_ckpt_name
    # svd_min_cfg = sdxl_args.svd_min_cfg
    # svd_width = sdxl_args.svd_width
    # svd_height = sdxl_args.svd_height
    # svd_video_frames = sdxl_args.svd_video_frames
    # svd_motion_bucket_id = sdxl_args.svd_motion_bucket_id
    # svd_fps = sdxl_args. svd_fps
    # svd_augmentation_level = sdxl_args.svd_augmentation_level
    # svd_sampler = sdxl_args.svd_sampler
    # svd_scheduler = sdxl_args.svd_scheduler
    # new_svd_model = svd_guidance.patch(svd_model, svd_min_cfg)[0]
    # svd_denoise = 1.00
    
    #Preview Vars/Ops
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

    #Progress Bar
    pbar = comfy.utils.ProgressBar(sdxl_args.steps)
    
    #Display Ops
    image_widget = Image()
    vbox = VBox([image_widget], layout=Layout(width="256px"))
    display(vbox)

    #Output Folder Ops
    output_folder = sdxl_args.output_folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    #Output Folder Count
    count = len(os.listdir(output_folder))
    
    #Initiate Preview Saves
    preview_save_path = os.path.join(sdxl_args.output_folder, f'{sdxl_args.saveprefix}_{count+1:05d}')
    if not os.path.exists(preview_save_path):
        os.makedirs(preview_save_path, exist_ok=True)

    #Batch Folder From Which to Load the Images From
    batch_folder = sorted([f for f in os.listdir(sdxl_args.init_image_folder_path_for_svd) if f.lower().endswith(('.png', '.jpeg', '.jpg'))])

    #Callback Func for Sampling Progress Display
    def callback(step, x0, x, total_steps):
        preview_bytes = None
        idx = len(os.listdir(preview_save_path))
        if previewer:
            preview_bytes = previewer.decode_latent_to_preview_image(preview_format, x0)
            if use_preview:
                new_bytes = preview_bytes[1]
                preview_save = os.path.join(preview_save_path, f'preview_{idx+1:05d}.png')
                # new_bytes.save(preview_save)
                display_bytes = BytesIO()
                new_bytes.save(display_bytes, format='PNG')
                image_data = display_bytes.getvalue()
                image_widget.value = image_data
        pbar.update_absolute(step + 1, total_steps, preview_bytes)

    #Temp Folder and Vars for Saving Operated Image
    outpath = "/workspace/tmprun/tmpimages"
    os.makedirs(outpath, exist_ok=True)
    outlist = os.listdir(outpath)
    pathlist = [f for f in outlist if f.endswith(".png")]
    count = len(pathlist)

    #Initiate Different Indexes for Ops in the Loop
    frame_index = 0
    idx = 0
    last_frame_index = None
    svd_pbar = tqdm(total=len(batch_folder),desc='rendering')

    #Load VAE to Encode and Decode Generated Image
    sd = comfy.utils.load_torch_file("/workspace/ComfyUI-Notebook/models/vae/sdxl_vae.safetensors")
    vae = comfy.sd.VAE(sd=sd)

    #Loop Initiation
    while True:
        svd_ckpt_name = sdxl_args.svd_ckpt_name
        svd_min_cfg = sdxl_args.svd_min_cfg
        svd_width = sdxl_args.svd_width
        svd_height = sdxl_args.svd_height
        svd_video_frames = sdxl_args.svd_video_frames
        svd_motion_bucket_id = sdxl_args.svd_motion_bucket_id
        svd_fps = sdxl_args.svd_fps
        svd_augmentation_level = sdxl_args.svd_augmentation_level
        svd_sampler = sdxl_args.svd_sampler
        svd_scheduler = sdxl_args.svd_scheduler
        new_svd_model = svd_guidance.patch(svd_model, svd_min_cfg)[0]
        svd_denoise = 1.00
        #Set Seed for Every Iteration
        svd_seed = seed_everything(torch.randint(0, 2**32 - 1, (1,)).item())
        index_incrementer = 3

        # Handle prompt logic
        if isinstance(sdxl_args.prompt, list):
            flux_prompt = sdxl_args.prompt[(frame_index // index_incrementer) % len(sdxl_args.prompt)]  # Use the same prompt for 7 idxes
        else:
            flux_prompt = sdxl_args.prompt  # Use the same prompt if it's a single string

        print(f"Running with Prompt: {flux_prompt} on Index: {frame_index}")

        #First Frame Initiation        
        if last_frame_index is None:
            frame_name = batch_folder[frame_index]
            frame_path = os.path.join(sdxl_args.init_image_folder_path_for_svd, frame_name)
            idx+=1
            print(f'rendering frame: {frame_name}')
            image_info = cv2.imread(frame_path)
            imageheight, imagewidth, _ = image_info.shape
            flux_denoise = 1.00
            flux_cfg = 2.5
            
        #Continuing The Loop with the last image
        else:
            pil_last_image = np.array(last_frame_index)
            imageheight, imagewidth, _ = pil_last_image.shape
            idx-=1
            flux_denoise = 1.00
            flux_cfg = 3.5
        
        frame_index+=1

        if frame_index % index_incrementer == 0:
            print(f"Running with Full Denoise to Add Detail on Frame Index: {frame_index}")
            flux_denoise = 1.00
            
        #Set Image Dimensions For SVD Based on the Input Image
        if imageheight > imagewidth:
            svd_height = 1024
            svd_width = 576
            flux_height = 1344
            flux_width = 768
        else:
            svd_height = 576
            svd_width = 1024
            flux_height = 768
            flux_width = 1344
            
        # Load and process the frame
        if last_frame_index is None:
            latent, svd_mask = load_image(frame_path)
            last_frame_index_frame_path = os.path.join(outpath, f'{idx:05d}_.png')
        else:
            print(f'Trying to load: {last_frame_index_frame_path}')
            print(f"Running with Index: {idx}")
            last_frame_name = pathlist[idx]
            last_frame_index_frame_path = os.path.join(outpath, last_frame_name)
            latent, svd_mask = load_image(last_frame_index_frame_path)
            idx+=1

        svd_positive, svd_negative, svd_latent = svd_conditioner.encode(svd_clipvision, latent, svd_vae, svd_width, svd_height, svd_video_frames, svd_motion_bucket_id, svd_fps, svd_augmentation_level)
        svd_latent = svd_latent["samples"]
    
        noise_mask = None
        batch_inds = None
        svd_noise = comfy.sample.prepare_noise(svd_latent, svd_seed, batch_inds)
        
        svd_samples = comfy.sample.sample(sdxl_args,
                                      new_svd_model, 
                                      svd_noise, 
                                      sdxl_args.svd_steps, 
                                      sdxl_args.svd_cfg, 
                                      svd_sampler, 
                                      svd_scheduler,
                                      svd_positive, 
                                      svd_negative, 
                                      svd_latent, 
                                      denoise=svd_denoise, 
                                      disable_noise=sdxl_args.refinerdisable_noise, 
                                      start_step=sdxl_args.svd_start_step, 
                                      last_step=sdxl_args.svd_last_step, 
                                      force_full_denoise=sdxl_args.refinerforce_full_denoise, 
                                      noise_mask=noise_mask, 
                                      callback=callback, 
                                      seed=svd_seed)
        
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

        # if last_frame_index is None:
        last_frame_index = images[-1]
        # else:
            # last_frame_index = images[0]
        print(f'Last Frame 1 Index Type: {type(last_frame_index)}')
        print(f'Last Frame 1 Index Shape: {last_frame_index.shape}')
        
        last_frame_index = last_frame_index.permute(2, 0, 1)  # [C, H, W]

        print(f'Last Frame 3 Index Type: {type(last_frame_index)}')
        print(f'Last Frame 3 Index Shape: {last_frame_index.shape}')
        last_frame_index = to_pil_image(last_frame_index)

        last_frame_index.save(last_frame_index_frame_path, format="PNG")
        print(f'Saved frame to: {last_frame_index_frame_path}')

        use_flux = True
        use_sdxl = not use_flux
        match_colors = False
        if use_flux:                
            if match_colors:
                print("Matching Colors for Temporal Consistency")
                sample_alpha = 1.00 if not frame_index % index_incrementer == 0 else 0.70
                sample_image_path = "/workspace/1345723.png"
                last_frame_index = process_image(last_frame_index_frame_path, sample_image_path, sample_alpha, "HM-MVGD-HM", frame_index)
                last_frame_index.save(last_frame_index_frame_path, format="PNG")
            print("Sampling the Image for Quality")
            last_frame_index = run_flux(flux_prompt, sdxl_args, width=flux_width, 
                                        height=flux_height, input_latent=last_frame_index_frame_path, cfg=flux_cfg, 
                                        flux_base_guidance=1.15, flux_min_guidance=0.5, seed=svd_seed, flux_denoise=flux_denoise,
                                       flux_steps=30, flux_sampler="euler")

            last_frame_index.resize((svd_width, svd_height), pilimage.Resampling.LANCZOS)
            last_frame_index.save(last_frame_index_frame_path, format="PNG")
            
        image_widget1 = Image()
        vbox1 = VBox([image_widget1], layout=Layout(width="256px"))
        display(vbox1)
        display_bytes1 = BytesIO()
        last_frame_index.save(display_bytes1, format='PNG')
        image_data1 = display_bytes1.getvalue()
        image_widget1.value = image_data1

        # clear_output(wait=True)

        # del svd_clipvision
        # del svd_vae
        # del new_svd_model
        # del svd_noise
        # del vae
        # del svd_samples, images
        # del new_svd_model, svd_positive, svd_negative, svd_latent, svd_noise, svd_samples, images

    # return new_svd_model, svd_samples, images, vfi_images[0]
    return vfi_images[0]

def seed_everything(seed, deterministic=False):
    print(f'Set global seed to {seed}')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed

def run_flux(prompt, sdxl_args, width=768, height=1344, input_latent=None, unet_name="flux1-dev.safetensors",
             clip_l="clip_l.safetensors", t5text="t5xxl_fp16.safetensors", weight_dtype="default",
             clip_dir='/content/flux_outputs', seed=torch.randint(0, 2**32 - 1, (1,)).item(), cfg=3.5,
             flux_base_guidance=1.15, flux_min_guidance=0.5, flux_denoise=1.00, flux_steps=20, flux_sampler="euler"):
    # Set seed if needed
    if seed == -1:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()
    seed = seed_everything(seed)

    # Load UNet and apply patches
    flux_model = nodes.UNETLoader().load_unet(unet_name, weight_dtype)
    flux_model = flux_model[0]
    flux_model = nodes_model_advanced.ModelSamplingFlux().patch(flux_model, flux_base_guidance, flux_min_guidance, width, height)
    flux_model = flux_model[0]

    # Load clip model
    clip = nodes.DualCLIPLoader().load_clip(t5text, clip_l, "flux")
    clip = clip[0]

    # Encode the text prompt
    clip_text_encoding = nodes.CLIPTextEncode().encode(clip, prompt)
    clip_text_encoding = clip_text_encoding[0]
    negative_clip_text_encoding = nodes.CLIPTextEncode().encode(clip, "text")
    negative_clip_text_encoding = negative_clip_text_encoding[0]

    # Load VAE
    vae = nodes.VAELoader().load_vae("ae.safetensors")
    vae = vae[0]

    # Generate latent image
    if input_latent is None:
        latentempty = nodes_sd3.EmptySD3LatentImage().generate(width, height, batch_size=1)
        latent = latentempty[0]
    else:
        emptylatent = {}
        latent, _ = load_image(input_latent)
        controlnet_latent = latent
        latent = vae.encode(latent)
        emptylatent["samples"] = latent
        latent = emptylatent

    run_controlnet = True
    if run_controlnet and input_latent is not None:
        print("Running ControlNet")
        control_net = nodes.ControlNetLoader().load_controlnet("flux-canny-controlnet-v3.safetensors")
        control_net = control_net[0]
        controlnet_latent = pilimage.open(input_latent)
        controlnet_latent.resize((width, height), pilimage.Resampling.LANCZOS)
        controlnet_latent.save(input_latent, format="PNG")
        controlnet_latent, _ = load_image(input_latent)
        clip_text_encoding = run_flux_controlnet(clip_text_encoding, negative_clip_text_encoding, control_net, controlnet_latent, vae)

    # clip_text_encoding = nodes_flux.FluxGuidance().append(clip_text_encoding, cfg)
    # clip_text_encoding = clip_text_encoding[0]
    # Get sigmas for sampling
    sigmas = nodes_custom_sampler.BasicScheduler().get_sigmas(flux_model, "simple", flux_steps, flux_denoise)
    sigmas = sigmas[0]

    # Apply guidance
    print("Running with Exposed Guider")
    guider = comfy.samplers.CFGGuider(sdxl_args, flux_model)
    guider.inner_set_conds({"positive": clip_text_encoding})

    # Select sampler and generate noise
    print(f"Running with Sampler: {flux_sampler}")
    sampler = comfy.samplers.sampler_object(sdxl_args, flux_sampler)
    noise = nodes_custom_sampler.Noise_RandomNoise(seed).generate_noise(latent)

    # Create output directory if it doesn't exist
    if not os.path.exists(clip_dir):
        os.makedirs(clip_dir, exist_ok=True)
    
    name_length = len([f for f in os.listdir(clip_dir) if os.path.isfile(os.path.join(clip_dir, f))])
    flux_outpath = os.path.join(clip_dir, f"flux_{name_length}.png")

    # Sampling and image generation
    samples = nodes_custom_sampler.SamplerCustomAdvanced().sample(
        nodes_custom_sampler.Noise_RandomNoise(seed), guider, sampler, sigmas, latent
    )
    output_samples = samples[0]["samples"]

    # Clear cache
    gc.collect()
    torch.cuda.empty_cache()

    x = output_samples.cuda()

    # Decode the latent samples to an image
    image = vae.decode(x)
    vaeimage = rearrange(image, 'b h w c -> b c h w')

    output_images = []
    for batch_number, sample in enumerate(vaeimage):
        img = to_pil_image(sample)
        bytes_image = BytesIO()
        img.save(bytes_image, format='PNG')
        img.save(flux_outpath)
        output_images.append(img)

    del flux_model, clip, clip_text_encoding, latent, sigmas, guider, noise, vae, seed, cfg
    
    return output_images[0]

def run_flux_controlnet(clip_text_encoding, negative_clip_text_encoding, control_net, input_image, vae):
    image = Canny().detect_edge(input_image, 0.2, 0.4)
    image = image[0]
    clip_text_encoding, _ = nodes.ControlNetApplyAdvanced().apply_controlnet(clip_text_encoding, negative_clip_text_encoding, control_net, image, 0.6, 0.00, 1.00, vae=vae)
    return clip_text_encoding