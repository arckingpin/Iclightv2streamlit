import os
import math
import streamlit as st
import numpy as np
import torch
import safetensors.torch as sf
from PIL import Image
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    AutoencoderKL,
    UNet2DConditionModel,
    DPMSolverMultistepScheduler,
)
from diffusers.models.attention_processor import AttnProcessor2_0
from transformers import CLIPTextModel, CLIPTokenizer
from rembg import remove  # Using rembg instead of briarmbg for background removal
from torch.hub import download_url_to_file


# Set device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load Models
@st.cache_resource
def load_models():
    model_name = "stablediffusionapi/realistic-vision-v51"
    tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet")

    # Load Stable Diffusion Pipelines
    t2i_pipe = StableDiffusionPipeline.from_pretrained(
        model_name, vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet
    ).to(device)

    i2i_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_name, vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet
    ).to(device)

    return tokenizer, text_encoder, vae, unet, t2i_pipe, i2i_pipe


tokenizer, text_encoder, vae, unet, t2i_pipe, i2i_pipe = load_models()


# Background Removal using rembg
def remove_background(image):
    image = Image.fromarray(image)
    output = remove(image)  # rembg removes background
    output = output.convert("RGBA")
    np_output = np.array(output)
    return np_output[..., :3], np_output[..., 3] / 255.0  # Return image and alpha mask


# Image Processing
@torch.inference_mode()
def generate_image(prompt, image_width, image_height, num_samples, steps, seed, cfg):
    generator = torch.Generator(device=device).manual_seed(seed)
    result = t2i_pipe(
        prompt=prompt,
        width=image_width,
        height=image_height,
        num_inference_steps=steps,
        num_images_per_prompt=num_samples,
        guidance_scale=cfg,
        generator=generator,
    ).images

    return result


# Streamlit UI
st.set_page_config(page_title="IC-Light V2", layout="wide")
st.title("IC-Light V2: Relighting with Stable Diffusion")
st.write("Upload your images, set parameters, and generate high-quality relit images.")

# Upload Images
col1, col2 = st.columns(2)
with col1:
    fg_file = st.file_uploader("Upload Foreground Image", type=["png", "jpg", "jpeg"])
with col2:
    bg_file = st.file_uploader("Upload Background Image", type=["png", "jpg", "jpeg"])

# User Inputs
prompt = st.text_input("Prompt", "A cinematic portrait of a beautiful woman")
num_samples = st.slider("Number of Images", 1, 4, 1)
seed = st.number_input("Seed", value=12345, step=1)
image_width = st.slider("Image Width", 256, 1024, 512, step=64)
image_height = st.slider("Image Height", 256, 1024, 640, step=64)
steps = st.slider("Steps", 1, 100, 20)
cfg = st.slider("CFG Scale", 1.0, 32.0, 7.0)

if st.button("Generate Image"):
    if fg_file:
        input_fg = Image.open(fg_file).convert("RGB")
        fg_np = np.array(input_fg)

        # Remove background
        fg_np, _ = remove_background(fg_np)

        with st.spinner("Generating image..."):
            results = generate_image(prompt, image_width, image_height, num_samples, steps, seed, cfg)

        st.write("Generated Images:")
        for img in results:
            st.image(img, use_column_width=True)
    else:
        st.error("Please upload a foreground image to proceed.")

st.write("Powered by Stable Diffusion & Streamlit")