
import os
import sys
import tempfile
import io
import numpy as np

import streamlit as st
import torch
import dlib

from PIL import Image
from torchvision import transforms
from argparse import Namespace

from skimage.metrics import (
    structural_similarity,
    peak_signal_noise_ratio
)

# --------------------------------------------------
# PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_DIR = "/content/SAM"

CHECKPOINT_PATH = os.path.join(
    PROJECT_DIR,
    "pretrained_models",
    "sam_ffhq_aging.pt"
)

PREDICTOR_PATH = os.path.join(
    PROJECT_DIR,
    "shape_predictor_68_face_landmarks.dat"
)

sys.path.insert(0, PROJECT_DIR)
os.chdir(PROJECT_DIR)

from models.psp import pSp
from scripts.align_all_parallel import align_face


# --------------------------------------------------
# STREAMLIT PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Face Ageing Studio",
    page_icon="🎭",
    layout="wide"
)

st.title("🎭 Face Ageing Studio")

st.write(
    "Explore face ageing and de-ageing using a pretrained "
    "Style-based Age Manipulation (SAM) model."
)

st.info(
    "Upload a clear face photograph, select a target age, "
    "and generate your transformed image."
)


# --------------------------------------------------
# LOAD MODEL AND FACE LANDMARK PREDICTOR
# --------------------------------------------------

@st.cache_resource
def load_models():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"SAM checkpoint not found: {CHECKPOINT_PATH}"
        )

    if not os.path.exists(PREDICTOR_PATH):
        raise FileNotFoundError(
            f"Landmark predictor not found: {PREDICTOR_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False
    )

    opts_dict = dict(checkpoint["opts"])
    opts_dict["checkpoint_path"] = CHECKPOINT_PATH
    opts_dict["device"] = str(device)

    opts = Namespace(**opts_dict)

    model = pSp(opts)
    model.eval()
    model.to(device)

    predictor = dlib.shape_predictor(PREDICTOR_PATH)

    return model, predictor, device


# --------------------------------------------------
# IMAGE PREPROCESSING
# --------------------------------------------------

def preprocess_image(aligned_image):

    image_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5]
        )
    ])

    image_tensor = image_transform(aligned_image)

    return image_tensor.unsqueeze(0)


# --------------------------------------------------
# IMAGE CONVERSION
# --------------------------------------------------

def tensor_to_image(tensor):

    tensor = tensor.detach().float().cpu()

    tensor = tensor[0]
    tensor = (tensor + 1) / 2
    tensor = tensor.clamp(0, 1)

    image = transforms.ToPILImage()(tensor)

    return image


# --------------------------------------------------
# IMAGE DOWNLOAD HELPER
# --------------------------------------------------

def image_bytes(image):

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="PNG"
    )

    return buffer.getvalue()


# --------------------------------------------------
# BASELINE SAM INFERENCE
# --------------------------------------------------

def generate_baseline(model, image_tensor, age, device):

    age_channel = torch.full(
        (1, 1, 256, 256),
        age / 100.0,
        dtype=torch.float32
    )

    model_input = torch.cat(
        [image_tensor, age_channel],
        dim=1
    ).to(device)

    with torch.no_grad():

        _, latent = model(
            model_input,
            randomize_noise=False,
            return_latents=True
        )

        generated, _ = model.decoder(
            [latent],
            input_is_latent=True,
            randomize_noise=False,
            return_latents=True
        )

    return generated, latent


# --------------------------------------------------
# WNNR-INSPIRED LATENT REGULARIZATION
# --------------------------------------------------

def apply_wnnr(latent, latent_avg, strength):

    # Center the latent representation
    centered_latent = (
        latent - latent_avg.unsqueeze(0)
    )

    # Singular Value Decomposition
    U, S, Vh = torch.linalg.svd(
        centered_latent,
        full_matrices=False
    )

    # Weighted singular-value shrinkage
    weights = torch.linspace(
        1.0,
        0.2,
        steps=S.shape[-1],
        device=latent.device,
        dtype=latent.dtype
    )

    modified_S = torch.clamp(
        S - strength * weights.unsqueeze(0),
        min=0
    )

    # Reconstruct modified latent
    modified_centered = (
        U @ torch.diag_embed(modified_S) @ Vh
    )

    modified_latent = (
        modified_centered + latent_avg.unsqueeze(0)
    )

    return modified_latent


# --------------------------------------------------
# DECODE MODIFIED LATENT
# --------------------------------------------------

def decode_latent(model, latent):

    with torch.no_grad():

        generated, _ = model.decoder(
            [latent],
            input_is_latent=True,
            randomize_noise=False,
            return_latents=True
        )

    return generated


# --------------------------------------------------
# CALCULATE SSIM AND PSNR
# --------------------------------------------------

def calculate_image_metrics(
    baseline_image,
    regularized_image
):

    baseline_array = np.array(
        baseline_image
    ).astype(np.uint8)

    regularized_array = np.array(
        regularized_image
    ).astype(np.uint8)

    if baseline_array.shape != regularized_array.shape:
        raise ValueError(
            "Baseline and regularized images must have "
            "the same dimensions for comparison."
        )

    # Structural Similarity Index
    ssim_value = structural_similarity(
        baseline_array,
        regularized_array,
        channel_axis=2,
        data_range=255
    )

    # Peak Signal-to-Noise Ratio
    psnr_value = peak_signal_noise_ratio(
        baseline_array,
        regularized_array,
        data_range=255
    )

    return ssim_value, psnr_value


# --------------------------------------------------
# USER INPUT
# --------------------------------------------------

st.sidebar.header("⚙️ Transformation Settings")

uploaded_file = st.sidebar.file_uploader(
    "Upload a face photograph",
    type=["jpg", "jpeg", "png"]
)

target_age = st.sidebar.slider(
    "Select Target Age",
    min_value=0,
    max_value=100,
    value=50,
    step=1
)

st.sidebar.divider()

use_wnnr = st.sidebar.checkbox(
    "Enable WNNR-inspired regularization",
    value=False
)

regularization_strength = 1.0

if use_wnnr:

    regularization_strength = st.sidebar.slider(
        "Regularization Strength (λ)",
        min_value=0.0,
        max_value=2.0,
        value=1.0,
        step=0.1
    )

generate_button = st.sidebar.button(
    "✨ Generate Face",
    type="primary",
    use_container_width=True
)


# --------------------------------------------------
# APPLICATION EXECUTION
# --------------------------------------------------

if uploaded_file is not None:

    original_image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.subheader("Uploaded Photograph")

    st.image(
        original_image,
        width=300
    )

    if generate_button:

        try:

            with st.spinner(
                "Loading SAM model and preparing your image..."
            ):

                model, predictor, device = load_models()

            st.success(
                f"Model loaded successfully on {device}."
            )

            # Save uploaded image temporarily
            with tempfile.NamedTemporaryFile(
                suffix=".jpg",
                delete=False
            ) as temp_file:

                temp_file.write(
                    uploaded_file.getvalue()
                )

                temp_path = temp_file.name

            try:

                with st.spinner("Aligning face landmarks..."):

                    aligned_image = align_face(
                        filepath=temp_path,
                        predictor=predictor
                    )

            finally:

                if os.path.exists(temp_path):
                    os.remove(temp_path)

            if aligned_image is None:
                raise ValueError(
                    "Could not align the face. Please upload "
                    "a clear, front-facing photograph."
                )

            aligned_image = aligned_image.convert("RGB")

            image_tensor = preprocess_image(
                aligned_image
            )

            st.subheader("Aligned Face")

            st.image(
                aligned_image,
                width=256
            )

            # --------------------------------------
            # BASELINE SAM RESULT
            # --------------------------------------

            with st.spinner(
                f"Generating face at age {target_age}..."
            ):

                baseline_tensor, original_latent = generate_baseline(
                    model,
                    image_tensor,
                    target_age,
                    device
                )

                baseline_image = tensor_to_image(
                    baseline_tensor
                )

            st.divider()

            st.subheader(
                f"Baseline SAM Result — Age {target_age}"
            )

            st.image(
                baseline_image,
                width=450
            )

            st.download_button(
                "⬇️ Download Baseline Result",
                data=image_bytes(baseline_image),
                file_name=f"sam_age_{target_age}.png",
                mime="image/png"
            )

            # --------------------------------------
            # WNNR EXPERIMENT
            # --------------------------------------

            if use_wnnr:

                with st.spinner(
                    "Applying WNNR-inspired latent regularization..."
                ):

                    latent_avg = model.latent_avg.to(device)

                    modified_latent = apply_wnnr(
                        original_latent,
                        latent_avg,
                        regularization_strength
                    )

                    regularized_tensor = decode_latent(
                        model,
                        modified_latent
                    )

                    regularized_image = tensor_to_image(
                        regularized_tensor
                    )

                    # Calculate nuclear norms
                    original_norm = torch.linalg.svdvals(
                        original_latent -
                        latent_avg.unsqueeze(0)
                    ).sum().item()

                    modified_norm = torch.linalg.svdvals(
                        modified_latent -
                        latent_avg.unsqueeze(0)
                    ).sum().item()

                    # Calculate SSIM and PSNR
                    ssim_value, psnr_value = calculate_image_metrics(
                        baseline_image,
                        regularized_image
                    )

                # ----------------------------------
                # IMAGE COMPARISON
                # ----------------------------------

                st.divider()

                st.subheader(
                    "WNNR-Regularized Result"
                )

                col1, col2 = st.columns(2)

                with col1:

                    st.image(
                        baseline_image,
                        caption="Baseline SAM",
                        width=350
                    )

                with col2:

                    st.image(
                        regularized_image,
                        caption="SAM + WNNR-inspired regularization",
                        width=350
                    )

                st.write(
                    f"**Regularization strength (λ):** "
                    f"{regularization_strength}"
                )

                # ----------------------------------
                # NUCLEAR NORM METRICS
                # ----------------------------------

                st.subheader(
                    "Latent Space Regularization Analysis"
                )

                metric1, metric2 = st.columns(2)

                metric1.metric(
                    "Original Nuclear Norm",
                    f"{original_norm:.3f}"
                )

                metric2.metric(
                    "Modified Nuclear Norm",
                    f"{modified_norm:.3f}",
                    delta=f"{modified_norm - original_norm:.3f}"
                )

                # ----------------------------------
                # IMAGE QUALITY METRICS
                # ----------------------------------

                st.divider()

                st.subheader(
                    "Image Quality Evaluation"
                )

                eval_col1, eval_col2 = st.columns(2)

                eval_col1.metric(
                    "SSIM",
                    f"{ssim_value:.5f}"
                )

                eval_col2.metric(
                    "PSNR",
                    f"{psnr_value:.2f} dB"
                )

                st.caption(
                    "SSIM and PSNR compare the regularized output "
                    "against the baseline SAM output. Higher values "
                    "indicate greater similarity, not necessarily "
                    "better image quality or identity preservation."
                )

                # ----------------------------------
                # DOWNLOAD REGULARIZED RESULT
                # ----------------------------------

                st.download_button(
                    "⬇️ Download Regularized Result",
                    data=image_bytes(regularized_image),
                    file_name=f"sam_wnnr_age_{target_age}.png",
                    mime="image/png"
                )

            st.success(
                "Face transformation completed!"
            )

            st.caption(
                "Research demo: WNNR-inspired regularization is "
                "an experimental latent-space modification. "
                "A lower nuclear norm does not by itself prove "
                "better identity preservation or image quality."
            )

        except Exception as error:

            st.error(
                f"An error occurred: {error}"
            )

            st.exception(error)

else:

    st.warning(
        "Please upload a face photograph using the sidebar "
        "to begin."
    )
