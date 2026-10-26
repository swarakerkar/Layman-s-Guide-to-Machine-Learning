r"""
File 07 — Final Streamlit Dashboard
Handwriting Analysis | EfficientNet-B0 + SHAP

Run from Anaconda Prompt / terminal:
    cd "C:\Users\SWARA\Documents\Handwriting Analysis"
    streamlit run 07_dashboard.py

IMPORTANT:
- This assumes a torchvision EfficientNet-B0 with a single-output regression head.
- Confirm the normalization, patch size/stride, output scaling, and aggregation used during training.
- The default patch strategy below uses 224x224 patches and averages their predictions.
  Change PATCH_SIZE / PATCH_STRIDE below to match the training pipeline exactly.
"""

from pathlib import Path
import json
import math
import warnings

import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageEnhance
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
import matplotlib.pyplot as plt

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False


# ---------------------------------------------------------------------
# 1. CONFIGURATION — check these against your training notebook
# ---------------------------------------------------------------------
PROJECT_DIR = Path(r"C:\Users\SWARA\Documents\Handwriting Analysis")
MODEL_PATH = PROJECT_DIR / "models" / "best_efficientnet_b0.pth"
OUTPUT_DIR = PROJECT_DIR / "dashboard_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PATCH_SIZE = 224
PATCH_STRIDE = 224  # Change to the stride used during training if overlapping patches were used.
INPUT_SIZE = 224

# This must match training-time normalization.
USE_IMAGENET_NORMALIZATION = True

# If your training targets were scaled (e.g. 0–1 instead of 0–10), configure this
# after checking the training notebook. Defaults do NOT rescale model outputs.
SCORE_MULTIPLIER = 1.0
SCORE_OFFSET = 0.0
DISPLAY_SCORE_MIN = None  # Example: 0.0; leave None if unknown
DISPLAY_SCORE_MAX = None  # Example: 10.0; leave None if unknown

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------
# 2. PAGE CONFIG + STYLE
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Handwriting Analysis | EfficientNet-B0",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .stApp { background: #f7f8fc; }
      .block-container { padding-top: 1.6rem; padding-bottom: 2rem; }
      .hero {
        padding: 1.5rem 1.7rem; border-radius: 18px;
        background: linear-gradient(120deg, #172554 0%, #4338ca 55%, #7c3aed 100%);
        color: white; margin-bottom: 1rem;
      }
      .hero h1 { color: white; margin: 0; font-size: 2.1rem; }
      .hero p { color: #e0e7ff; margin: .45rem 0 0 0; font-size: 1rem; }
      .small-note { color: #64748b; font-size: .88rem; }
      div[data-testid="stMetric"] {
        background: white; border: 1px solid #e2e8f0;
        padding: 14px 16px; border-radius: 14px;
      }
      section[data-testid="stSidebar"] { background: #eef2ff; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>✍️ Handwriting Analysis</h1>
      <p>AI-powered handwriting quality assessment with interpretable predictions</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# 3. MODEL LOADING
# ---------------------------------------------------------------------
def extract_state_dict(checkpoint):
    """Support common checkpoint layouts without assuming every checkpoint is identical."""
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict) and value and all(
                isinstance(k, str) for k in value.keys()
            ):
                return value
        if checkpoint and all(isinstance(k, str) for k in checkpoint.keys()):
            # A raw state_dict is usually a dict of parameter names -> tensors.
            if all(torch.is_tensor(v) for v in checkpoint.values()):
                return checkpoint
    raise ValueError(
        "Could not identify a model state_dict in the checkpoint. "
        "Inspect the checkpoint structure and adapt extract_state_dict()."
    )


@st.cache_resource(show_spinner="Loading EfficientNet-B0 model...")
def load_model(model_path_string: str):
    model_path = Path(model_path_string)
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found:\n{model_path}")

    # weights_only=True is safer for state-dict checkpoints on supported PyTorch versions.
    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(model_path, map_location="cpu")

    state_dict = extract_state_dict(checkpoint)
    state_dict = {
        key.removeprefix("module."): value for key, value in state_dict.items()
    }

    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint keys do not match the assumed EfficientNet-B0 regression "
            "architecture. Recreate the exact model head used in training.\n\n"
            f"Original error: {exc}"
        ) from exc

    model.to(DEVICE)
    model.eval()
    return model


def get_transform():
    ops = [transforms.Resize((INPUT_SIZE, INPUT_SIZE)), transforms.ToTensor()]
    if USE_IMAGENET_NORMALIZATION:
        ops.append(
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        )
    return transforms.Compose(ops)


# ---------------------------------------------------------------------
# 4. IMAGE / PATCH HELPERS
# ---------------------------------------------------------------------
def prepare_image(uploaded_file):
    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image).convert("RGB")
    return image


def make_patch_positions(length, patch_size, stride):
    if length <= patch_size:
        return [0]
    positions = list(range(0, length - patch_size + 1, stride))
    last = length - patch_size
    if positions[-1] != last:
        positions.append(last)
    return positions


def image_to_patches(image: Image.Image, patch_size=PATCH_SIZE, stride=PATCH_STRIDE):
    """Create square patches. Small images are resized to one patch; large images are tiled."""
    width, height = image.size
    if width < patch_size or height < patch_size:
        return [image.resize((patch_size, patch_size), Image.Resampling.LANCZOS)]

    xs = make_patch_positions(width, patch_size, stride)
    ys = make_patch_positions(height, patch_size, stride)
    patches = []
    for y in ys:
        for x in xs:
            patches.append(image.crop((x, y, x + patch_size, y + patch_size)))
    return patches


def predict_patches(model, patches):
    transform = get_transform()
    batch = torch.stack([transform(patch) for patch in patches]).to(DEVICE)
    with torch.inference_mode():
        outputs = model(batch).reshape(-1).detach().cpu().numpy()
    outputs = outputs * SCORE_MULTIPLIER + SCORE_OFFSET
    return outputs.astype(float)


def format_score(score):
    if DISPLAY_SCORE_MIN is not None and DISPLAY_SCORE_MAX is not None:
        score = float(np.clip(score, DISPLAY_SCORE_MIN, DISPLAY_SCORE_MAX))
    return score


def make_shap_predictor(model):
    """SHAP passes batches of RGB uint8 images; return one scalar per image."""
    transform = get_transform()

    def predictor(images):
        converted = []
        for arr in images:
            arr = np.asarray(arr)
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            converted.append(transform(Image.fromarray(arr).convert("RGB")))
        batch = torch.stack(converted).to(DEVICE)
        with torch.inference_mode():
            values = model(batch).reshape(-1).detach().cpu().numpy()
        return values * SCORE_MULTIPLIER + SCORE_OFFSET

    return predictor


# ---------------------------------------------------------------------
# 5. SIDEBAR
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Choose a section",
        ["Dashboard", "Assess Handwriting", "Explain Prediction (SHAP)", "About the Model"],
    )
    st.divider()
    st.caption("Project model")
    st.write("**EfficientNet-B0**")
    st.write("**Explainability:** SHAP")
    st.caption(f"Device: `{DEVICE}`")
    st.caption("For research/educational use. Predictions depend on training data and preprocessing.")


# ---------------------------------------------------------------------
# 6. LOAD MODEL WHEN NEEDED
# ---------------------------------------------------------------------
def model_status():
    if not MODEL_PATH.exists():
        st.error(
            "Model checkpoint was not found. Expected location:\n\n"
            f"`{MODEL_PATH}`\n\n"
            "Confirm that `best_efficientnet_b0.pth` exists at this path."
        )
        return None
    try:
        return load_model(str(MODEL_PATH))
    except Exception as exc:
        st.error("The model could not be loaded.")
        st.exception(exc)
        return None


# ---------------------------------------------------------------------
# 7. DASHBOARD HOME
# ---------------------------------------------------------------------
if page == "Dashboard":
    st.subheader("Project Overview")
    st.write(
        "This application uses a trained EfficientNet-B0 regression model to estimate "
        "handwriting quality from an uploaded image. SHAP can be used to inspect which "
        "image regions influence a selected patch prediction."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Deep learning model", "EfficientNet-B0")
    c2.metric("Input strategy", f"{PATCH_SIZE} × {PATCH_SIZE} patches")
    c3.metric("Explainability", "SHAP")

    st.markdown("### Workflow")
    st.markdown(
        """
        1. **Upload** a clear image of handwritten text.
        2. **Assess** the image using the trained model.
        3. Review the patch-level predictions and their average.
        4. Use **SHAP** to inspect a selected patch's contribution map.
        """
    )
    st.info(
        "Before reporting results, verify that patch size, stride, normalization, "
        "score scaling, and page-level aggregation match the training notebook."
    )
    st.markdown("### Checkpoint")
    if MODEL_PATH.exists():
        st.success(f"Checkpoint found: `{MODEL_PATH}`")
    else:
        st.warning(f"Checkpoint not found at `{MODEL_PATH}`")


# ---------------------------------------------------------------------
# 8. ASSESS HANDWRITING
# ---------------------------------------------------------------------
elif page == "Assess Handwriting":
    st.subheader("Handwriting Quality Assessment")
    st.write("Upload a JPG, JPEG, or PNG image of handwritten text.")

    uploaded = st.file_uploader(
        "Choose a handwriting image",
        type=["jpg", "jpeg", "png"],
        key="assessment_upload",
    )

    if uploaded:
        try:
            image = prepare_image(uploaded)
            left, right = st.columns([1, 1])
            with left:
                st.image(image, caption="Uploaded handwriting", use_container_width=True)
                st.caption(f"Image dimensions: {image.width} × {image.height} pixels")

            with right:
                st.markdown("#### Image settings")
                st.write(f"Patch size: **{PATCH_SIZE} × {PATCH_SIZE}**")
                st.write(f"Patch stride: **{PATCH_STRIDE}**")
                st.write(
                    "Normalization: **ImageNet**"
                    if USE_IMAGENET_NORMALIZATION
                    else "Normalization: **None**"
                )
                st.caption(
                    "These settings are configuration defaults. They must match the "
                    "settings used when the model was trained."
                )

            if st.button("Assess handwriting", type="primary", use_container_width=True):
                model = model_status()
                if model is not None:
                    with st.spinner("Preparing image patches and predicting..."):
                        patches = image_to_patches(image)
                        predictions = predict_patches(model, patches)
                        score = format_score(float(np.mean(predictions)))

                    st.session_state["last_image"] = image
                    st.session_state["last_patches"] = patches
                    st.session_state["last_predictions"] = predictions
                    st.session_state["last_score"] = score

                    st.success("Assessment complete.")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Average predicted score", f"{score:.4f}")
                    m2.metric("Patches analysed", str(len(patches)))
                    m3.metric("Patch score spread", f"{np.std(predictions):.4f}")

                    st.caption(
                        "The score is the model's raw regression output after the configured "
                        "multiplier/offset. Do not label it as a 0–10 score unless that is "
                        "the scale used during training."
                    )

                    with st.expander("View patch-level predictions"):
                        results = pd.DataFrame(
                            {
                                "Patch": np.arange(1, len(predictions) + 1),
                                "Predicted score": predictions,
                            }
                        )
                        st.dataframe(results, use_container_width=True, hide_index=True)

                    fig, ax = plt.subplots(figsize=(8, 3.5))
                    ax.plot(np.arange(1, len(predictions) + 1), predictions, marker="o")
                    ax.axhline(np.mean(predictions), linestyle="--", label="Mean")
                    ax.set_xlabel("Patch number")
                    ax.set_ylabel("Predicted regression output")
                    ax.set_title("Patch-level predictions")
                    ax.legend()
                    ax.grid(alpha=0.25)
                    st.pyplot(fig)
                    plt.close(fig)

                    st.download_button(
                        "Download patch predictions (CSV)",
                        data=pd.DataFrame(
                            {
                                "patch_number": np.arange(1, len(predictions) + 1),
                                "predicted_score": predictions,
                            }
                        ).to_csv(index=False).encode("utf-8"),
                        file_name="handwriting_patch_predictions.csv",
                        mime="text/csv",
                    )
        except Exception as exc:
            st.error("Could not process this image.")
            st.exception(exc)


# ---------------------------------------------------------------------
# 9. SHAP EXPLANATION
# ---------------------------------------------------------------------
elif page == "Explain Prediction (SHAP)":
    st.subheader("Explain a Prediction with SHAP")
    st.write(
        "SHAP highlights image regions that influence the selected patch's prediction. "
        "This is a local explanation of one patch, not a proof of why a person writes "
        "in a particular way."
    )

    if not SHAP_AVAILABLE:
        st.error("SHAP is not installed in this environment.")
        st.code("pip install shap")
    else:
        st.info(
            "Image SHAP can be computationally expensive. Start with a smaller evaluation "
            "budget; increase it only if runtime and memory allow."
        )

        use_previous = (
            "last_patches" in st.session_state
            and len(st.session_state["last_patches"]) > 0
        )
        uploaded_shap = st.file_uploader(
            "Or upload an image for SHAP",
            type=["jpg", "jpeg", "png"],
            key="shap_upload",
        )

        chosen_patch = None
        if uploaded_shap is not None:
            try:
                shap_image = prepare_image(uploaded_shap)
                shap_patches = image_to_patches(shap_image)
                st.image(shap_image, caption="Image for SHAP", use_container_width=True)
            except Exception as exc:
                st.error(f"Could not open image: {exc}")
                shap_patches = []
        elif use_previous:
            shap_patches = st.session_state["last_patches"]
            st.caption("Using the most recently assessed image.")
        else:
            shap_patches = []
            st.warning("Assess an image first, or upload an image here.")

        if shap_patches:
            selected_index = st.number_input(
                "Patch number to explain",
                min_value=1,
                max_value=len(shap_patches),
                value=1,
                step=1,
            )
            chosen_patch = shap_patches[int(selected_index) - 1]
            st.image(chosen_patch, caption=f"Selected patch {selected_index}", width=350)

            budget = st.select_slider(
                "SHAP evaluation budget (higher may take longer)",
                options=[100, 200, 300, 500],
                value=200,
            )

            if st.button("Generate SHAP explanation", type="primary"):
                model = model_status()
                if model is not None:
                    try:
                        with st.spinner("Calculating SHAP values..."):
                            rgb = np.asarray(chosen_patch.convert("RGB"), dtype=np.uint8)
                            predictor = make_shap_predictor(model)
                            masker = shap.maskers.Image("blur(16,16)", shape=rgb.shape)
                            explainer = shap.Explainer(
                                predictor,
                                masker,
                                algorithm="partition",
                                output_names=["Predicted handwriting quality"],
                            )
                            explanation = explainer(
                                rgb[np.newaxis, ...],
                                max_evals=int(budget),
                                batch_size=8,
                            )

                        values = np.asarray(explanation.values)
                        # Expected common image-output layouts:
                        # (1, H, W, C) for scalar output, or (1, H, W, C, 1).
                        values = values[0]
                        if values.ndim == 4 and values.shape[-1] == 1:
                            values = values[..., 0]
                        if values.ndim != 3:
                            st.warning(
                                f"Unexpected SHAP output shape {values.shape}. "
                                "Check the installed SHAP version's output format."
                            )
                        else:
                            # Reduce RGB-channel contributions to a signed pixel map.
                            signed_map = values.sum(axis=-1)
                            magnitude_map = np.abs(values).sum(axis=-1)
                            pred = float(predictor(rgb[np.newaxis, ...])[0])

                            fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
                            axes[0].imshow(rgb)
                            axes[0].set_title(f"Input patch\nPrediction: {pred:.4f}")
                            axes[0].axis("off")

                            im1 = axes[1].imshow(signed_map, cmap="coolwarm")
                            axes[1].set_title("Signed SHAP contribution")
                            axes[1].axis("off")
                            fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

                            im2 = axes[2].imshow(magnitude_map, cmap="viridis")
                            axes[2].set_title("Contribution magnitude")
                            axes[2].axis("off")
                            fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

                            fig.tight_layout()
                            st.pyplot(fig)
                            plt.close(fig)

                            st.caption(
                                "Positive signed values push the model output upward; "
                                "negative values push it downward. Magnitude shows strength, "
                                "not whether the contribution is positive or negative."
                            )

                            col1, col2 = st.columns(2)
                            with col1:
                                st.download_button(
                                    "Download signed SHAP map (CSV)",
                                    data=pd.DataFrame(signed_map).to_csv(
                                        index=False, header=False
                                    ).encode("utf-8"),
                                    file_name="shap_signed_map.csv",
                                    mime="text/csv",
                                )
                            with col2:
                                import io
                                npy_buffer = io.BytesIO()
                                np.save(npy_buffer, values)
                                st.download_button(
                                    "Download SHAP values (NumPy)",
                                    data=npy_buffer.getvalue(),
                                    file_name="shap_values.npy",
                                    mime="application/octet-stream",
                                )
                    except Exception as exc:
                        st.error("SHAP explanation failed.")
                        st.exception(exc)


# ---------------------------------------------------------------------
# 10. ABOUT / MODEL DETAILS
# ---------------------------------------------------------------------
elif page == "About the Model":
    st.subheader("About This Application")
    st.markdown(
        """
        **Model:** EfficientNet-B0  
        **Task:** Handwriting quality score regression  
        **Explainability method:** SHAP  
        **Input:** Image of handwritten text  
        **Output:** Regression prediction for each processed patch, aggregated by mean
        """
    )
    st.markdown("### Configuration")
    config = {
        "project_directory": str(PROJECT_DIR),
        "checkpoint": str(MODEL_PATH),
        "checkpoint_exists": MODEL_PATH.exists(),
        "device": str(DEVICE),
        "patch_size": PATCH_SIZE,
        "patch_stride": PATCH_STRIDE,
        "input_size": INPUT_SIZE,
        "imagenet_normalization": USE_IMAGENET_NORMALIZATION,
        "score_multiplier": SCORE_MULTIPLIER,
        "score_offset": SCORE_OFFSET,
        "display_score_min": DISPLAY_SCORE_MIN,
        "display_score_max": DISPLAY_SCORE_MAX,
        "shap_available": SHAP_AVAILABLE,
    }
    st.json(config)

    st.markdown("### Limitations")
    st.markdown(
        """
        - A prediction is an estimate learned from the training labels, not an objective
          or universal measure of handwriting quality.
        - Input image quality, writing instrument, paper, language, and dataset bias may
          affect predictions.
        - SHAP describes model behaviour for the selected input; it does not establish
          causation.
        - This dashboard uses a mean of patch predictions. If training or evaluation used
          a different page-level aggregation method, update this app to match it.
        """
    )

st.divider()
st.markdown(
    '<p class="small-note">Handwriting Analysis · EfficientNet-B0 · SHAP</p>',
    unsafe_allow_html=True,
)
