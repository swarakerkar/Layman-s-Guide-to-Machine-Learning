# Handwriting Quality Rating Prediction Using Deep Learning and Explainable AI

## Overview

This project aims to develop a machine learning system that predicts handwriting quality ratings from handwritten document images using deep learning and Explainable Artificial Intelligence (XAI).

The system uses image preprocessing, overlapping patch extraction, and an EfficientNet-B0 regression model to estimate continuous handwriting quality ratings based on human-assigned scores. Explainability techniques are incorporated to investigate the visual regions influencing model predictions.

The project focuses on automated handwriting assessment, reliable model evaluation, and interpretable predictions.

## Objectives

- Develop a reproducible handwriting image preprocessing pipeline.
- Extract overlapping image patches while preserving source-page information.
- Predict handwriting quality ratings using deep learning-based regression.
- Evaluate model performance using appropriate regression metrics.
- Investigate prediction errors and generalization across different handwriting samples.
- Apply Explainable AI to understand the visual patterns influencing predictions.

## Key Features

- **Image preprocessing:** Orientation correction, grayscale conversion, denoising, contrast enhancement, and optional sharpening.
- **Patch extraction:** Overlapping 640 × 640 image patches with a stride of 480 pixels.
- **Image resizing:** Conversion of patches into 224 × 224 RGB images.
- **Rating integration:** Association of handwriting images with human-assigned ratings from an Excel spreadsheet.
- **Deep learning:** EfficientNet-B0 pretrained on ImageNet and adapted for regression.
- **Model evaluation:** MAE, RMSE, R², and actual-versus-predicted visualizations.
- **Explainable AI:** Grad-CAM-based visualization of image regions influencing predicted ratings.
- **Data integrity:** Metadata generation, image validation, and source-page traceability.
- **Leakage prevention:** Page-level data splitting to ensure that patches from the same page remain in one dataset split.

## Technology Stack

| Component | Technologies |
|---|---|
| Programming language | Python |
| Development environment | Jupyter Notebook, Anaconda |
| Image processing | OpenCV, Pillow |
| Data manipulation | NumPy, pandas |
| Visualization | Matplotlib |
| Deep learning | PyTorch, torchvision |
| Machine learning utilities | scikit-learn |
| Regression model | EfficientNet-B0 |
| Explainability | Grad-CAM |
| Dataset annotations | Excel (`ratingsFinal.xlsx`) |

## Project Workflow

```text
Handwriting Image Dataset
          |
          v
Dataset Inspection and Metadata Generation
          |
          v
Rating Integration Using ratingsFinal.xlsx
          |
          v
Image Preprocessing and Enhancement
          |
          v
Overlapping Patch Extraction
     (640 x 640, stride 480)
          |
          v
Resize to 224 x 224 RGB
          |
          v
Page-Level Train / Validation / Test Split
          |
          v
EfficientNet-B0 Regression Model
          |
          v
Patch-Level Rating Predictions
          |
          v
Page-Level Prediction Aggregation
          |
          v
Regression Evaluation and Error Analysis
          |
          v
Grad-CAM Visual Explanations
```

## Dataset

The project uses a handwritten document image dataset containing pages written by multiple writers.

The rating annotations are maintained in `ratingsFinal.xlsx`. The intended target is the `avg` column, representing the average rating assigned by two raters.

The dataset preparation pipeline is designed to:

- Inspect image formats, dimensions, and validity.
- Match original filenames with the corresponding rating records.
- Identify missing ratings, duplicate filenames, and unmatched records.
- Generate structured metadata for reproducibility.
- Preserve the relationship between each patch and its original page.

**Note:** Dataset statistics, the number of usable images, and the number of writers should be reported only after inspecting the actual dataset.

## Methodology

### 1. Image Preprocessing

The original handwriting images are processed carefully to improve visual consistency while preserving fine strokes, loops, punctuation, and letter shapes.

Depending on image quality, preprocessing may include:

- Orientation correction.
- Grayscale conversion.
- Noise reduction.
- Local contrast enhancement.
- Mild sharpening.
- Thresholding for diagnostic purposes.

Original images are retained to allow comparison with processed outputs.

### 2. Patch Extraction

Large document pages are divided into overlapping patches to create manageable inputs for the deep learning model.

| Parameter | Configuration |
|---|---|
| Patch size | 640 × 640 pixels |
| Stride | 480 pixels |
| Overlap | 160 pixels |
| Model input size | 224 × 224 pixels |
| Input channels | 3 (RGB) |

Edge regions are included, and patch metadata records the source page, patch index, spatial coordinates, and file paths.

Each patch inherits the rating of its original page. Since the rating describes the entire page rather than an individual patch, this is treated as a weak-supervision assumption.

### 3. Deep Learning Model

EfficientNet-B0 pretrained on ImageNet serves as the initial regression architecture.

The final classification layer is replaced with a single-output regression layer that predicts a continuous handwriting quality score.

Initial training configuration:

- Optimizer: AdamW
- Learning rate: 0.0001
- Weight decay: 0.0001
- Loss function: Smooth L1 Loss
- Maximum epochs: 30
- Early stopping patience: 6 epochs
- Learning-rate scheduler: ReduceLROnPlateau
- Random seed: 42
- Device: CUDA GPU when available; otherwise CPU

These are initial experimental settings and may be adjusted based on validation performance and available hardware.

Page-aware weighting or sampling is used to prevent pages that produce more patches from dominating the training objective.

### 4. Data Splitting

The initial evaluation uses a 70:15:15 training, validation, and test split at the original-page level.

All patches from an individual page remain within the same split to prevent information leakage between training and evaluation.

A separate writer-wise evaluation is recommended when writer identifiers are available and reliable. This evaluates generalization to writers whose handwriting was not present during training.

### 5. Model Evaluation

Predictions from all patches belonging to a page are averaged to obtain one page-level prediction.

The following metrics are used:

- **Mean Absolute Error (MAE):** Measures the average absolute difference between predicted and actual ratings.
- **Root Mean Squared Error (RMSE):** Penalizes larger prediction errors more strongly.
- **R² Score:** Measures how well the predictions explain the variation in the target ratings relative to a mean-prediction baseline.

Additional analyses include training and validation curves, actual-versus-predicted plots, error distributions, and examination of pages with the largest prediction errors.

### 6. Explainable AI

Grad-CAM is planned to visualize image regions that influence the regression model's predicted rating.

The explanations will be examined to investigate whether the model responds to meaningful handwriting patterns, such as stroke structure, spacing, alignment, and letter formation.

These visualizations explain aspects of model behavior but do not independently prove that the highlighted features cause a particular human rating.

## Installation

### Prerequisites

- Python 3.10 or a compatible Python version.
- Jupyter Notebook or JupyterLab.
- Anaconda or another Python environment manager.

### Install Dependencies

```bash
pip install numpy pandas matplotlib opencv-python Pillow scikit-learn openpyxl
pip install torch torchvision
```

For GPU acceleration, install a PyTorch build compatible with your CUDA environment by following the official [PyTorch installation guide](https://pytorch.org/get-started/locally/).

## Configuration

Update the dataset paths in the notebook according to your local environment.

Example Windows directory structure:

```text
Handwriting Analysis/
├── dataset/
│   ├── raw/
│   ├── metadata/
│   └── processed/
├── notebooks/
├── models/
├── outputs/
├── ratingsFinal.xlsx
└── README.md
```

The exact directory structure can be adjusted to match the actual repository.

## Expected Outputs

The pipeline is designed to generate:

- Original dataset inventory and image metadata.
- Processed handwriting page images.
- Extracted and resized image patches.
- Patch metadata with source-page traceability.
- Training, validation, and test split information.
- Trained model checkpoints.
- Page-level predicted ratings and absolute errors.
- MAE, RMSE, and R² evaluation metrics.
- Training curves and prediction plots.
- Grad-CAM visualizations.

## Research Considerations

The project accounts for several methodological challenges:

1. **Data leakage:** Patches from the same page must not appear across different data splits.
2. **Weak supervision:** A page-level rating does not necessarily describe the quality of every individual patch.
3. **Writer variability:** Page-wise evaluation does not establish generalization to unseen writers.
4. **Subjective labels:** Human-assigned ratings may contain disagreement and measurement noise.
5. **Model interpretability:** Highlighted image regions must be interpreted cautiously and validated where possible.

These considerations are important when assessing the reliability and scientific validity of the system.

## Current Status

The project methodology and initial pipeline design are defined. Dataset validation, preprocessing, model training, evaluation, and explainability outputs should be documented as they are implemented and verified.

Performance metrics and experimental results will be added after the corresponding experiments have been executed.

## Future Enhancements

- Compare EfficientNet-B0 with ResNet and other suitable regression architectures.
- Evaluate writer-independent generalization.
- Extract interpretable handwriting features for a complementary tabular regression model.
- Apply SHAP or TreeSHAP to a compatible feature-based model.
- Investigate alternative patch-to-page aggregation strategies.
- Study inter-rater agreement and prediction uncertainty.
- Develop an interface for uploading handwriting images and obtaining predicted ratings.

## Applications

Potential applications include:

- Automated handwriting assessment.
- Educational handwriting analysis.
- Handwriting quality research.
- Document image analysis.
- Explainable computer vision for handwritten documents.

The system is intended as an experimental assessment tool and should not be treated as a definitive or universally valid measure of handwriting quality.

## License

Add an appropriate license before distributing the repository publicly. Ensure that the dataset's license and usage permissions allow redistribution or publication of derived materials.

## Acknowledgements

This project explores the application of deep learning, image processing, and Explainable AI to handwriting quality assessment.
