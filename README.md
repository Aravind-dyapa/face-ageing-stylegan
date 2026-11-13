
# Face Ageing and De-ageing using Conditional StyleGAN

An interactive face ageing and de-ageing application built using the pretrained SAM (Style-based Age Manipulation) framework, with an experimental WNNR-inspired latent-space regularization extension.

## Overview

This project explores facial age transformation using a pretrained StyleGAN-based framework.

Users can upload a face image, select a target age, and generate an age-transformed output. The application also provides an experimental latent-space regularization option to study its effect on generated images.

## Key Features

- Face ageing and de-ageing across target ages from 0 to 100.
- Face alignment and preprocessing using dlib.
- Pretrained SAM model for age transformation.
- WNNR-inspired latent-space regularization.
- Comparison between baseline and regularized outputs.
- Singular value analysis and nuclear norm measurements.
- SSIM and PSNR experimental evaluation.
- Interactive Streamlit application.
- Experimental dashboard with recorded results.

## Methodology

### 1. Baseline Age Transformation

The pretrained SAM framework maps an input face into the StyleGAN latent space and performs age transformation using the pretrained generator.

### 2. WNNR-inspired Latent Regularization

An experimental regularization operation is applied to the SAM-generated W+ latent representation.

The implementation:
1. Centers the latent representation using the latent average.
2. Performs Singular Value Decomposition (SVD).
3. Applies weighted singular value shrinkage.
4. Reconstructs the modified latent representation.
5. Generates the output using the pretrained StyleGAN2 decoder.

The singular value shrinkage is defined as:

\[
\sigma'_i = \max(\sigma_i-\lambda w_i,0)
\]

where:
- \(\sigma_i\) represents a singular value.
- \(\lambda\) controls regularization strength.
- \(w_i\) represents the corresponding weight.

This is an experimental WNNR-inspired extension, not a newly proposed WNNR algorithm.

## Technology Stack

- Python
- PyTorch
- StyleGAN2
- SAM (Style-based Age Manipulation)
- Singular Value Decomposition (SVD)
- NumPy
- OpenCV
- dlib
- Streamlit
- scikit-image

## Project Structure

```text
SAM/
├── configs/
├── criteria/
│   └── wnnr_loss.py
├── datasets/
├── experiments/
│   ├── baseline/
│   ├── wnnr/
│   └── multi_age_wnnr/
├── face_ageing_app/
│   ├── app.py
│   └── experiment_dashboard.py
├── models/
│   └── stylegan2/
│       └── op/
│           ├── fused_act.py
│           └── upfirdn2d.py
├── pretrained_models/
├── scripts/
├── .gitignore
└── README.md
```

## Experimental Evaluation

The project investigates the effect of different regularization strengths on latent representations.

Evaluation includes:
- Nuclear norm measurements.
- Structural Similarity Index (SSIM).
- Peak Signal-to-Noise Ratio (PSNR).
- Baseline versus regularized output comparisons.

**Note:** SSIM and PSNR in the current experiments measure similarity between regularized outputs and baseline SAM outputs. They do not independently establish improved perceptual quality or identity preservation. The experiments are preliminary and should not be interpreted as broad validation.

## Running the Application

The application is developed and tested in Google Colab with GPU acceleration.

1. Clone this repository.
2. Obtain the pretrained SAM checkpoint and required landmark predictor.
3. Place the checkpoint at `pretrained_models/sam_ffhq_aging.pt`.
4. Place the landmark predictor at `shape_predictor_68_face_landmarks.dat` in the project root.
5. Install the required dependencies.
6. Launch the Streamlit application:

```bash
cd face_ageing_app
streamlit run app.py
```

To launch the experimental dashboard:

```bash
streamlit run experiment_dashboard.py --server.port 8502
```

The pretrained model and landmark predictor are not included in this repository because of their size and licensing considerations. Refer to the original SAM project for model acquisition instructions.

## Original Research and Acknowledgements

This implementation builds upon:

**Only a Matter of Style: Age Transformation Using a Style-Based Regression Model**

Yuval Alaluf, Or Patashnik, and Daniel Cohen-Or, SIGGRAPH 2021.

Original implementation:
https://github.com/yuval-alaluf/SAM

The original SAM authors retain credit for their framework and pretrained model. This repository contains an experimental extension and application built on top of their work.

## Disclaimer

This project is intended for academic and research purposes. Generated age transformations are synthetic and may not represent actual human ageing.
