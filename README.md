# CelebA Image Generation using Diffusion Models (PyTorch)

## Project Overview

This repository implements **image generation on the CelebA dataset using diffusion models** in PyTorch.  
Diffusion models are state-of-the-art generative models that produce images by learning a reverse denoising process from noise. This project applies this technique to generate new face images similar to the CelebA dataset.

This repository contains code to train and evaluate different diffusion models for generating CelebA images and visualize the results and also to interpret the trained models using various interpretability methods such as attributions, metrics, etc.

---

## Dataset

- **CelebA (Celeb Faces Attributes)**
- ~200,000 face images
- Multiple annotated facial attributes
- Used here only for generative image modeling

Due to size, the dataset needs to be downloaded separately and prepared before training.

Both conditional and unconditional of denoising diffusion models as well as latent diffsuion models are trained and evaluated.

---

## What This Project Includes

- PyTorch implementation of a diffusion model
- Data loading and preprocessing
- Training loop with noise schedule
- Sampling/generation pipeline
- Visualization of generated images
- Interpretability of the trained models.

---
