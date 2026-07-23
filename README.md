# GAN-Based Photo Restoration

A deep learning project for restoring old and damaged family photographs using a Pix2Pix Generative Adversarial Network (GAN). The model learns to reconstruct damaged images by training on paired clean and artificially damaged image datasets.

---

## Project Structure

```text
GAN-Photo-Restoration/
│
├── config.py
├── train.py
├── test.py
├── requirements.txt
├── README.md
│
├── dataset/
│   ├── __init__.py
│   ├── augment.py
│   ├── dataset.py
│   ├── clean/
│   └── damaged/
│
├── models/
│   ├── __init__.py
│   ├── generator.py
│   ├── discriminator.py
│   ├── pix2pix.py
│   └── weights.py
│
├── losses/
│   ├── __init__.py
│   ├── loss.py
│   └── perceptual.py
│
├── utils/
│   ├── __init__.py
│   ├── checkpoint.py
│   ├── image_utils.py
│   └── metrics.py
│
└── outputs/
    ├── checkpoints/
    ├── samples/
    └── results/
```

---

## Features Implemented

- U-Net Generator
- PatchGAN Discriminator
- Artificial image damage generation
- Custom PyTorch Dataset
- Pix2Pix Loss (GAN + L1)
- Training pipeline
- Model checkpointing
- Sample image saving
- Modular project structure

---

## Technology Stack

- Python
- PyTorch
- Torchvision
- OpenCV
- Matplotlib

---

## Training Pipeline

```text
Clean Images
      │
      ▼
Artificial Damage Generation
      │
      ▼
Damaged Images
      │
      ▼
Custom Dataset & DataLoader
      │
      ▼
Pix2Pix GAN
      │
      ├── Generator (U-Net)
      └── Discriminator (PatchGAN)
      │
      ▼
Loss Computation
(GAN Loss + L1 Loss)
      │
      ▼
Checkpoint Saving
```

---

## Current Progress

- ✔ Project structure created
- ✔ Data preprocessing pipeline
- ✔ Artificial damage generation
- ✔ Dataset loader
- ✔ U-Net Generator
- ✔ PatchGAN Discriminator
- ✔ Pix2Pix Loss
- ✔ Training pipeline
- ✔ Checkpoint utility
- ✔ Sample image saving

---

## Next Steps

- Train on FFHQ dataset using Kaggle GPU
- Build inference pipeline
- Develop Streamlit web application
- Evaluate using PSNR and SSIM
- Integrate Perceptual Loss

---

## Running the Project

Generate damaged images

```bash
python dataset/augment.py
```

Train the model

```bash
python train.py
```

---

## License

This project is intended for learning, research, and educational purposes.