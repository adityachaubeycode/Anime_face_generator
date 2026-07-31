![Python](https://img.shields.io/badge/Python-3.12-blue?style=plastic&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=plastic&logo=pytorch&logoColor=white)
![DenseNet121](https://img.shields.io/badge/Model-DenseNet121-green?style=plastic)
![License](https://img.shields.io/badge/License-MIT-green?style=plastic)

# 🎨 AnimeFace-DDPM
### A PyTorch Implementation of Denoising Diffusion Probabilistic Models (DDPM) from Scratch

---

## 📖 Overview

AnimeFace-DDPM is a deep learning project that implements **Denoising Diffusion Probabilistic Models (DDPM)** completely from scratch using **PyTorch**.

The model was trained on the **Anime Face Dataset** in a **Kaggle Notebook** using an **NVIDIA Tesla T4 GPU**. During inference, **Exponential Moving Average (EMA)** shadow weights are used alongside **DDIM sampling** to improve image quality while significantly reducing sampling time.

The primary objective of this project was to build a complete understanding of diffusion models by implementing every major component from the ground up rather than relying on existing diffusion libraries.

---

## ✨ Features

- 🚀 DDPM implemented entirely from scratch in PyTorch
- 🧠 Custom U-Net architecture
- ⏱️ DDIM sampling for faster inference
- 📈 EMA (Exponential Moving Average) for improved image quality
- 🎨 Trained on the Anime Face Dataset
- ⚡ GPU-accelerated training on NVIDIA Tesla T4
- 📊 Approximately **7.4 Million** trainable parameters

---

# 🖼️ Results

## Generated Samples

<p align="center">
  <img width="794" height="238" alt="image" src="https://github.com/user-attachments/assets/e69f80ba-f419-4247-b5e5-acb4291bd3c0" />
</p>

---

## Training Loss

The model converged smoothly during training with a consistently decreasing loss.

<p align="center">
 <img width="872" height="470" alt="image" src="https://github.com/user-attachments/assets/a1cf4970-31ba-4576-b273-3d8a9704a146" />
</p>

---

# 🏛️ Model Architecture

The diffusion model uses a custom **U-Net** backbone designed specifically for image generation.

**Architecture Components**

- Sinusoidal Time Embeddings
- Residual Blocks
- Encoder-Decoder U-Net
- Skip Connections
- Multi-scale Feature Learning
- Approximately **7.4M Parameters**
- 
| Component | Description |
|-----------|-------------|
| Input Convolution | Initial feature extraction |
| Time Embeddings | Sinusoidal positional embeddings followed by an MLP |
| Encoder | Residual blocks with downsampling |
| Bottleneck | Two residual blocks |
| Decoder | Residual blocks with transposed convolution upsampling |
| Output Layer | Final convolution predicting the added Gaussian noise |

### Model Statistics

| Metric | Value |
|--------|-------|
| Parameters | **7,520,963** |
| Trainable Parameters | **7,520,963** |
| Non-trainable Parameters | **0** |
<details>
<summary><b>Expand Full Model Summary</b></summary>

```text
======================================================================
Layer (type:depth-idx)                        Param #
======================================================================
Unet                                          --
├─Time_embeddings: 1-1                        --
│    └─SinusoidalPositionEmbeddings: 2-1      --
│    └─Sequential: 2-2                        --
│    │    └─Linear: 3-1                       33,024
│    │    └─SiLU: 3-2                         --
│    │    └─Linear: 3-3                       65,792
├─Conv2d: 1-2                                 1,792
├─ResBlock: 1-3                               --
│    └─Conv2d: 2-3                            73,856
│    └─Conv2d: 2-4                            147,584
│    └─GroupNorm: 2-5                         256
│    └─GroupNorm: 2-6                         256
│    └─SiLU: 2-7                              --
│    └─Linear: 2-8                            32,896
│    └─Conv2d: 2-9                            8,320
├─Down_sample: 1-4                            --
│    └─Conv2d: 2-10                           262,272
├─ResBlock: 1-5                               --
│    └─Conv2d: 2-11                           295,168
│    └─Conv2d: 2-12                           590,080
│    └─GroupNorm: 2-13                        512
│    └─GroupNorm: 2-14                        512
│    └─SiLU: 2-15                             --
│    └─Linear: 2-16                           65,792
│    └─Conv2d: 2-17                           33,024
├─Down_sample: 1-6                            --
│    └─Conv2d: 2-18                           1,048,832
├─ResBlock: 1-7                               --
│    └─Conv2d: 2-19                           590,080
│    └─Conv2d: 2-20                           590,080
│    └─GroupNorm: 2-21                        512
│    └─GroupNorm: 2-22                        512
│    └─SiLU: 2-23                             --
│    └─Linear: 2-24                           65,792
│    └─Identity: 2-25                         --
├─ResBlock: 1-8                               --
│    └─Conv2d: 2-26                           590,080
│    └─Conv2d: 2-27                           590,080
│    └─GroupNorm: 2-28                        512
│    └─GroupNorm: 2-29                        512
│    └─SiLU: 2-30                             --
│    └─Linear: 2-31                           65,792
│    └─Identity: 2-32                         --
├─Up_sample: 1-9                              --
│    └─ConvTranspose2d: 2-33                  1,048,832
├─ResBlock: 1-10                              --
│    └─Conv2d: 2-34                           589,952
│    └─Conv2d: 2-35                           147,584
│    └─GroupNorm: 2-36                        256
│    └─GroupNorm: 2-37                        256
│    └─SiLU: 2-38                             --
│    └─Linear: 2-39                           32,896
│    └─Conv2d: 2-40                           65,664
├─Up_sample: 1-11                             --
│    └─ConvTranspose2d: 2-41                  262,272
├─ResBlock: 1-12                              --
│    └─Conv2d: 2-42                           147,520
│    └─Conv2d: 2-43                           36,928
│    └─GroupNorm: 2-44                        128
│    └─GroupNorm: 2-45                        128
│    └─SiLU: 2-46                             --
│    └─Linear: 2-47                           16,448
│    └─Conv2d: 2-48                           16,448
├─Conv2d: 1-13                                1,731
======================================================================
Total params: 7,520,963
Trainable params: 7,520,963
Non-trainable params: 0
======================================================================
```

</details>
---

# ⚙️ Training Configuration

| Parameter | Value |
|-----------|-------|
| Framework | PyTorch |
| Dataset | Anime Face Dataset |
| Training Platform | Kaggle Notebook |
| GPU | NVIDIA Tesla T4 |
| Image Resolution | **64 × 64**  |
| Diffusion Timesteps | **1000** |
| Optimizer | AdamW |
| Batch Size | **64** |
| Epochs | **50** |
| Parameters | **~7.4 Million** |
| Sampling | DDIM |
| EMA | Enabled |

---

# 🔄 Inference Pipeline

```text
Random Gaussian Noise
          │
          ▼
    DDIM Sampling
          │
          ▼
   EMA Shadow Weights
          │
          ▼
 Generated Anime Face
```

---

# 🚀 Getting Started

## Clone the repository

```bash
git clone https://github.com/yourusername/AnimeFace-DDPM.git

cd AnimeFace-DDPM
```

---

## Install dependencies

```bash
pip install -r requirements.txt
```

---

## Training

Training was performed in a Kaggle Notebook using an NVIDIA Tesla T4 GPU.


## Generate Images

To generate new anime faces, run :

```
anime_face_generator.ipynb
```

The inference pipeline uses:

- EMA shadow weights
- DDIM sampling

to produce high-quality samples efficiently.

---

# 📚 Technologies Used

- PyTorch
- Torchvision
- NumPy
- Matplotlib
- tqdm
- torchinfo

---

# 🎯 Future Improvements

- Higher resolution image generation
- Classifier-Free Guidance (CFG)
- Conditional generation
- FID and Inception Score evaluation
- Hugging Face Spaces demo
- ONNX/TorchScript export

---

# 🙏 Acknowledgements

- **PyTorch** for the deep learning framework.
- **Kaggle** for providing the training environment.
- **DDPM** – Ho et al., *Denoising Diffusion Probabilistic Models* (2020).
- **DDIM** – Song et al., *Denoising Diffusion Implicit Models* (2020).

---

# 📜 License

This project is licensed under the **MIT License**.

---

# ⭐ Support

If you found this project useful, please consider giving it a **⭐ Star** on GitHub.

Contributions, suggestions, and feedback are always welcome.
