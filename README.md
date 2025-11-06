# DCGAN Album Cover Generator

A Deep Convolutional Generative Adversarial Network (DCGAN) implementation for generating album cover artwork, trained on 20,000 diverse album covers from multiple genres.

## 🎨 Overview

This project implements a DCGAN to generate 128×128 RGB album cover images. The model was trained on the HuggingFace dataset [eong/20k-Album-Covers-within-20-Genres](https://huggingface.co/datasets/eong/20k-Album-Covers-within-20-Genres) containing album covers across 20 different music genres.

## 🖼️ Results

**Training Progression:**
- [View training progression images](blues_output_FIXED/)
- [Final generated album covers](blues_output_FIXED/final_64_covers.png)
- [Real vs Generated comparison](blues_output_FIXED/real_vs_fake_final.png)
- [Training loss curves](blues_output_FIXED/training_losses.png)

## 🏗️ Architecture

- **Generator:** 6-layer ConvTranspose network (latent → 128×128 RGB)
- **Discriminator:** 6-layer Conv network (128×128 RGB → probability)
- **Latent dimension:** 100
- **Image size:** 128×128 pixels
- **Color space:** RGB (3 channels)

### Training Improvements

This implementation includes several stability improvements over the standard DCGAN:

1. **Separate learning rates:** Discriminator (0.0001) slower than Generator (0.0002)
2. **Label smoothing:** Real labels = 0.9 instead of 1.0
3. **Noise injection:** Added noise to real images during training
4. **Alternating updates:** Discriminator updates every other iteration

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA/MPS support (optional, CPU also works)

## 🚀 Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd gan_albumcover_mod9

# Install dependencies
pip install -r requirements.txt
```

## 💻 Usage

### Training from Scratch

```bash
python main.py
```

The script will:
1. Download the dataset from HuggingFace (automatic)
2. Train for 25 epochs (~2 hours on Apple Silicon MPS)
3. Save checkpoints every 5 epochs
4. Generate validation images throughout training
5. Create final comparison visualizations

### Monitoring Training

In a separate terminal:

```bash
python monitor_training.py
```

This will display real-time progress including:
- Current epoch
- Number of files generated
- Estimated time remaining

### Output Structure

```
blues_output_FIXED/
├── final_64_covers.png          # 64 generated album covers (final)
├── real_vs_fake_final.png       # Side-by-side comparison
├── training_losses.png          # Loss curves over training
├── fake_samples_epoch_XXX.png   # Samples every 5 epochs
├── netG_epoch_XX.pth            # Generator checkpoints
└── netD_epoch_XX.pth            # Discriminator checkpoints
```

## 🎯 Training Details

- **Dataset:** 20,000 album covers
- **Batch size:** 8
- **Epochs:** 25
- **Optimizer:** Adam (β₁=0.5, β₂=0.999)
- **Learning rates:** G=0.0002, D=0.0001
- **Device:** MPS (Apple Silicon) / CUDA / CPU
- **Training time:** ~2 hours on M1/M2 Mac

## 📊 Performance

The model successfully generates diverse, colorful album cover-style images. Key achievements:

- ✅ RGB color learning (avoiding grayscale collapse)
- ✅ Stable training (no mode collapse)
- ✅ Diverse outputs across training
- ✅ Recognizable album cover aesthetics

## 🔧 Configuration

Key hyperparameters in `main.py`:

```python
batch_size = 8
image_size = 128
nc = 3          # RGB channels
nz = 100        # Latent dimension
ngf = 64        # Generator features
ndf = 64        # Discriminator features
num_epochs = 25
lr_d = 0.0001   # Discriminator learning rate
lr_g = 0.0002   # Generator learning rate
```

## 📚 References

- Original DCGAN paper: [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)
- PyTorch DCGAN Tutorial: [https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
- Dataset: [eong/20k-Album-Covers-within-20-Genres](https://huggingface.co/datasets/eong/20k-Album-Covers-within-20-Genres)

## 📝 License

[Add your license here]

## 👤 Author

[Add your name/info here]

## 🙏 Acknowledgments

- PyTorch team for the DCGAN tutorial
- HuggingFace for hosting the dataset
- Original album cover artists
