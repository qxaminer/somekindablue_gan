"""
DCGAN v2: Blues-Filtered Photorealistic Training
================================================
Based on: blues_dcgan_FIXED_FINAL.py (v1)

KEY IMPROVEMENTS FROM V1:
========================
1. Dataset Filtering: Blues genre only (~1k covers vs 20k multi-genre)
2. Architecture: Increased capacity (ngf=128, ndf=128 vs 64)
3. Latent Space: Larger (nz=200 vs 100)  
4. Training: More epochs (35 vs 25)
5. Learning Rate: Even more conservative for D (lr_d=0.00008 vs 0.0001)

EXPECTED RESULTS:
================
- Photorealistic Blues album covers
- Centered musician portraits
- Visible instruments (guitars, harmonicas)
- Vintage brown/orange color palette
- Less abstract than v1

V1 RESULTS (for comparison):
============================
- Multi-genre dataset → Abstract/glitch aesthetic
- Network averaged across diverse visual cultures
- Interesting but not genre-specific

Training time: ~20-25 hours on M4 Mac Mini

Branch: blues-filtered
"""

from __future__ import print_function
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import pandas as pd

#============================================
# CUSTOM DATASET CLASS
#============================================
class AlbumCoverDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        img_data = self.dataframe.iloc[idx]['image']
        
        if isinstance(img_data, dict) and 'bytes' in img_data:
            img_bytes = img_data['bytes']
        elif isinstance(img_data, bytes):
            img_bytes = img_data
        else:
            raise ValueError(f"Unexpected image data format: {type(img_data)}")
        
        img = Image.open(io.BytesIO(img_bytes))
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            
        return img, 0

#============================================
# WEIGHT INITIALIZATION
#============================================
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

#============================================
# GENERATOR (128x128)
#============================================
class Generator(nn.Module):
    def __init__(self, ngpu, nz, ngf, nc):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # Input: (nz, 1, 1) → Output: (ngf*16, 4, 4)
            nn.ConvTranspose2d(nz, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            # (ngf*16, 4, 4) → (ngf*8, 8, 8)
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # (ngf*8, 8, 8) → (ngf*4, 16, 16)
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # (ngf*4, 16, 16) → (ngf*2, 32, 32)
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # (ngf*2, 32, 32) → (ngf, 64, 64)
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # (ngf, 64, 64) → (nc, 128, 128)
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

#============================================
# DISCRIMINATOR (128x128)
#============================================
class Discriminator(nn.Module):
    def __init__(self, ngpu, nc, ndf):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # Input: (nc, 128, 128) → Output: (ndf, 64, 64)
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf, 64, 64) → (ndf*2, 32, 32)
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*2, 32, 32) → (ndf*4, 16, 16)
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*4, 16, 16) → (ndf*8, 8, 8)
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*8, 8, 8) → (ndf*16, 4, 4)
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*16, 4, 4) → (1, 1, 1)
            nn.Conv2d(ndf * 16, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

#============================================
# MAIN TRAINING FUNCTION
#============================================
def main():
    # Set random seed
    manualSeed = 999
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    #============================================
    # CONFIGURATION
    #============================================
    workers = 0  # For macOS compatibility
    
    # V2 IMPROVEMENTS
    ngf = 128          # Generator feature maps (DOUBLED for more detail)
    ndf = 128          # Discriminator feature maps (DOUBLED for more capacity)  
    nz = 200           # Latent vector size (DOUBLED for expressiveness)
    num_epochs = 35    # More epochs for convergence
    lr_d = 0.00008     # EVEN SLOWER D (was 0.0001)
    lr_g = 0.0002      # Keep G same
    batch_size = 8     # May need to reduce to 4 if memory issues
    
    image_size = 128
    nc = 3  # RGB
    beta1 = 0.5
    ngpu = 1
    
    # Label smoothing (prevents D overconfidence)
    real_label = 0.9  # Instead of 1.0
    fake_label = 0.0
    
    # Noise for real images (prevents D memorization)
    add_noise_to_real = True
    noise_std = 0.1
    
    outf = './blues_output_v2'  # NEW OUTPUT DIRECTORY
    os.makedirs(outf, exist_ok=True)

    #============================================
    # LOAD FULL DATASET
    #============================================
    print("\n" + "="*60)
    print("LOADING DATASET FROM HUGGINGFACE")
    print("="*60)
    
    print("Loading parquet file...")
    df = pd.read_parquet("hf://datasets/eong/20k-Album-Covers-within-20-Genres/data/train-00000-of-00001-f37f5042abc5be8d.parquet")
    
    print(f"✓ Loaded {len(df)} total album covers")
    print(f"✓ Columns: {list(df.columns)}")

    #============================================
    # V2: FILTER FOR BLUES GENRE ONLY
    #============================================
    print("\n" + "="*60)
    print("V2: BLUES-ONLY DATASET FILTERING")
    print("="*60)

    print(f"\nOriginal dataset: {len(df)} covers across all genres")

    # Check label structure
    print(f"\nLabel column type: {df['label'].dtype}")
    print(f"Unique labels (first 20): {sorted(df['label'].unique())[:20]}")

    # Get label distribution
    print(f"\nLabel distribution:")
    label_counts = df['label'].value_counts().head(20)
    print(label_counts)

    # Determine Blues label value
    # IMPORTANT: Adjust this based on your label structure
    # Option A: If labels are numeric (0-19)
    blues_label = 0  # CHECK THE OUTPUT ABOVE - may need to change this number!

    # Option B: If labels are strings, use this instead:
    # blues_df = df[df['label'].str.contains('Blues', case=False, na=False)].copy()

    # Filter for Blues
    blues_df = df[df['label'] == blues_label].copy()

    print(f"\nFiltered to Blues only: {len(blues_df)} covers")
    print(f"Percentage of dataset: {len(blues_df)/len(df)*100:.1f}%")

    if len(blues_df) < 100:
        print("\n⚠️  WARNING: Very small Blues dataset!")
        print("   Consider checking if label value is correct")
        print(f"   Current filter: label == {blues_label}")
    elif len(blues_df) > 5000:
        print("\n📊 NOTE: Large Blues subset")
        print("   This is good - enough diversity for training")

    blues_df = blues_df.reset_index(drop=True)

    # Use blues_df instead of df for dataset creation
    print(f"\nProceeding with {len(blues_df)} Blues album covers")
    print("="*60 + "\n")

    #============================================
    # CREATE DATASET & DATALOADER
    #============================================
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = AlbumCoverDataset(blues_df, transform=transform)
    
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=workers
    )

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Using CUDA")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Dataset info
    print("\n" + "="*60)
    print("DATASET VERIFICATION")
    print("="*60)
    print(f"Number of album covers: {len(dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"Batches per epoch: {len(dataloader)}")
    print(f"Image size: {image_size}x{image_size}")
    print(f"Channels (nc): {nc}")
    print("="*60 + "\n")

    # Test loading
    try:
        test_batch = next(iter(dataloader))
        print(f"✓ Test batch shape: {test_batch[0].shape}")
        
        if test_batch[0].shape[1] != 3:
            print("❌ ERROR: Images are not 3-channel RGB!")
            return
        else:
            print("✓ Images are RGB (3 channels)")
    except Exception as e:
        print(f"❌ ERROR loading batch: {e}")
        return

    # Visualize training data
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training Images Sample - V2 Blues-Filtered")
    plt.imshow(np.transpose(vutils.make_grid(test_batch[0][:min(64, batch_size*8)], padding=2, normalize=True).cpu(),(1,2,0)))
    plt.savefig(f'{outf}/00_training_data_sample.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {outf}/00_training_data_sample.png\n")

    #============================================
    # CREATE NETWORKS
    #============================================
    netG = Generator(ngpu, nz, ngf, nc).to(device)
    netG.apply(weights_init)
    
    netD = Discriminator(ngpu, nc, ndf).to(device)
    netD.apply(weights_init)

    print("="*60)
    print("NETWORK ARCHITECTURE")
    print("="*60)
    print("\nGenerator:")
    print(netG)
    print("\nDiscriminator:")
    print(netD)
    
    # Test Generator
    test_noise = torch.randn(1, nz, 1, 1, device=device)
    test_output = netG(test_noise)
    print(f"\n✓ Generator output: {test_output.shape}")
    print(f"  Expected: [1, 3, 128, 128]")
    
    if test_output.shape[1] != 3:
        print("❌ ERROR: Generator wrong output channels!")
        return
    
    print("="*60 + "\n")

    print("\n" + "="*60)
    print("V2 ARCHITECTURE IMPROVEMENTS")
    print("="*60)
    print(f"Generator feature maps: {ngf} (v1: 64)")
    print(f"Discriminator feature maps: {ndf} (v1: 64)")
    print(f"Latent vector size: {nz} (v1: 100)")
    print(f"Training epochs: {num_epochs} (v1: 25)")
    print(f"Discriminator LR: {lr_d} (v1: 0.0001)")
    print(f"Generator LR: {lr_g}")
    print(f"Output directory: {outf}")
    print("\nExpected improvements:")
    print("  - More detail in generated covers")
    print("  - Better quality facial features")
    print("  - Clearer instruments and textures")
    print("  - Photorealistic Blues aesthetic")
    print("="*60 + "\n")

    #============================================
    # TRAINING SETUP
    #============================================
    criterion = nn.BCELoss()
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)
    
    optimizerD = optim.Adam(netD.parameters(), lr=lr_d, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr_g, betas=(beta1, 0.999))

    print("="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    print(f"Epochs: {num_epochs}")
    print(f"Batches per epoch: {len(dataloader)}")
    print(f"Discriminator LR: {lr_d}")
    print(f"Generator LR: {lr_g}")
    print(f"Real label: {real_label} (smoothed)")
    print(f"Fake label: {fake_label}")
    print(f"Add noise to real images: {add_noise_to_real}")
    print("="*60 + "\n")

    #============================================
    # TRAINING LOOP
    #============================================
    G_losses = []
    D_losses = []
    img_list = []
    iters = 0

    print("Starting Training Loop...\n")

    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            
            ############################
            # (1) Update D network
            ###########################
            
            # Only update D every other iteration
            if i % 2 == 0:  # Update D less frequently
                netD.zero_grad()
                
                # Train with real
                real_cpu = data[0].to(device)
                b_size = real_cpu.size(0)
                
                # Add noise to real images (prevents memorization)
                if add_noise_to_real:
                    noise = torch.randn_like(real_cpu) * noise_std
                    real_noisy = real_cpu + noise
                    real_noisy = torch.clamp(real_noisy, -1, 1)
                else:
                    real_noisy = real_cpu
                
                # Label smoothing (0.9 instead of 1.0)
                label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
                
                output = netD(real_noisy).view(-1)
                errD_real = criterion(output, label)
                errD_real.backward()
                D_x = output.mean().item()

                # Train with fake
                noise = torch.randn(b_size, nz, 1, 1, device=device)
                fake = netG(noise)
                label.fill_(fake_label)
                
                output = netD(fake.detach()).view(-1)
                errD_fake = criterion(output, label)
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                
                errD = errD_real + errD_fake
                optimizerD.step()
            else:
                # Don't update D, but still calculate for logging
                with torch.no_grad():
                    real_cpu = data[0].to(device)
                    b_size = real_cpu.size(0)
                    output = netD(real_cpu).view(-1)
                    D_x = output.mean().item()
                    
                    noise = torch.randn(b_size, nz, 1, 1, device=device)
                    fake = netG(noise)
                    output = netD(fake).view(-1)
                    D_G_z1 = output.mean().item()
                    
                    errD = torch.tensor(0.0)

            ############################
            # (2) Update G network (ALWAYS)
            ###########################
            netG.zero_grad()
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            
            output = netD(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            # Print stats
            if i % 250 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(dataloader),
                         errD.item() if isinstance(errD, torch.Tensor) else errD, 
                         errG.item(), D_x, D_G_z1, D_G_z2))

            G_losses.append(errG.item())
            D_losses.append(errD.item() if isinstance(errD, torch.Tensor) else 0)
            iters += 1

        # Save validation images every 5 epochs
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            
            vutils.save_image(fake, f'{outf}/fake_samples_epoch_{epoch:03d}.png', 
                             normalize=True, nrow=8)
            
            # Enhanced RGB channel analysis
            first_img = fake[0]
            r_g_diff = torch.abs(first_img[0] - first_img[1]).sum().item()
            r_b_diff = torch.abs(first_img[0] - first_img[2]).sum().item()
            g_b_diff = torch.abs(first_img[1] - first_img[2]).sum().item()
            avg_diff = (r_g_diff + r_b_diff + g_b_diff) / 3
            
            print(f'\n{"="*60}')
            print(f'EPOCH {epoch}/{num_epochs} VALIDATION - V2 BLUES-FILTERED')
            print(f'{"="*60}')
            print(f'RGB channel differences:')
            print(f'  R-G: {r_g_diff:.2f}')
            print(f'  R-B: {r_b_diff:.2f}')
            print(f'  G-B: {g_b_diff:.2f}')
            print(f'  Average: {avg_diff:.2f}')
            
            if avg_diff < 1.0:
                print('⚠️  WARNING: Channels nearly identical (grayscale)')
            elif avg_diff < 5.0:
                print('⚠️  Low channel diversity (check training)')
            else:
                print('✓ Good channel diversity (learning color)')
            
            # Quality indicators
            print(f'\nTraining health check:')
            print(f'  Latest Loss_D: {errD.item() if isinstance(errD, torch.Tensor) else errD:.4f}')
            print(f'  Latest Loss_G: {errG.item():.4f}')
            print(f'  D(x): {D_x:.4f} (should be 0.6-0.8)')
            print(f'  D(G(z)): {D_G_z2:.4f} (should be 0.2-0.4)')
            
            if D_x > 0.95:
                print('⚠️  WARNING: D may be dominating (D(x) too high)')
            elif D_x < 0.5:
                print('⚠️  WARNING: G may be dominating (D(x) too low)')
            
            print(f'{"="*60}\n')
            
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        # Save checkpoints
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            torch.save(netG.state_dict(), f'{outf}/netG_epoch_{epoch}.pth')
            torch.save(netD.state_dict(), f'{outf}/netD_epoch_{epoch}.pth')

    #============================================
    # FINAL OUTPUTS
    #============================================
    print("\n" + "="*60)
    print("TRAINING COMPLETE - V2 BLUES-FILTERED")
    print("="*60)

    # Loss plot
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss - V2 Blues-Filtered")
    plt.plot(G_losses, label="G", alpha=0.7)
    plt.plot(D_losses, label="D", alpha=0.7)
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f'{outf}/training_losses.png', dpi=150, bbox_inches='tight')
    
    # Final covers
    with torch.no_grad():
        final_fake = netG(fixed_noise).detach().cpu()
    vutils.save_image(final_fake, f'{outf}/final_64_covers.png', normalize=True, nrow=8)
    
    # Real vs Fake comparison
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(15,8))
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("Real Blues Album Covers")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0][:64], padding=5, normalize=True).cpu(),(1,2,0)))

    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Generated Blues Album Covers")
    plt.imshow(np.transpose(img_list[-1],(1,2,0)))
    plt.savefig(f'{outf}/real_vs_fake_final.png', dpi=150, bbox_inches='tight')
    
    print(f"\n✓ All outputs saved to: {outf}/")
    print("="*60)

if __name__ == '__main__':
    main()

