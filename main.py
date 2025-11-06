"""
DCGAN for Album Covers (128x128) - FIXED VERSION
Based on: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

FIXES APPLIED:
1. Using FULL 20k dataset (not 1000 sample)
2. Separate learning rates: D=0.0001, G=0.0002
3. Added label smoothing to prevent D overconfidence
4. Added noise to real images
5. Update D every other iteration (G gets more chances)
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
    batch_size = 8
    image_size = 128
    nc = 3  # RGB
    nz = 100  # Latent vector size
    ngf = 64  # Generator feature maps
    ndf = 64  # Discriminator feature maps
    num_epochs = 25
    
    # FIX 1: SEPARATE LEARNING RATES
    lr_d = 0.0001  # Lower for Discriminator (slow it down)
    lr_g = 0.0002  # Keep higher for Generator
    
    beta1 = 0.5
    ngpu = 1
    
    # FIX 2: LABEL SMOOTHING (prevents D overconfidence)
    real_label = 0.9  # Instead of 1.0
    fake_label = 0.0
    
    # FIX 3: NOISE FOR REAL IMAGES (prevents D memorization)
    add_noise_to_real = True
    noise_std = 0.1
    
    outf = './blues_output_FIXED'
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
    
    # FIX 4: USE FULL DATASET (not just 1000)
    print(f"\n✓ Using FULL dataset: {len(df)} images")
    blues_df = df.copy()  # ALL 20,000 images
    
    blues_df = blues_df.reset_index(drop=True)
    print(f"✓ Training on {len(blues_df)} album covers")
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
    plt.title("Training Images Sample")
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

    #============================================
    # TRAINING SETUP
    #============================================
    criterion = nn.BCELoss()
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)
    
    # FIX: DIFFERENT LEARNING RATES
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
            
            # FIX 5: Only update D every other iteration
            if i % 2 == 0:  # Update D less frequently
                netD.zero_grad()
                
                # Train with real
                real_cpu = data[0].to(device)
                b_size = real_cpu.size(0)
                
                # FIX: Add noise to real images (prevents memorization)
                if add_noise_to_real:
                    noise = torch.randn_like(real_cpu) * noise_std
                    real_noisy = real_cpu + noise
                    real_noisy = torch.clamp(real_noisy, -1, 1)
                else:
                    real_noisy = real_cpu
                
                # FIX: Label smoothing (0.9 instead of 1.0)
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
            
            vutils.save_image(fake, f'{outf}/fake_samples_epoch_{epoch:03d}.png', normalize=True, nrow=8)
            
            # Check RGB
            first_img = fake[0]
            r_g_diff = torch.abs(first_img[0] - first_img[1]).sum().item()
            r_b_diff = torch.abs(first_img[0] - first_img[2]).sum().item()
            
            print(f'\n{"="*60}')
            print(f'EPOCH {epoch} VALIDATION')
            print(f'{"="*60}')
            print(f'RGB channel differences: R-G={r_g_diff:.2f}, R-B={r_b_diff:.2f}')
            if r_g_diff < 1.0 and r_b_diff < 1.0:
                print('⚠️  WARNING: Channels nearly identical (grayscale)')
            else:
                print('✓ Channels different (learning color)')
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
    print("TRAINING COMPLETE")
    print("="*60)

    # Loss plot
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss")
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
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0][:64], padding=5, normalize=True).cpu(),(1,2,0)))

    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Generated Images")
    plt.imshow(np.transpose(img_list[-1],(1,2,0)))
    plt.savefig(f'{outf}/real_vs_fake_final.png', dpi=150, bbox_inches='tight')
    
    print(f"\n✓ All outputs saved to: {outf}/")
    print("="*60)

if __name__ == '__main__':
    main()
