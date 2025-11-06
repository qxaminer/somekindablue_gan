#!/usr/bin/env python3
"""
Real-time training monitor for DCGAN
Checks output directory and reports progress
"""

import os
import time
from datetime import datetime
from pathlib import Path

def monitor_training(output_dir='blues_output_FIXED', check_interval=30, max_checks=100):
    """Monitor training progress by watching output directory"""
    
    output_path = Path(output_dir)
    
    print("="*70)
    print("🔍 DCGAN TRAINING MONITOR")
    print("="*70)
    print(f"Monitoring: {output_path.absolute()}")
    print(f"Check interval: {check_interval} seconds")
    print(f"Press Ctrl+C to stop monitoring")
    print("="*70)
    print()
    
    last_file_count = 0
    check_count = 0
    start_time = time.time()
    
    try:
        while check_count < max_checks:
            check_count += 1
            current_time = datetime.now().strftime("%H:%M:%S")
            elapsed = time.time() - start_time
            elapsed_mins = int(elapsed // 60)
            elapsed_secs = int(elapsed % 60)
            
            print(f"\n[{current_time}] Check #{check_count} (Elapsed: {elapsed_mins}m {elapsed_secs}s)")
            print("-" * 70)
            
            if not output_path.exists():
                print("⏳ Waiting for output directory to be created...")
                time.sleep(check_interval)
                continue
            
            # List all files
            files = sorted(output_path.glob("*"), key=lambda x: x.stat().st_mtime, reverse=True)
            
            if not files:
                print("⏳ Output directory empty, waiting for training to start...")
                time.sleep(check_interval)
                continue
            
            # Count different file types
            png_files = [f for f in files if f.suffix == '.png']
            pth_files = [f for f in files if f.suffix == '.pth']
            
            print(f"📊 Files created: {len(files)} total")
            print(f"   - Images (.png): {len(png_files)}")
            print(f"   - Checkpoints (.pth): {len(pth_files)}")
            
            # Show most recent files
            print(f"\n📁 Latest files (most recent first):")
            for i, f in enumerate(files[:10]):
                size_kb = f.stat().st_size / 1024
                mtime = datetime.fromtimestamp(f.stat().st_mtime).strftime("%H:%M:%S")
                print(f"   {i+1}. {f.name:<40} {size_kb:>8.1f} KB  ({mtime})")
            
            # Parse epoch from filenames
            epoch_files = [f for f in png_files if 'epoch' in f.name]
            if epoch_files:
                # Extract epoch numbers
                epochs = []
                for f in epoch_files:
                    try:
                        # Parse "fake_samples_epoch_XXX.png"
                        parts = f.stem.split('_')
                        for i, part in enumerate(parts):
                            if part == 'epoch' and i + 1 < len(parts):
                                epoch_num = int(parts[i + 1])
                                epochs.append(epoch_num)
                                break
                    except:
                        pass
                
                if epochs:
                    current_epoch = max(epochs)
                    total_epochs = 25  # From config
                    progress_pct = (current_epoch / total_epochs) * 100
                    
                    print(f"\n🎯 Training Progress:")
                    print(f"   Current Epoch: {current_epoch}/{total_epochs} ({progress_pct:.1f}%)")
                    
                    # Progress bar
                    bar_length = 40
                    filled = int(bar_length * current_epoch / total_epochs)
                    bar = '█' * filled + '░' * (bar_length - filled)
                    print(f"   [{bar}]")
                    
                    # Estimate remaining time (rough)
                    if current_epoch > 0:
                        time_per_epoch = elapsed / (current_epoch + 1)
                        remaining_epochs = total_epochs - current_epoch
                        est_remaining = time_per_epoch * remaining_epochs
                        est_mins = int(est_remaining // 60)
                        est_hours = int(est_mins // 60)
                        est_mins = est_mins % 60
                        
                        if est_hours > 0:
                            print(f"   Estimated time remaining: ~{est_hours}h {est_mins}m")
                        else:
                            print(f"   Estimated time remaining: ~{est_mins}m")
            else:
                print(f"\n⏳ Training in progress (no epoch checkpoints yet)...")
                print(f"   Dataset loading or initial setup phase")
            
            # Check for completion
            final_files = [f for f in files if 'final' in f.name.lower()]
            training_complete_marker = [f for f in files if 'training_losses' in f.name]
            
            if final_files or training_complete_marker:
                print("\n" + "="*70)
                print("🎉 TRAINING COMPLETE!")
                print("="*70)
                print(f"Total time: {elapsed_mins}m {elapsed_secs}s")
                print(f"Total files created: {len(files)}")
                print("\nFinal outputs:")
                for f in final_files + training_complete_marker:
                    print(f"   ✓ {f.name}")
                break
            
            # Detect if new files were created
            if len(files) > last_file_count:
                new_files = len(files) - last_file_count
                print(f"\n✨ {new_files} new file(s) created since last check!")
            
            last_file_count = len(files)
            
            print(f"\n⏰ Next check in {check_interval} seconds...")
            time.sleep(check_interval)
    
    except KeyboardInterrupt:
        print("\n\n" + "="*70)
        print("🛑 Monitoring stopped by user")
        print("="*70)
        print(f"Total monitoring time: {elapsed_mins}m {elapsed_secs}s")
        print(f"Last file count: {last_file_count}")
    
    print("\n✓ Monitor finished")

if __name__ == '__main__':
    monitor_training()

