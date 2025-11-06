# Dataset Filtering Methodology

## Problem Statement

### V1 Multi-Genre Dilution

**Dataset composition:**
- 20,000 album covers
- 20 genres (Blues, Rock, Jazz, Hip-Hop, Electronic, Classical, Country, R&B, Soul, Punk, Metal, Folk, Reggae, Latin, World, Alternative, Indie, Pop, Dance, Funk)
- ~1,000 covers per genre

**Network learning:**
- Averaged visual patterns across all genres
- No genre-specific aesthetic coherence
- Generic "album cover-ness" rather than Blues identity

**Result:**
- Abstract/glitch aesthetic
- Surreal blended forms
- No recognizable genre characteristics

---

## Solution: Genre Isolation

Filter dataset to Blues genre only before training.

**Benefits:**
- Learn Blues-specific visual patterns
- Consistent color palette (vintage browns/oranges)
- Recognizable iconography (guitars, Delta imagery)
- Cultural aesthetic coherence

---

## Implementation Steps

### Step 1: Inspect Dataset Structure
```python
import pandas as pd

# Load dataset
df = pd.read_parquet("hf://datasets/eong/20k-Album-Covers-within-20-Genres/data/train-00000-of-00001-f37f5042abc5be8d.parquet")

print(f"Total covers: {len(df)}")
print(f"Columns: {list(df.columns)}")
print(f"\nLabel column type: {df['label'].dtype}")
```

**Expected output:**
```
Total covers: 20000
Columns: ['image', 'label']
Label column type: int64  # or object (string)
```

---

### Step 2: Analyze Label Distribution
```python
# Check unique labels
print(f"Unique labels: {sorted(df['label'].unique())}")
print(f"Number of unique labels: {df['label'].nunique()}")

# Get distribution
label_counts = df['label'].value_counts().sort_index()
print("\nLabel distribution:")
print(label_counts)
```

**Expected output:**
```
Unique labels: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
Number of unique labels: 20

Label distribution:
0     1000
1     1000
2     1000
...
19    1000
```

---

### Step 3: Identify Blues Label

**Method A: If labels are numeric (0-19)**

Inspect the HuggingFace dataset to find the mapping:
```python
from datasets import load_dataset

# Load just the dataset info
dataset = load_dataset("eong/20k-Album-Covers-within-20-Genres", split='train')

# Get label names
label_names = dataset.features['label'].names
print("Label mapping:")
for idx, name in enumerate(label_names):
    print(f"{idx}: {name}")
```

**Expected output:**
```
Label mapping:
0: Blues
1: Rock
2: Jazz
3: Hip-Hop
...
19: Funk
```

**Find Blues:**
```python
blues_idx = label_names.index('Blues')
print(f"\nBlues label index: {blues_idx}")
```

**Method B: If labels are strings**

If the dataset already has string labels:
```python
# Check if labels are strings
print(df['label'].head())

# Filter using string matching
blues_df = df[df['label'].str.contains('Blues', case=False, na=False)].copy()
```

---

### Step 4: Filter Dataset
```python
# Numeric labels (most common)
blues_label = 0  # Adjust based on Step 3
blues_df = df[df['label'] == blues_label].copy()

# Verify filtering
print(f"Original dataset: {len(df)} covers")
print(f"Blues-filtered: {len(blues_df)} covers")
print(f"Percentage: {len(blues_df)/len(df)*100:.1f}%")

# Reset index for clean iteration
blues_df = blues_df.reset_index(drop=True)
```

**Expected output:**
```
Original dataset: 20000 covers
Blues-filtered: 1000 covers
Percentage: 5.0%
```

---

### Step 5: Validation Checks

**Check 1: Sufficient data**
```python
if len(blues_df) < 100:
    print("⚠️  WARNING: Very small dataset - may not generalize well")
elif len(blues_df) < 500:
    print("⚠️  Small dataset - expect some overfitting")
elif len(blues_df) < 1000:
    print("✓ Adequate dataset size")
else:
    print("✓ Good dataset size - sufficient diversity")
```

**Check 2: Label consistency**
```python
# Verify all filtered rows have the correct label
assert (blues_df['label'] == blues_label).all(), "Filtering error detected!"
print("✓ All filtered images have correct label")
```

**Check 3: Sample visualization**
```python
import matplotlib.pyplot as plt
from PIL import Image
import io

# Display first 9 Blues covers
fig, axes = plt.subplots(3, 3, figsize=(10, 10))
for idx, ax in enumerate(axes.flat):
    if idx < len(blues_df):
        img_data = blues_df.iloc[idx]['image']
        if isinstance(img_data, dict):
            img_bytes = img_data['bytes']
        else:
            img_bytes = img_data
        img = Image.open(io.BytesIO(img_bytes))
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f"Blues #{idx}")

plt.tight_layout()
plt.savefig('blues_sample_filtered.png')
print("✓ Sample visualization saved")
```

---

## Integration with Training Pipeline

### Before (V1):
```python
# V1: Use all genres
df = pd.read_parquet("...")
dataset = AlbumCoverDataset(df, transform=transform)  # All 20k
```

### After (V2):
```python
# V2: Filter to Blues only
df = pd.read_parquet("...")

# Filter
blues_df = df[df['label'] == blues_label].copy()
blues_df = blues_df.reset_index(drop=True)

# Use filtered dataset
dataset = AlbumCoverDataset(blues_df, transform=transform)  # Only ~1k
```

**Impact:**
- Batches per epoch: 2500 → 125 (20× reduction)
- Training time per epoch: Similar (batch size same)
- Total training time: May be slightly less, but need more epochs
- Memory usage: Same (batch size unchanged)

---

## Common Issues & Solutions

### Issue 1: Wrong Label Index
**Symptom:** Filtered dataset is empty or wrong size
```python
blues_df = df[df['label'] == 0].copy()
print(f"Result: {len(blues_df)} covers")  # Returns 0 or wrong number
```

**Solution:** Double-check the label mapping
```python
# Always verify first
print(df['label'].value_counts())
print(dataset.features['label'].names)
```

---

### Issue 2: Label Type Mismatch
**Symptom:** Filtering fails with type error
```python
# Error if labels are strings but you use int
blues_df = df[df['label'] == 0].copy()  # May not work if labels are strings
```

**Solution:** Check dtype first
```python
if df['label'].dtype == 'object':
    # String labels
    blues_df = df[df['label'] == 'Blues'].copy()
else:
    # Numeric labels
    blues_label = 0  # (verify correct index)
    blues_df = df[df['label'] == blues_label].copy()
```

---

### Issue 3: Data Leakage
**Symptom:** Training looks great, validation poor
**Cause:** Not resetting index after filtering

**Solution:** Always reset index
```python
blues_df = df[df['label'] == blues_label].copy()
blues_df = blues_df.reset_index(drop=True)  # CRITICAL
```

---

### Issue 4: Insufficient Diversity
**Symptom:** Mode collapse, all generated images look identical
**Cause:** Blues subset too small or not representative

**Solutions:**
1. **Data augmentation:**
   ```python
   transform = transforms.Compose([
       transforms.Resize(128),
       transforms.RandomHorizontalFlip(p=0.5),  # Add flip
       transforms.ColorJitter(0.1, 0.1, 0.1),   # Add color variation
       transforms.CenterCrop(128),
       transforms.ToTensor(),
       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
   ])
   ```

2. **Increase network capacity:** (Already done in V2)
   - `ngf=128, ndf=128` provides more parameters to learn diversity

3. **More training epochs:** (Already done in V2)
   - 35 epochs ensures network sees data many times

---

## Verification Checklist

Before training V2, verify:

- [ ] **Dataset loaded:** `df` contains 20,000 rows
- [ ] **Labels present:** `'label'` column exists
- [ ] **Blues identified:** Correct label index/value determined
- [ ] **Filtered correctly:** `blues_df` contains ~1,000 rows (5%)
- [ ] **Index reset:** No gaps in index (0, 1, 2, ..., 999)
- [ ] **All Blues:** All rows have same label value
- [ ] **Images intact:** Sample visualization shows Blues covers
- [ ] **DataLoader works:** Test batch loads without error

```python
# Quick verification script
print("="*60)
print("FILTERING VERIFICATION")
print("="*60)
print(f"✓ Original dataset: {len(df)} covers")
print(f"✓ Filtered dataset: {len(blues_df)} covers ({len(blues_df)/len(df)*100:.1f}%)")
print(f"✓ Index range: {blues_df.index.min()} to {blues_df.index.max()}")
print(f"✓ All same label: {(blues_df['label'] == blues_label).all()}")

# Test DataLoader
dataset = AlbumCoverDataset(blues_df, transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
test_batch = next(iter(dataloader))
print(f"✓ Test batch shape: {test_batch[0].shape}")
print("="*60)
```

---

## Expected Results After Filtering

### Visual Characteristics (Blues Genre)

**Common elements:**
- **Musicians:** Often solo artists, centered portraits
- **Instruments:** Acoustic/electric guitars, harmonicas, microphones
- **Clothing:** Casual, often denim, hats
- **Setting:** Studio shots, outdoor rural scenes, stages
- **Typography:** Bold, often hand-drawn fonts
- **Color palette:** Browns, oranges, creams, earth tones
- **Era:** 1950s-1970s aesthetic (classic Blues era)
- **Composition:** Simple, direct, performer-focused

**What to avoid (not Blues):**
- ❌ Electronic/digital aesthetics (Electronic genre)
- ❌ Minimalist typography (Classical genre)
- ❌ Urban/street imagery (Hip-Hop genre)
- ❌ Band group shots (Rock genre)
- ❌ Bright neon colors (Dance/Pop genres)

---

## Post-Training Analysis

After V2 training completes, compare:

**V1 (Multi-Genre):**
- Abstract forms
- Mixed visual styles
- Unpredictable color palettes
- Surreal compositions

**V2 (Blues-Filtered):**
- Consistent genre aesthetic
- Recognizable Blues elements
- Coherent color schemes
- Genre-appropriate compositions

**Success metrics:**
1. **Visual coherence:** Do outputs look like Blues covers?
2. **Element presence:** Guitars, musicians, vintage aesthetic?
3. **Color consistency:** Brown/orange earth tones?
4. **Genre specificity:** Could you identify genre from generated image?

---

## Conclusion

Genre filtering transforms GAN training from generic image generation to genre-specific synthesis. This methodology:

1. **Reduces noise:** Eliminates cross-genre confusion
2. **Focuses learning:** Network learns one aesthetic well
3. **Improves quality:** Photorealism vs abstract art
4. **Enables analysis:** Clear V1 vs V2 comparison

The filtered approach demonstrates how dataset curation drives model behavior more than architecture alone.

