# V2 Improvements: Blues-Filtered Training

## Motivation

**V1 Problem:** Trained on 20,000 album covers across all genres → abstract/glitch aesthetic

**Root Cause:** Multi-genre dataset caused network to learn averaged patterns across diverse visual cultures

**V2 Solution:** Filter to Blues genre only + increase architecture capacity → photorealistic genre-specific outputs

---

## Key Changes from V1

### 1. Dataset Filtering
**V1:** 20,000 covers (Blues, Rock, Jazz, Hip-Hop, Electronic, Classical, Country, R&B, Soul, Punk, Metal, Folk, Reggae, Latin, World, Alternative, Indie, Pop, Dance, Funk)  
**V2:** ~1,000 covers (Blues only)

**Rationale:** 
- Genre-specific training enables learning Blues visual DNA
- Centered musician portraits
- Vintage brown/orange palette  
- Guitar/harmonica prominence
- 1950s-1970s aesthetic consistency

**Implementation:**
```python
blues_df = df[df['label'] == blues_label].copy()
# Filters to ~5% of original dataset
```

---

### 2. Increased Network Capacity
**V1:** `ngf=64, ndf=64`  
**V2:** `ngf=128, ndf=128`

**Rationale:**
- Doubled feature maps = doubled capacity to learn fine details
- Better facial features and expressions
- Clearer instrument shapes (guitars, harmonicas, microphones)
- Finer texture learning (vintage grain, analog quality)
- Improved composition understanding

**Trade-off:** 
- 4× more parameters → 4× more memory
- ~30% slower training per epoch
- Requires more data per parameter (Blues subset sufficient)

---

### 3. Larger Latent Space
**V1:** `nz=100`  
**V2:** `nz=200`

**Rationale:**
- Doubled latent dimensions = more expressive generation space
- Greater output diversity (not every cover looks identical)
- More nuanced variations (pose, lighting, composition)
- Better interpolation between styles

**Mathematical impact:**
- Latent space volume increases exponentially (2^100 → 2^200)
- More room for network to encode variation
- Less mode collapse risk

---

### 4. Extended Training
**V1:** 25 epochs  
**V2:** 35 epochs

**Rationale:**
- Smaller dataset (1k vs 20k) = less data per epoch
- Higher capacity networks = longer to converge
- Need ~40% more iterations for quality plateau

**Calculation:**
- V1: 25 epochs × 2500 batches = 62,500 iterations
- V2: 35 epochs × 125 batches = 4,375 iterations (if 1k images)
- Actually needs MORE epochs to see data sufficiently

---

### 5. Learning Rate Adjustment
**V1:** `lr_d=0.0001, lr_g=0.0002`  
**V2:** `lr_d=0.00008, lr_g=0.0002`

**Rationale:**
- Smaller dataset = D can memorize easier
- Even slower D learning (20% reduction) prevents dominance
- Maintains G/D balance with less data
- Keeps G rate same (already optimized in v1)

---

### 6. Output Directory
**V1:** `blues_output_FIXED/`  
**V2:** `blues_output_v2/`

**Rationale:**
- Preserves v1 outputs for comparison
- Side-by-side analysis of abstract vs photorealistic
- Portfolio can show both approaches

---

## Parameter Comparison Table

| Parameter | V1 (Main) | V2 (Blues-Filtered) | Change | Rationale |
|-----------|-----------|---------------------|--------|-----------|
| **Dataset** | 20k (all genres) | 1k (Blues only) | 95% reduction | Genre specificity |
| **ngf** | 64 | 128 | +100% | More detail capacity |
| **ndf** | 64 | 128 | +100% | More discrimination |
| **nz** | 100 | 200 | +100% | More expressiveness |
| **Epochs** | 25 | 35 | +40% | Convergence time |
| **lr_d** | 0.0001 | 0.00008 | -20% | Prevent D dominance |
| **lr_g** | 0.0002 | 0.0002 | 0% | Already optimal |
| **batch_size** | 8 | 8 | 0% | Memory constrained |
| **Training time** | ~18 hrs | ~20-25 hrs | +30% | Larger networks |

---

## Expected vs Actual Results

### Expected Results (Pre-Training)
- ✓ Photorealistic Blues album covers
- ✓ Recognizable musician faces
- ✓ Clear instruments (guitars prominent)
- ✓ Consistent vintage aesthetic
- ✓ Brown/orange/cream color palette
- ✓ 1950s-1970s stylistic coherence

### Actual Results (Post-Training)
**Epoch 5:**
[TO BE ADDED AFTER TRAINING]

**Epoch 10:**
[TO BE ADDED]

**Epoch 20:**
[TO BE ADDED]

**Epoch 35 (Final):**
[TO BE ADDED]

---

## Comparative Analysis

### Visual Comparison
[TO BE ADDED: Side-by-side v1 vs v2 images after training]

### Quantitative Metrics
[TO BE ADDED: Loss curves, D(x) values, training stability]

---

## Lessons Learned

### What Worked
[TO BE ADDED AFTER TRAINING]

### What Didn't Work
[TO BE ADDED AFTER TRAINING]

### Unexpected Findings
[TO BE ADDED AFTER TRAINING]

---

## Future Improvements (V3?)

Potential next steps:
1. **Resolution increase:** 256×256 or 512×512
2. **Progressive growing:** Start 32×32 → grow to 128×128
3. **StyleGAN architecture:** More control, higher quality
4. **Conditional generation:** Control specific attributes (decade, artist style)
5. **Data augmentation:** Flip, rotate, color jitter to increase effective dataset size

---

## Conclusion

V2 demonstrates systematic refinement based on V1 learnings:
- Identified issue: Multi-genre dilution
- Root cause: Dataset too diverse
- Solution: Genre filtering + capacity increase
- Method: Iterative experimentation

This process mirrors real-world ML development: train, analyze, refine, repeat.

