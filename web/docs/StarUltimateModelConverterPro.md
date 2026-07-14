# ⭐ Star Ultimate Model Converter Pro

Advanced profile-based converter that applies quantization strategies from saved profiles to new models with strict conversion rules.

## Purpose

The Star Ultimate Model Converter Pro enables reproducible, profile-based quantization. Extract the layer-by-layer quantization blueprint from a well-optimized model and apply that same strategy to new models, ensuring consistent quality and compression across your model pipeline.

## Inputs

### Required

- **model_name**: Select a diffusion model from your `models/diffusion_models` folder
- **profile**: Select a quantization profile from dropdown (created by Star Model Layers Info)
- **model_type**: Select model architecture for blacklist (Ideogram-4, Flux1/Flux2, SDXL, etc.)
- **use_blacklist**: Enable/disable models.json blacklist protection (default: yes)
- **target_quant_format**: Deep quantization format for heavily compressed layers
  - NVFP4, INT4_CONVROT, FP8, INT8, INT8_CONVROT, MXFP8
- **manual_mode**: Enable manual control over quantization (default: disabled)
- **quantize_fp32**: [Manual Mode] Convert FP32 layers to FP8 (default: no)
- **quantize_bf16**: [Manual Mode] Convert BF16 layers to FP8 (default: yes)
- **quantize_fp16**: [Manual Mode] Convert FP16 layers to FP8 (default: yes)
- **quantize_fp8_e4m3fn**: [Manual Mode] Convert FP8_E4M3FN layers to FP8 (default: yes)
- **quantize_fp8**: [Manual Mode] Convert FP8 layers to FP8 (default: yes)
- **device**: Processing device (cuda for GPU, cpu for CPU)

### Optional

- **output_name**: Custom name for output model (leave blank for auto-naming)

## Outputs

- **status**: Conversion summary with format breakdown and statistics
- **model**: Converted MODEL object ready for immediate use

## How It Works

1. **Load Profile**: Reads the selected JSON profile from `profiles/` folder
2. **Load Blacklist**: Loads layer protection patterns from `models.json` based on model_type
3. **Extract State Dict**: Gets the input model's state dictionary
4. **Apply Priority Rules**: For each layer:
   - **Priority 1**: Blacklist → Preserve in BF16 (highest priority)
   - **Priority 2**: Manual Mode → User-controlled quantization
   - **Priority 3**: Profile Format → Use profile specification
   - **Priority 4**: Fallback → Safe defaults
5. **Deep Quantization**: Actually quantize using comfy-kitchen (NVFP4, INT4_CONVROT, etc.)
6. **Save with Metadata**: Writes converted model with quantization metadata
7. **Load**: Returns converted model for output connector

## New Features

### 🚫 models.json Blacklist Integration

The converter now uses the same blacklist system as Star Ultimate Model Converter:
- **15+ Model Profiles**: Ideogram-4, Flux1/Flux2, SDXL, Qwen, LTX, and more
- **Pattern Matching**: Automatically preserves critical layers (embeddings, norms, modulations)
- **Priority Override**: Blacklist takes precedence over profile settings
- **Toggle Control**: Can be disabled if you want pure profile-based conversion

**Example Blacklist Patterns** (Ideogram-4):
```
bias, norm, scale, final_layer, modulation, 
adaln, q_norm, k_norm, embeddings, etc.
```

### 🎮 Manual Mode

Fine-grained control over which source block types get quantized:

**How it works**: Manual mode uses the profile as a matrix:
- Profile says "BF16" + You set `quantize_bf16 = YES` → Converts to FP8
- Profile says "BF16" + You set `quantize_bf16 = NO` → Keeps as BF16
- Profile says "FP8_E4M3FN" + You set `quantize_fp8_e4m3fn = YES` → Converts to FP8
- Profile says "FP8_E4M3FN" + You set `quantize_fp8_e4m3fn = NO` → Keeps as-is

**Default Settings**:
- FP32: NO (preserves high precision)
- BF16: YES (converts to FP8)
- FP16: YES (converts to FP8)
- FP8_E4M3FN: YES (converts to FP8)
- FP8: YES (converts to FP8)

### 💎 Actual Deep Quantization

The converter now performs real quantization using comfy-kitchen:
- **NVFP4**: 4-bit NVIDIA floating point with Tensor Cores
- **INT4_CONVROT**: 4-bit with Hadamard rotation (group size 256/64)
- **INT8_CONVROT**: 8-bit with rotation (group size 256)
- **MXFP8**: Microscaling FP8 with block scaling
- **FP8**: 8-bit floating point with per-tensor scaling

Each quantized layer includes proper metadata for ComfyUI loading.

## Processing Priority (Highest to Lowest)

1. **Blacklist** (if enabled): Layers matching blacklist patterns → BF16
2. **Manual Mode** (if enabled): Override profile decisions for specific dtypes
3. **Profile Format**: Use the format specified in the profile
4. **Fallback Rules**: 
   - Missing norm/embedding layers → FP16
   - Other missing layers → FP8

## The 5 Conversion Rules

### Rule 1: FP32 Preservation
**Profile Layer**: `FP32`  
**Action**: Keep high precision (no quantization)  
**Reason**: Critical layers that need full precision

```
Profile: layer.weight = "FP32"
Input:   layer.weight = FP16
Output:  layer.weight = FP16 (kept as-is)
```

### Rule 2: Half Precision
**Profile Layer**: `FP16` or `BF16`  
**Action**: Convert to half precision (only if input is higher)  
**Reason**: Balanced precision layers

```
Profile: layer.weight = "FP16"
Input:   layer.weight = FP32
Output:  layer.weight = FP16
```

### Rule 3: 8-bit Quantization
**Profile Layer**: `FP8` or `INT8`  
**Action**: Convert to 8-bit format  
**Reason**: Moderate compression layers

```
Profile: layer.weight = "FP8"
Input:   layer.weight = FP32
Output:  layer.weight = FP8
```

### Rule 4: Deep Quantization
**Profile Layer**: `NVFP4`, `INT4_CONVROT`, `TORCH.UINT8`, etc.  
**Action**: Actually quantize using comfy-kitchen to `target_quant_format`  
**Reason**: Heavy compression layers

```
Profile: layer.weight = "TORCH.UINT8 + SCALE"
Input:   layer.weight = FP32
Target:  INT4_CONVROT
Output:  layer.weight (quantized) + layer.weight_scale + layer.weight_scale_2
        Multiple tensors with proper quantization metadata
```

**Note**: Deep quantization only applies to 2D weight tensors. Biases and other tensors are kept as BF16.

### Rule 5: Missing Layer Fallback
**Profile Layer**: Not in profile  
**Action**: FP8 by default, FP16 for norms/embeddings  
**Reason**: Safe fallback prevents NaN errors

```
Profile: (layer not found)
Input:   new_layer.weight = FP32
Output:  new_layer.weight = FP8

Profile: (norm layer not found)
Input:   norm.scale = FP32
Output:  norm.scale = FP16 (safe fallback)
```

## Profile Tooltip

Hover over the profile dropdown to see metadata:

```
┌──────────────────────────────────┐
│ 📋 Profile Information           │
│                                  │
│ Model: flux-dev-nvfp4            │
│ Created: 1/13/2025, 8:15:30 PM   │
│ Layers: 1234                     │
│                                  │
│ Created by: Star Model Layers... │
└──────────────────────────────────┘
```

## Example Workflows

### Workflow 1: Standard Profile-Based Conversion

```
[Star Ultimate Model Converter Pro]
  - model_name: ideogram4_fp8_scaled.safetensors
  - profile: ideogram4_nvfp4_mixed.json
  - model_type: Ideogram-4
  - use_blacklist: yes
  - target_quant: INT4_CONVROT
  - manual_mode: disabled
  - device: cuda
  ↓
Output: ideogram4_int4_convrot_pro.safetensors

Result:
  - Blacklisted layers (499): BF16
  - Deep quant layers (170): INT4_CONVROT
  - Other layers: Follow profile
```

### Workflow 2: Manual Mode - Preserve Quality

```
[Star Ultimate Model Converter Pro]
  - model_name: flux-schnell-fp16.safetensors
  - profile: flux-dev-nvfp4.json
  - model_type: Flux1 / Flux2
  - use_blacklist: yes
  - target_quant: NVFP4
  - manual_mode: enabled
  - quantize_fp32: no
  - quantize_bf16: no  ← Keep BF16 as-is
  - quantize_fp16: no  ← Keep FP16 as-is
  - quantize_fp8_e4m3fn: yes
  - quantize_fp8: yes
  - device: cuda
  ↓
Output: Higher quality, larger file
```

### Workflow 3: Apply Existing Profile

```
[Star Ultimate Model Converter Pro]
  - model_name: flux-schnell-fp16.safetensors
  - profile: flux-dev-nvfp4.json
  - target_quant: NVFP4
  - device: cuda
  ↓
Output: flux-schnell-fp16-nvfp4-pro.safetensors
```

### Workflow 2: Create and Apply Profile

```
Step 1: Create Profile
[Star Model Layers Info]
  - model: flux-dev-nvfp4
  - save_profile: ✓ True
  ↓
profiles/flux-dev-nvfp4.json

Step 2: Apply to New Model
[Star Ultimate Model Converter Pro]
  - model_name: flux-schnell-fp16.safetensors
  - profile: flux-dev-nvfp4.json
  - target_quant: NVFP4
  ↓
flux-schnell-fp16-nvfp4-pro.safetensors
```

### Workflow 3: Different Target Format

```
[Star Ultimate Model Converter Pro]
  - model_name: sdxl-base-fp16.safetensors
  - profile: flux-dev-nvfp4.json
  - target_quant: FP8
  ↓
sdxl-base-fp16-fp8-pro.safetensors
```

## Use Cases

### 1. Consistent Pipeline
Apply the same quantization strategy across all models in your pipeline.

### 2. Quality Preservation
Use profiles from well-tested models to ensure quality on new models.

### 3. Rapid Deployment
Quickly quantize new models using proven profiles.

### 4. A/B Testing
Test different profiles on the same model to find optimal settings.

## Status Output

The node provides detailed conversion breakdown:

```
✅ Profile-based conversion complete
Profile: ideogram4_nvfp4_mixed
Target format: INT4_CONVROT
Total layers: 880

Conversion breakdown:
  - blacklisted_to_bf16: 499 layers
  - int4_convrot: 170 layers
  - kept_non_float: 211 layers

Output size: 2.45 GB
Time: 19.4s
Saved to: E:\...\models\unet\Ideogramm4_int4_convrot.safetensors
```

**Manual Mode Example**:
```
Conversion breakdown:
  - blacklisted_to_bf16: 499 layers
  - manual_kept_bf16: 150 layers
  - manual_fp16_to_fp8: 50 layers
  - int4_convrot: 170 layers
  - kept_non_float: 211 layers
```

## Profile Management

### Creating Profiles

Use Star Model Layers Info with `save_profile` enabled:

```
[Star Model Layers Info]
  - model: your-optimized-model
  - save_profile: ✓ True
  
→ profiles/your-model.json
```

### Profile Structure

```json
{
  "__metadata__": {
    "original_model_name": "flux-dev-nvfp4",
    "timestamp": "2025-01-13T20:15:30",
    "total_layers": 1234
  },
  "layers": {
    "double_blocks.0.img_attn.proj.weight": "NVFP4",
    "double_blocks.0.img_attn.norm.scale": "BF16",
    ...
  }
}
```

### Profile Location

Profiles are stored in: `comfyui-starnodes-modelconverter/profiles/`

## Tips

- **Profile Selection**: Choose profiles from similar model architectures
- **Target Format**: Use NVFP4 for maximum compression, FP8 for balance
- **Device**: CUDA is much faster than CPU
- **Custom Naming**: Use descriptive output names for easy identification
- **Profile Library**: Build a library of profiles for different model types

## Troubleshooting

### "No profiles found"
- Create a profile using Star Model Layers Info
- Check `profiles/` folder exists
- Verify `.json` files are present

### Tooltip not showing
- Check browser console for errors
- Verify API routes loaded (check ComfyUI console)
- Restart ComfyUI

### Conversion produces NaN
- Profile may have aggressive quantization on norms
- Edit profile: Set norm layers to FP16/BF16
- Use Rule 5 Exception (automatic for new profiles)

### Profile not loading
- Validate JSON structure
- Check file encoding (UTF-8)
- Verify `__metadata__` and `layers` keys exist

## Advanced Usage

### Custom Profiles

Manually create or edit profiles:

```json
{
  "__metadata__": {
    "original_model_name": "custom-profile",
    "timestamp": "2025-01-13T20:00:00",
    "total_layers": 100
  },
  "layers": {
    "encoder.layers.0.weight": "FP16",
    "encoder.layers.1.weight": "FP8",
    "decoder.layers.0.weight": "NVFP4"
  }
}
```

### Profile Merging

Combine multiple profiles programmatically:

```python
import json

# Load profiles
with open("profiles/profile1.json") as f:
    p1 = json.load(f)
with open("profiles/profile2.json") as f:
    p2 = json.load(f)

# Merge (p2 overrides p1)
merged = {
    "__metadata__": p1["__metadata__"],
    "layers": {**p1["layers"], **p2["layers"]}
}

# Save
with open("profiles/merged.json", "w") as f:
    json.dump(merged, f, indent=2)
```

## Requirements

- ComfyUI (v0.27.0+ for INT8/INT4 ConvRot)
- Quantization profile (created by Star Model Layers Info)
- **comfy-kitchen** (required for deep quantization: NVFP4, INT4_CONVROT, MXFP8, etc.)
- PyTorch 2.0+ with CUDA
- NVIDIA GPU (required for NVFP4, recommended for others)
- models.json (included, defines blacklist patterns)

## Technical Details

### Dtype Mapping

```python
"FP32" → torch.float32
"FP16" → torch.float16
"BF16" → torch.bfloat16
"FP8"  → torch.float8_e4m3fn
"INT8" → torch.int8
```

### Normalization Detection

Automatically detects norm/embedding layers:
- `norm`, `ln`, `layernorm`, `groupnorm`, `batchnorm`
- `embed`, `embedding`, `token`, `position`

### Memory Management

- Efficient state dict extraction
- Contiguous memory allocation
- VRAM spike prevention

## Credits

Part of the Starnodes Model Converter suite for advanced profile-based quantization.
