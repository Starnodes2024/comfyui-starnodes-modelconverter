# Star Ultimate Model Converter Pro v1.3.0 Updates

## Summary

The Pro converter has been significantly enhanced with models.json blacklist integration, manual mode control, and actual deep quantization using comfy-kitchen.

## What's New

### 🚫 models.json Blacklist Integration

The converter now uses the same intelligent blacklist system as Star Ultimate Model Converter:

- **15+ Model Profiles**: Ideogram-4, Flux1/Flux2, SDXL, Qwen, LTX, Krea, Wan, ERNIE, Lens, and more
- **Pattern Matching**: Automatically preserves critical layers (embeddings, norms, modulations, final layers)
- **Priority Override**: Blacklist takes highest priority over all other settings
- **Toggle Control**: Can be disabled for pure profile-based conversion

**Example**: Ideogram-4 blacklist preserves 499 layers including bias, norm, scale, final_layer, modulation, adaln, q_norm, k_norm, embeddings, etc.

### 🎮 Manual Mode

Fine-grained control over which source block types get quantized:

**How it works**: Manual mode uses the profile as a matrix:
- Profile says "BF16" + You set `quantize_bf16 = YES` → Converts to FP8
- Profile says "BF16" + You set `quantize_bf16 = NO` → Keeps as BF16
- Profile says "FP8_E4M3FN" + You set `quantize_fp8_e4m3fn = YES` → Converts to FP8
- Profile says "FP8_E4M3FN" + You set `quantize_fp8_e4m3fn = NO` → Keeps as-is

**Controls**:
- `quantize_fp32`: Default NO (preserves high precision)
- `quantize_bf16`: Default YES (converts to FP8)
- `quantize_fp16`: Default YES (converts to FP8)
- `quantize_fp8_e4m3fn`: Default YES (converts to FP8)
- `quantize_fp8`: Default YES (converts to FP8)

### 💎 Actual Deep Quantization

The converter now performs **real quantization** using comfy-kitchen instead of FP8 fallback:

- **NVFP4**: 4-bit NVIDIA floating point with Tensor Cores
- **INT4_CONVROT**: 4-bit with Hadamard rotation (group size 256/64)
- **INT8_CONVROT**: 8-bit with rotation (group size 256)
- **MXFP8**: Microscaling FP8 with block scaling
- **FP8**: 8-bit floating point with per-tensor scaling

Each quantized layer includes:
- Multiple tensors (weight + scales)
- Proper quantization metadata
- ComfyUI-compatible format

### 📊 Priority System

Processing order (highest to lowest):

1. **Blacklist** (if enabled): Layers matching blacklist patterns → BF16
2. **Manual Mode** (if enabled): Override profile decisions for specific dtypes
3. **Profile Format**: Use the format specified in the profile
4. **Fallback Rules**: 
   - Missing norm/embedding layers → FP16
   - Other missing layers → FP8

### 🔧 Technical Improvements

- **Format Normalization**: Handles complex profile formats like "FP8_E4M3FN + SCALE", "TORCH.UINT8 + SCALE"
- **Proper Metadata**: Saves quantization metadata to safetensors for ComfyUI compatibility
- **Result Handling**: Supports both single tensors and dictionaries (for multi-tensor quantization)
- **File Saving**: Fixed - files are now actually saved with proper metadata
- **JavaScript Loading**: Added web/index.js for ComfyUI to load profile tooltips

## New Inputs

- `model_type`: Select model architecture for blacklist (dropdown with 15+ options)
- `use_blacklist`: Enable/disable blacklist protection (yes/no, default: yes)
- `manual_mode`: Enable manual control (disabled/enabled, default: disabled)
- `quantize_fp32`: [Manual Mode] Convert FP32 layers (yes/no, default: no)
- `quantize_bf16`: [Manual Mode] Convert BF16 layers (yes/no, default: yes)
- `quantize_fp16`: [Manual Mode] Convert FP16 layers (yes/no, default: yes)
- `quantize_fp8_e4m3fn`: [Manual Mode] Convert FP8_E4M3FN layers (yes/no, default: yes)
- `quantize_fp8`: [Manual Mode] Convert FP8 layers (yes/no, default: yes)

## Example Output

### Before (v1.2.0)
```
Conversion breakdown:
  - blacklisted_to_bf16: 499 layers
  - kept_non_float: 211 layers
  - deep_quant_to_int4_convrot: 170 layers  ← Just FP8 fallback

Output size: 9.22 GB  ← Not actually quantized
File saved: ✗ (not created)
```

### After (v1.3.0)
```
Conversion breakdown:
  - blacklisted_to_bf16: 499 layers
  - int4_convrot: 170 layers  ← Actually quantized!
  - kept_non_float: 211 layers

Output size: 2.45 GB  ← Real compression
File saved: ✓ E:\...\models\unet\Ideogramm4_int4_convrot.safetensors
```

## Files Updated

### Code
- `star_model_converter_pro.py`: Complete rewrite of conversion logic
  - Added comfy-kitchen imports
  - Implemented actual deep quantization
  - Added blacklist integration
  - Added manual mode
  - Fixed file saving with metadata

### Documentation
- `web/docs/StarUltimateModelConverterPro.md`: Complete documentation update
- `README.md`: Updated Pro converter section
- `CONVERTER_PRO_FEATURES.md`: Updated feature documentation
- `UPDATES_v1.3.0.md`: This file

### JavaScript
- `web/index.js`: Created to load profile_tooltip.js (fixes tooltip loading)

## Migration Guide

### For Existing Users

No breaking changes! Your existing workflows will continue to work with default settings:
- `model_type`: Defaults to "Unknown" (safe fallback blacklist)
- `use_blacklist`: Defaults to "yes" (recommended)
- `manual_mode`: Defaults to "disabled" (profile-based conversion)

### Recommended Settings

**Standard Use** (Best Quality):
```
- model_type: [Match your model]
- use_blacklist: yes
- manual_mode: disabled
```

**Maximum Compression**:
```
- model_type: [Match your model]
- use_blacklist: no
- manual_mode: enabled
- quantize_fp32: yes
- quantize_bf16: yes
- quantize_fp16: yes
```

**Quality Preservation**:
```
- model_type: [Match your model]
- use_blacklist: yes
- manual_mode: enabled
- quantize_fp32: no
- quantize_bf16: no
- quantize_fp16: no
```

## Requirements

- **comfy-kitchen**: Required for deep quantization (install via pip)
- **ComfyUI**: v0.27.0+ for INT8/INT4 ConvRot
- **PyTorch**: 2.0+ with CUDA
- **NVIDIA GPU**: Required for NVFP4, recommended for others

## Credits

Special thanks to the Comfy-Org team for comfy-kitchen, which powers all deep quantization formats.
