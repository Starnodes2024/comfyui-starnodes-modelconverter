# Star Ultimate Model Converter Pro - Enhanced Features

## Overview
The Pro converter combines profile-based quantization with models.json blacklist support and manual control options.

## Key Features

### 1. **Profile-Based Quantization**
- Uses profiles created by Star Model Layers Info
- Each layer has a specific quantization format (FP32, BF16, FP16, FP8, etc.)
- Profiles are stored in the `profiles/` folder

### 2. **models.json Blacklist Integration**
- **Model Type Dropdown**: Select your model architecture (Flux1/Flux2, Ideogram-4, SDXL, etc.)
- **Blacklist Patterns**: Automatically preserves critical layers in BF16
  - Embeddings (text_embedding, time_embedding, position_embedding)
  - Normalizations (norm, scale, bias)
  - Modulations (adaln, modulation)
  - Final layers (final_layer, proj_out, head)
- **Use Blacklist Toggle**: Enable/disable blacklist (default: yes)

### 3. **Manual Mode**
Fine-grained control over which source block types get quantized.

**How it works**: Manual mode uses the profile as a matrix. For each layer in the profile:
- If the profile says "BF16" and you set `quantize_bf16 = YES` → Converts to target format (e.g., INT4_CONVROT)
- If the profile says "BF16" and you set `quantize_bf16 = NO` → Keeps as BF16
- If the profile says "FP8_E4M3FN" and you set `quantize_fp8_e4m3fn = YES` → Converts to target format
- If the profile says "FP8_E4M3FN" and you set `quantize_fp8_e4m3fn = NO` → Keeps as-is

**Default Settings**:
- **FP32**: Default NO (preserves high precision layers)
- **BF16**: Default YES (converts to target format)
- **FP16**: Default YES (converts to target format)
- **FP8_E4M3FN**: Default YES (converts to target format)
- **FP8**: Default YES (converts to target format)

## Processing Priority (Highest to Lowest)

1. **Blacklist** (if enabled): Layers matching blacklist patterns → BF16
2. **Manual Mode** (if enabled): Override profile decisions for specific dtypes
3. **Profile Format**: Use the format specified in the profile
4. **Fallback Rules**: 
   - Missing norm/embedding layers → FP16
   - Other missing layers → FP8

## Example Workflow

### Standard Usage
1. Select your model from diffusion_models
2. Choose a profile (created by Star Model Layers Info)
3. Select model type (e.g., "Ideogram-4")
4. Keep "use_blacklist" = yes
5. Keep "manual_mode" = disabled
6. Choose target format (NVFP4, FP8, INT8, etc.)
7. Run conversion

**Result**: Profile-based quantization with critical layers preserved in BF16

### Advanced Usage with Manual Mode
1. Follow steps 1-3 above
2. Set "manual_mode" = enabled
3. Configure which source blocks to quantize:
   - Set "quantize_fp32" = yes to quantize FP32 layers
   - Set "quantize_bf16" = no to preserve BF16 layers
   - Adjust other settings as needed
4. Run conversion

**Result**: Custom quantization with full control over each data type

## Blacklist Examples from models.json

### Ideogram-4
Preserves: bias, norm, scale, final_layer, embeddings, modulations, q_norm, k_norm

### Flux1/Flux2
Preserves: bias, txt_attn, img_in, txt_in, time_in, vector_in, guidance_in, final_layer, class_embedding, modulations

### SDXL
Preserves: time_embed, label_emb, add_time_embed, add_embed, out, bias, norm

### Unknown (Fallback)
Preserves: Most common critical layers across all architectures

## Console Output

The converter provides detailed logging:
- Blacklist patterns loaded
- Manual mode settings (if enabled)
- Conversion statistics by format
- Processing progress every 100 layers

## Tips

1. **Use the correct model_type** for best results - it ensures critical layers are preserved
2. **Start with defaults** (blacklist=yes, manual_mode=disabled) for most cases
3. **Enable manual mode** only when you need fine-grained control
4. **Check console output** to verify blacklist patterns are loading correctly
5. **Profile + Blacklist** work together: Profile defines quantization strategy, blacklist protects critical layers
