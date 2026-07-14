# ⭐ Star Ultimate Model Converter

Convert and quantize diffusion models to various precision formats with intelligent layer-specific optimization.

## Purpose

The Star Ultimate Model Converter provides advanced quantization capabilities for diffusion models, allowing you to reduce model size while maintaining quality through architecture-specific layer preservation.

## Inputs

### Required

- **model_name**: Select a diffusion model from your `models/diffusion_models` folder
- **model_type**: Choose the model architecture profile (Flux, SDXL, LTX, etc.)
- **target_format**: Target quantization format (NVFP4, FP8, MXFP8, INT8, INT8 ConvRot, INT4 ConvRot, FP16, FP32)
- **device**: Processing device (cuda for GPU, cpu for CPU)

### Optional

- **use_custom_path**: Enable to use a custom file path
- **custom_path**: Full path to .safetensors file or folder with shards
- **use_aio_checkpoint**: Enable to convert from an AIO checkpoint
- **checkpoint_name**: Select AIO checkpoint (only diffusion model is extracted)

## Outputs

- **status**: Detailed conversion summary with statistics
- **MODEL** (optional): Converted model ready for immediate use

## How It Works

1. **Load Model**: Reads the source model from the selected location
2. **Detect Format**: Automatically detects and dequantizes pre-quantized models
3. **Apply Profile**: Uses architecture-specific blacklists to preserve critical layers
4. **Quantize**: Converts layers to target format using comfy-kitchen
5. **Save**: Writes converted model with proper metadata

## Supported Formats

### NVFP4 (4-bit)
- **Size**: ~25% of original
- **Quality**: Excellent
- **Hardware**: NVIDIA GPUs only
- **Best for**: Maximum compression with minimal quality loss

### FP8 (8-bit)
- **Size**: ~50% of original
- **Quality**: Very good
- **Hardware**: NVIDIA GPUs recommended
- **Best for**: Balanced compression and quality

### MXFP8 (8-bit Microscaling)
- **Size**: ~50% of original
- **Quality**: Near FP16
- **Hardware**: Growing support (OCP standard)
- **Best for**: High quality with good compression

### INT8 (8-bit Integer)
- **Size**: ~50% of original
- **Quality**: Good
- **Hardware**: Most hardware
- **Best for**: Wide compatibility

### INT8 ConvRot (8-bit with Rotation)
- **Size**: ~50% of original
- **Quality**: Very good
- **Hardware**: ComfyUI v0.27.0+
- **Best for**: Better quality than plain INT8

### INT4 ConvRot (4-bit with Rotation)
- **Size**: ~25% of original
- **Quality**: Excellent for 4-bit
- **Hardware**: SM 8.0+ (Ampere/Ada/Blackwell)
- **Best for**: Maximum compression with rotation

### FP16 (Half Precision)
- **Size**: ~50% of FP32
- **Quality**: Excellent
- **Hardware**: All hardware
- **Best for**: Standard precision

### FP32 (Full Precision)
- **Size**: 100% (no compression)
- **Quality**: Perfect
- **Hardware**: All hardware
- **Best for**: Maximum quality

## Model Type Profiles

Each profile includes carefully tuned blacklists:

- **Flux Family**: Optimized for Flux.1/Flux.2 models
- **SDXL**: Stable Diffusion XL architecture
- **LTX**: LTX video models
- **Qwen**: Qwen image models
- **Default**: Generic profile for unknown architectures

## Tips

- **Choose the right profile**: Select the model_type that matches your model architecture
- **Start with FP8**: Good balance of size and quality for testing
- **Use CUDA**: GPU conversion is 10-50x faster than CPU
- **Check memory**: Large models may require 64GB+ RAM
- **Custom paths**: Load models from anywhere, including HuggingFace cache

## Example Workflow

```
[Star Ultimate Model Converter]
  - model_name: flux-dev-fp16
  - model_type: Flux Family
  - target_format: nvfp4
  - device: cuda
  
→ flux-dev-nvfp4.safetensors (11.9 GB → 3.0 GB)
```

## Advanced Features

### Automatic Dequantization
Detects and dequantizes pre-quantized models for clean re-quantization.

### Multi-Shard Support
Automatically processes models split across multiple .safetensors files.

### Metadata Preservation
Maintains model metadata and quantization information for proper loading.

### Layer Blacklisting
Architecture-specific profiles preserve critical layers in high precision.

## Troubleshooting

### "comfy-kitchen not found"
Install comfy-kitchen: `pip install comfy-kitchen`

### "CUDA out of memory"
- Use `device: cpu` (slower but uses system RAM)
- Close other applications
- Use a smaller target format

### Model quality issues
- Verify correct model_type profile
- Try less aggressive format (FP8 instead of NVFP4)
- Check if source model is already quantized

## Requirements

- ComfyUI v0.27.0+ (for INT8/INT4 ConvRot)
- comfy-kitchen (for quantization)
- PyTorch 2.0+ with CUDA
- NVIDIA GPU (recommended)
- 64GB+ RAM (for large models)

## Credits

Part of the Starnodes Model Converter suite for comprehensive model quantization and management.
