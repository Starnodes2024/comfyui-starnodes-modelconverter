# ⭐ Star Ultimate Model Converter

Convert and quantize diffusion models to various precision formats with intelligent layer-specific optimization.

## Purpose

The Star Ultimate Model Converter provides advanced quantization capabilities for diffusion models, allowing you to reduce model size while maintaining quality through architecture-specific layer preservation.

## Inputs

### Required

- **mode**: What you are converting — `Diffusion Model`, `Checkpoint`, `AIO`, `Text-Encoder`, `VAE` or `Custom Path`
- **diffusion_model / checkpoint / text_encoder / vae**: Source file from the matching ComfyUI folder (only the one for your `mode` is used)
- **model_type**: Choose the model architecture profile (Flux, Qwen, Wan, MiniMax-H3, MOSS, etc.)
- **target_format**: Target format — see [Supported Formats](#supported-formats)
- **device**: Processing device (cuda for GPU, cpu for CPU)

### Optional

- **custom_path**: File, shard folder or HF snapshot dir, used with `mode` = Custom Path (must be inside a ComfyUI model folder)
- **learned_rounding / _iters / _lr / _topk_ratio**: Gradient-descent refinement of FP8 and INT8 ConvRot rounding (much slower)
- **svdquant_rank / svdquant_refine_iters**: `svdquant_w4a4` only
- **gguf_arch / llama_quantize_bin / keep_intermediate_gguf**: GGUF targets only

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
- **Hardware**: ComfyUI with per-module `comfy_quant`
- **Best for**: Better quality than plain INT8

### W8A16 / W8A16 ConvRot (INT8 weights, 16-bit compute)
- **Size**: ~50% of original (identical to INT8 / INT8 ConvRot)
- **Quality**: Closest to BF16 — no activation-quantization error
- **Hardware**: ComfyUI with per-module `comfy_quant`
- **Best for**: When activation quantization hurts; ~2x slower matmul than INT8 ConvRot
- **How**: writes the same INT8 storage as `int8` / `int8_convrot` and adds
  `"full_precision_matrix_mult": true`, so ComfyUI dequantizes the weight and runs a bf16 matmul
  instead of the fused INT8 GEMM

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
  - mode: Diffusion Model
  - diffusion_model: flux-dev-fp16.safetensors
  - model_type: Flux1 / Flux2
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
