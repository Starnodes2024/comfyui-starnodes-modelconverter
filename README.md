# ⭐ Starnodes Model Converter v1.2.1

ComfyUI custom nodes for converting, quantizing and splitting diffusion models.
Big thangs to ComfyUI Org for https://github.com/Comfy-Org/comfy-kitchen which made this possible!

<img width="755" height="410" alt="image" src="https://github.com/user-attachments/assets/ab3d23ec-ab33-4427-a29e-4ed7c132876e" />


**Included nodes:**
- **⭐ Star Model Layers Info**
- **⭐ Star Ultimate Model Converter**: Convert and quantize diffusion models to various precision formats with intelligent layer-specific optimization
- **⭐ Starnodes AIO Splitter**: Split all-in-one checkpoints into separate diffusion model, text encoder and VAE files

<img width="755" height="309" alt="image" src="https://github.com/user-attachments/assets/a4a32045-0b9f-4807-8b95-1550009b43b8" />


## Features

- **Multiple Format Support**: Convert models to NVFP4, FP8, MXFP8, INT8, INT8 ConvRot, INT4 ConvRot, FP16, or FP32
- **Smart Layer Preservation**: Architecture-specific profiles ensure critical layers stay in high precision
- **Automatic Dequantization**: Intelligently handles pre-quantized models for clean re-quantization
- **Multi-Shard Support**: Seamlessly processes models split across multiple .safetensors files
- **Metadata Preservation**: Maintains model metadata and quantization information
- **Progress Tracking**: Real-time conversion progress with detailed statistics
- **Custom Path Support**: Load models from anywhere on your system, not just ComfyUI folders

## Supported Models

The converter includes optimized profiles for:

- NEW All types of text encoders!
- **Flux Family**: Flux.1-dev, Flux.1-Fill, Flux.2-dev, Flux.2-Klein-9b
- **Z-Image**: Z-Image-Turbo, Z-Image-Base
- **Qwen**: Qwen-Image-Edit-2511, Qwen-Image-2512
- **LTX**: LTX-2-19b-dev-or-distilled, LTXV_EROX
- **Krea**: Krea-2-Turbo
- **Wan**: Wan2.2-i2v-high-low
- **ACE**: ACE-Step-1.5-XL-Turbo
- **ERNIE**: ERNIE-Image, ERNIE-Image-Turbo
- **Lens**: Lens models

Each profile has carefully tuned blacklists to preserve model quality while maximizing compression.

## Installation

1. Clone or download this repository into your ComfyUI custom_nodes folder:
   ```bash
   cd ComfyUI/custom_nodes
   git clone <repository-url> comfyui-starnodes-modelconverter
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Restart ComfyUI

## Requirements

- **ComfyUI**: Latest version recommended (v0.27.0+ for INT8/INT4 ConvRot)
- **comfy-kitchen**: Required for NVFP4, FP8, MXFP8, INT8, and INT4 quantization
- **PyTorch**: 2.0+ with CUDA support (for GPU acceleration)
- **safetensors**: For model loading and saving
- **NVIDIA GPU**: Required for NVFP4 format, recommended for FP8/MXFP8/INT8/INT4

### System Requirements

- **RAM**: **64GB+ recommended** for large models (Flux, LTX, etc.)
- **Pagefile**: If you encounter memory issues, set a **100GB+ pagefile** (Windows) or swap space (Linux)
- **Device Selection**: Use `cpu` device for conversion if GPU memory is insufficient - it's slower but works with large models

**Memory Tips:**
- Large models (20GB+) may require significant system RAM during conversion
- If conversion fails with out-of-memory errors, try using `device: cpu` instead of `cuda`
- Windows users: Set a large pagefile via System Properties → Advanced → Performance Settings → Advanced → Virtual Memory

## Usage

### Basic Workflow

1. Add the "⭐ Star Ultimate Model Converter" node to your workflow
2. Select your model from the dropdown or enable custom path
3. Choose the appropriate model type (architecture profile)
4. Select target format (nvfp4, fp8, int8, fp16, or fp32)
5. Choose device (cuda for GPU, cpu for CPU)
6. Execute the node

### Widget Guide

#### **model_name**
Select a model from your ComfyUI `diffusion_models` folder. This will be used unless "Use Custom Path" or "Use Model from AIO Checkpoint" is enabled.

#### **model_type**
Choose the model architecture profile. This determines which layers to keep in high precision and which can be quantized safely. Select the profile that matches your model architecture for best results.

#### **target_format**
- **nvfp4**: Smallest size (~25% of original), NVIDIA GPUs only, excellent quality
- **fp8**: Small size (~50% of original), good quality, NVIDIA GPUs recommended
- **mxfp8**: OCP Microscaling 8-bit standard that uses hardware-efficient microscaling (block scaling) to achieve excellent visual quality, near FP16
- **int8**: Compatible with most hardware, moderate compression
- **int8_convrot**: INT8 with block-Hadamard weight rotation (ConvRot), better quality than plain INT8, requires ComfyUI v0.27.0+
- **int4_convrot**: INT4 with block-Hadamard weight rotation (ConvRot), smallest size (~25% of original), excellent quality-to-size ratio, requires SM 8.0+ (Ampere/Ada/Blackwell)
- **fp16**: Standard half precision, widely compatible
- **fp32**: Full precision, no compression

#### **device**
- **cuda**: Much faster, requires NVIDIA GPU
- **cpu**: Works on all systems but significantly slower

#### **use_custom_path** (Optional)
Enable this toggle to use a custom file path instead of selecting from the model list. When enabled, the converter will use the path specified in the "custom_path" field.

#### **custom_path** (Optional)
Full path to a .safetensors file or a folder containing model shards. Only used when "Use Custom Path" is enabled. Supports:
- Single .safetensors files: `/path/to/model.safetensors`
- Folders with shards: `/path/to/model_folder/` (will process all .safetensors files)
- HuggingFace cache folders: `~/.cache/huggingface/hub/models--user--model/snapshots/abc123/`

#### **use_aio_checkpoint** (Optional)
Enable this to convert directly from an all-in-one checkpoint. **ONLY the diffusion model (UNet) is extracted and converted** - the text encoder and VAE are NOT included in the output. The converted model is saved to `models/diffusion_models`.

#### **checkpoint_name** (Optional)
Select an all-in-one checkpoint from your ComfyUI `checkpoints` folder. Only used when "Use Model from AIO Checkpoint" is enabled. To also extract the text encoder and VAE from an AIO checkpoint, use the Starnodes AIO Splitter node.

### Output

The converted model will be saved in the same directory as the source model with the format appended to the filename:
- Original: `flux-dev-fp16.safetensors`
- Converted: `flux-dev-nvfp4.safetensors`

The node outputs a detailed status message including:
- Input and output file information
- Original and new file sizes
- Compression ratio
- Layer conversion statistics
- Processing time
- Output path

## ⭐ Starnodes AIO Splitter

Splits an all-in-one checkpoint (model + text encoder + VAE in one file) into separate files, saved directly to the correct ComfyUI model folders.

### Widget Guide

#### **checkpoint_name**
Select an all-in-one checkpoint from your ComfyUI `checkpoints` folder to split into its components.

#### **save_model**
Extract the diffusion model and save it to `models/diffusion_models` with the `_model` suffix. Enabled by default.

#### **save_text_encoder**
Extract the text encoder (CLIP) and save it to `models/text_encoders` with the `_clip` suffix. Disabled by default.

#### **save_vae**
Extract the VAE and save it to `models/vae` with the `_vae` suffix. Disabled by default.

### Output

Files are saved as `.safetensors` with the original filename plus a component suffix:
- `my-checkpoint.safetensors` → `models/diffusion_models/my-checkpoint_model.safetensors`
- `my-checkpoint.safetensors` → `models/text_encoders/my-checkpoint_clip.safetensors`
- `my-checkpoint.safetensors` → `models/vae/my-checkpoint_vae.safetensors`

The node outputs a status string listing each saved component with its tensor count, file size and output path, plus the processing time. Components with no matching keys in the checkpoint are skipped with a warning.

## Format Comparison

| Format | Size | Quality | Compatibility | Speed |
|--------|------|---------|---------------|-------|
| NVFP4  | ★★★★★ | ★★★★☆ | NVIDIA only | ★★★★★ |
| FP8    | ★★★★☆ | ★★★★☆ | NVIDIA recommended | ★★★★☆ |
| MXFP8  | ★★★★☆ | ★★★★★ | Growing (OCP Standard) | ★★★★★ |
| INT8   | ★★★☆☆ | ★★★☆☆ | Most hardware | ★★★☆☆ |
| INT8 ConvRot | ★★★☆☆ | ★★★★☆ | ComfyUI v0.27.0+ | ★★★☆☆ |
| INT4 ConvRot | ★★★★★ | ★★★★☆ | SM 8.0+ (Ampere+) | ★★★★☆ |
| FP16   | ★★☆☆☆ | ★★★★★ | All hardware | ★★★★☆ |
| FP32   | ★☆☆☆☆ | ★★★★★ | All hardware | ★★☆☆☆ |

## Advanced Features

### Automatic Dequantization

The converter automatically detects and dequantizes pre-quantized models:
- ComfyUI scaled FP8 checkpoints
- Per-tensor FP8/INT8 with weight scales
- Quantization metadata from previous conversions

This ensures clean re-quantization without quality degradation from multiple quantization passes.

### Layer Blacklisting

Each model profile includes a blacklist of layers that should remain in high precision:
- Embedding layers
- Normalization layers
- Final output layers
- Architecture-specific critical components

This preserves model quality while maximizing compression on less sensitive layers.

### Metadata Preservation

The converter maintains:
- Original model format information
- Model architecture specifications
- Quantization metadata for proper loading
- Extended metadata (for models like LTX that require it)

## Troubleshooting

### "comfy-kitchen not found"
Install comfy-kitchen for quantization support:
```bash
pip install comfy-kitchen
```

### "CUDA out of memory"
- Try using `device: cpu` (slower but uses system RAM)
- Close other applications
- Use a smaller target format (fp16 instead of fp32)

### "No .safetensors files found"
- Verify the custom path is correct
- Ensure the folder contains .safetensors files
- Check file permissions

### Model quality issues
- Verify you selected the correct model_type profile
- Try a less aggressive format (fp8 instead of nvfp4)
- Check if the source model is already quantized

## Performance Tips

1. **Use CUDA**: GPU conversion is 10-50x faster than CPU
2. **Batch conversions**: Convert multiple models in sequence
3. **Storage**: Ensure sufficient disk space (converted models are saved alongside originals)
4. **Memory**: NVFP4 and FP8 require less VRAM during conversion than INT8

## Technical Details

### Quantization Methods

- **NVFP4**: 4-bit floating point using NVIDIA Tensor Cores
- **FP8**: 8-bit floating point (e4m3fn format)
- **MXFP8**: OCP Microscaling 8-bit floating point (e4m3 data with power-of-2 E8M0 block scales, block size 32)
- **INT8**: 8-bit integer with per-tensor scaling
- **INT8 ConvRot**: 8-bit integer with per-channel scaling and group-wise Hadamard rotation (group size 256)
- **INT4 ConvRot**: 4-bit integer with per-group scaling and group-wise Hadamard rotation (rotation group size 256, quantization group size 64). Weights are packed as signed int8 (2 nibbles per byte) with fp32 per-group scales.
- **FP16/FP32**: Standard IEEE floating point

### File Naming

The converter automatically strips existing precision suffixes and adds the new format:
- Input: `model-fp16-v2.safetensors`
- Output: `model-v2-nvfp4.safetensors`

## Contributing

Contributions are welcome! To add support for a new model architecture:

1. Test the model with the default profile
2. Identify layers that need high precision (check for quality issues)
3. Add a new profile to `models.json` with appropriate blacklist
4. Submit a pull request with test results

## License

This project is provided as-is for use with ComfyUI. Please respect the licenses of the models you convert.

## Credits

Developed for the ComfyUI community with support for modern quantization techniques and model architectures.

## Changelog

### v1.1.0 (2025-01-10)
- ✨ **New Format**: Added INT4 ConvRot support - 4-bit quantization with Hadamard rotation for ~25% model size
- 📝 **System Requirements**: Added RAM and pagefile recommendations (64GB+ RAM, 100GB+ pagefile for large models)
- 🔧 **Hardware**: INT4 ConvRot requires SM 8.0+ (Ampere/Ada/Blackwell GPUs)
- 📚 **Documentation**: Updated format comparison table and technical details

### v1.0.0
- Initial release with NVFP4, FP8, MXFP8, INT8, INT8 ConvRot support
- Smart layer preservation with architecture-specific profiles
- AIO checkpoint splitter node
- Architecture-specific profiles for 15+ model families
- Automatic dequantization and metadata preservation
