# ⭐ Starnodes Model Converter v1.3.0

ComfyUI custom nodes for converting, quantizing, analyzing, and managing diffusion models with advanced profile-based quantization.

**Included nodes:**

- **⭐ Star Ultimate Model Converter**: Convert and quantize diffusion models to various precision formats with intelligent layer-specific optimization
- **⭐ Star AIO Splitter**: Split all-in-one checkpoints into separate diffusion model, text encoder and VAE files
- **⭐ Star Model Layers Info**: Analyze model layers and generate detailed quantization reports with optional profile saving
- **⭐ Star Ultimate Model Converter Pro**: Profile-based mixed-precision quantization with strict conversion rules

<img width="854" height="296" alt="image" src="https://github.com/user-attachments/assets/a290dc38-7f4f-4977-a34d-ec8859531dc7" />



## Features

### Core Conversion Features
- **Multiple Format Support**: Convert models to NVFP4, FP8, MXFP8, INT8, INT8 ConvRot, INT4 ConvRot, FP16, or FP32
- **Smart Layer Preservation**: Architecture-specific profiles ensure critical layers stay in high precision
- **Automatic Dequantization**: Intelligently handles pre-quantized models for clean re-quantization
- **Multi-Shard Support**: Seamlessly processes models split across multiple .safetensors files
- **Metadata Preservation**: Maintains model metadata and quantization information
- **Progress Tracking**: Real-time conversion progress with detailed statistics
- **Custom Path Support**: Load models from anywhere on your system, not just ComfyUI folders

### New in v1.3.0
- **Profile-Based Quantization**: Extract quantization blueprints from optimized models and apply to new models
- **Layer Analysis**: Detailed layer-by-layer format inspection with Normal and Tree views
- **Profile Management**: Save, load, and apply quantization profiles with metadata tracking
- **Interactive Tooltips**: Hover over profiles to see model information and creation details
- **Strict Conversion Rules**: 5-rule system ensures safe, predictable quantization
- **AIO Management**: Complete workflow for splitting and merging all-in-one checkpoints

## Supported Models

The converter includes optimized profiles for:

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

## ⭐ Star Model Layers Info

Analyzes diffusion models and generates detailed reports about layer quantization, storage formats, and memory usage. Supports two view modes and optional profile saving for use with the Converter Pro.

### Widget Guide

#### **model_name**
Select a diffusion model from your `models/diffusion_models` folder to analyze.

#### **view_mode** (Optional)
- **Normal View**: Flat list of all layers with full details
- **Tree View**: Hierarchical grouped view with layer ranges (e.g., `[0-27]`) and sub-components

#### **use_file_path** (Optional)
Enable to analyze a model from a custom file path instead of the model list.

#### **file_path** (Optional)
Full path to a `.safetensors` file. Only used when "Use File Path" is enabled.

#### **save_profile** (Optional)
When enabled, saves a quantization profile as JSON in the `profiles/` folder. This profile can be used with Star Ultimate Model Converter Pro to apply the same quantization strategy to other models.

### Output

- **status**: Summary with model name, layer count, file size, and report location
- **layers_info**: Complete multiline text report with all layer details

### Files Created

- **Report**: `output/modelinfo/{model_name}_{view_mode}.txt`
- **Profile** (if enabled): `profiles/{model_name}.json`

The profile includes metadata (model name, timestamp, layer count) and a complete layer-to-format mapping for profile-based conversion.

## ⭐ Star Ultimate Model Converter Pro

Advanced profile-based converter with models.json blacklist integration, manual mode control, and actual deep quantization using comfy-kitchen.

### New in v1.3.0

- **🚫 models.json Blacklist**: Automatically preserves critical layers (embeddings, norms, modulations) based on model architecture
- **🎮 Manual Mode**: Fine-grained control over which source block types get quantized
- **💎 Actual Deep Quantization**: Real NVFP4, INT4_CONVROT, MXFP8 quantization using comfy-kitchen (not just FP8 fallback)
- **📊 Priority System**: Blacklist → Manual Mode → Profile → Fallback

### Widget Guide

#### **model_name** (Required)
Select a diffusion model from your `models/diffusion_models` folder to quantize.

#### **profile** (Required)
Select a quantization profile from the dropdown. Profiles are created by Star Model Layers Info with "Save Profile" enabled.

**Hover Tooltip**: Hover over a profile to see metadata including the original model name, creation date, and layer count.

#### **model_type** (Required)
Select the model architecture to load the appropriate blacklist from models.json:
- Ideogram-4, Flux1/Flux2, SDXL, Qwen, LTX, Krea, Wan, ERNIE, Lens, etc.
- Blacklisted layers are automatically preserved in BF16

#### **use_blacklist** (Required)
Enable/disable blacklist protection (default: yes). When enabled, layers matching blacklist patterns are preserved in BF16.

#### **target_quant_format** (Required)
Deep quantization format for heavily compressed layers:
- **NVFP4**: 4-bit NVIDIA floating point
- **INT4_CONVROT**: 4-bit INT with Hadamard rotation
- **FP8**: 8-bit floating point
- **INT8**: 8-bit integer
- **INT8_CONVROT**: 8-bit INT with rotation
- **MXFP8**: Microscaling FP8

#### **manual_mode** (Required)
Enable manual control over quantization (default: disabled). When enabled, you can control which source block types get quantized.

#### **quantize_fp32/bf16/fp16/fp8_e4m3fn/fp8** (Required)
[Manual Mode Only] Control whether to quantize each source block type:
- **YES**: Convert layers of this type to target quantization format (NVFP4, INT4_CONVROT, etc.)
- **NO**: Keep layers of this type in their original format

**How it works**: Manual mode uses the profile as a matrix:
- Profile says "BF16" + You set `quantize_bf16 = YES` → Converts to target format (e.g., INT4_CONVROT)
- Profile says "BF16" + You set `quantize_bf16 = NO` → Keeps as BF16

#### **device** (Required)
- **cuda**: GPU acceleration (much faster)
- **cpu**: CPU processing (slower but works with any hardware)

#### **output_name** (Optional)
Custom name for the output model. Leave blank for automatic naming.

### Processing Priority

1. **Blacklist** (highest): Layers matching blacklist patterns → BF16
2. **Manual Mode**: Override profile decisions for specific dtypes
3. **Profile Format**: Use the format specified in the profile
4. **Fallback**: Safe defaults (FP16 for norms, FP8 for others)

### Output

- **status**: Conversion summary with format breakdown and statistics
- **model**: Converted MODEL object ready for immediate use

The converted model is saved to `models/diffusion_models/` with proper quantization metadata.

### Workflow Example

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
Output: ideogram4_int4_convrot_pro.safetensors (2.45 GB)

Conversion breakdown:
  - blacklisted_to_bf16: 499 layers
  - int4_convrot: 170 layers (actual quantization!)
  - kept_non_float: 211 layers
```

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

**Special thanks to:**
- **[Comfy-Org Team](https://github.com/Comfy-Org)** for developing [comfy-kitchen](https://github.com/Comfy-Org/comfy-kitchen), the quantization library that powers NVFP4, FP8, MXFP8, INT8, and INT4 ConvRot formats
- The ComfyUI community for testing and feedback

## Changelog

### v1.3.0 (2026-07-14)
- ✨ **Pro Converter Enhanced**: models.json blacklist integration with 15+ model profiles
- 🎮 **Manual Mode**: Fine-grained control over which source block types get quantized
- 💎 **Actual Deep Quantization**: Real NVFP4, INT4_CONVROT, MXFP8 quantization using comfy-kitchen
- 📊 **Priority System**: Blacklist → Manual Mode → Profile → Fallback
- 🔧 **Format Normalization**: Handles complex profile formats like "FP8_E4M3FN + SCALE"
- 💾 **Proper Metadata**: Saves quantization metadata for ComfyUI compatibility
- 🐛 **Fixed**: File saving and deep quantization now work correctly
- 📚 **Documentation**: Updated with new features and examples

### v1.2.0 (2025-01-13)
- ✨ **New Node**: Star Model Layers Info - Analyze models with Normal/Tree views and profile saving
- ✨ **New Node**: Star Ultimate Model Converter Pro - Profile-based mixed-precision quantization
- 🔄 **Profile System**: Extract quantization blueprints and apply to new models
- 📊 **Layer Analysis**: Detailed layer-by-layer format inspection with hierarchical tree view
- 🎯 **Strict Rules**: 5-rule conversion system for safe, predictable quantization
- 💡 **Interactive UI**: Hover tooltips show profile metadata
- 📋 **Profile Management**: JSON-based profiles with metadata tracking
- 🔌 **API Endpoint**: RESTful API for profile metadata
- 📚 **Documentation**: Comprehensive guides for profile system
- 🙏 **Credits**: Added attribution to Comfy-Org team for comfy-kitchen library

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
