# ⭐ Star Model Layers Info

Analyze diffusion models and generate detailed reports about layer quantization and storage formats.

## Purpose

This node inspects a diffusion model and creates a comprehensive report showing how each layer is stored, including quantization formats, parameter counts, and memory usage. Perfect for understanding model compression and verifying quantization results.

## Inputs

### Required

- **model_name**: Select a diffusion model from your `models/diffusion_models` folder
  - Dropdown list of all available models
  - Used unless "Use File Path" is enabled

### Optional

- **use_file_path**: Toggle to use a custom file path
  - **Default**: False
  - Enable to load from a custom location
- **file_path**: Custom path to .safetensors file
  - Only used when "Use File Path" is enabled
  - Supports absolute paths
  - Example: `E:/Models/my_model.safetensors`

## Outputs

- **status**: Summary of the analysis (model name, layer count, file size, output location)
- **layers_info**: Complete multiline text report with all layer details

## What It Analyzes

### For Each Layer

1. **Layer Name**: Full key path in the model
2. **Shape**: Tensor dimensions
3. **Storage Format**: How the layer is stored
   - NVFP4 (4-bit)
   - FP8 (e4m3fn)
   - MXFP8 (8-bit microscaling)
   - INT8 / INT8 ConvRot
   - INT4 ConvRot
   - FP16, BF16, FP32
   - Scaled formats
4. **Parameter Count**: Number of parameters in the layer
5. **Size**: Memory usage in bytes/KB/MB/GB

### Report Sections

1. **Summary**:
   - Model name and file path
   - Total size and parameter count
   - Total number of layers
   - Layer type distribution

2. **Layer Details**:
   - Complete list of all layers
   - Formatted table with all information
   - Sorted by layer name

## Output File

### Location
```
ComfyUI/output/modelinfo/{model_name}.txt
```

### Format
```
Model: flux-dev-nvfp4
File: flux-dev-nvfp4.safetensors
Total size: 11.92 GB
Total parameters: 12,345,678,901
Total layers: 1,234

Layer Type Distribution:
  - nvfp4: 856 layers
  - bf16: 378 layers

============================================================
Layer Details:
============================================================
double_blocks.0.img_attn.norm.key_norm.scale    | Shape: [3072]              | Storage: bf16                | Params:        3,072 | Size: 6.00 KB
double_blocks.0.img_attn.norm.query_norm.scale  | Shape: [3072]              | Storage: bf16                | Params:        3,072 | Size: 6.00 KB
double_blocks.0.img_attn.proj.weight            | Shape: [3072, 3072]        | Storage: NVFP4 (4-bit)       | Params:    9,437,184 | Size: 4.50 MB
...
```

## Use Cases

### 1. Verify Quantization
```
Star Ultimate Model Converter
  ↓ (convert model)
Star Model Layers Info
  ↓
Check which layers were quantized vs preserved
```

### 2. Compare Models
```
Star Model Layers Info (original model)
Star Model Layers Info (quantized model)
  ↓
Compare layer formats and sizes
```

### 3. Debug Conversion Issues
```
Star Model Layers Info
  ↓
Identify layers that failed to quantize
Check for unexpected formats
```

### 4. Model Documentation
```
Star Model Layers Info
  ↓
Generate detailed technical documentation
Share layer information with others
```

## Example Output

### Console Status
```
============================================================
✅ Model analysis complete
Model: flux-dev-nvfp4
Total layers: 1,234
Total parameters: 12,345,678,901
File size: 11.92 GB
Analysis time: 2.3s
Report saved to: E:/ComfyUI/output/modelinfo/flux-dev-nvfp4.txt
============================================================
```

### Layers Info Output (excerpt)
```
Model: flux-dev-nvfp4
Total size: 11.92 GB
Total parameters: 12,345,678,901
Total layers: 1,234

Layer Type Distribution:
  - nvfp4: 856 layers
  - bf16: 378 layers

============================================================
Layer Details:
============================================================
[Complete formatted table of all layers]
```

## Detected Formats

The node automatically detects and reports:

- **NVFP4**: 4-bit NVIDIA floating point
- **FP8**: 8-bit floating point (e4m3fn, e5m2)
- **MXFP8**: OCP Microscaling 8-bit FP
- **INT8**: 8-bit integer quantization
- **INT8 ConvRot**: INT8 with Hadamard rotation
- **INT4 ConvRot**: INT4 with Hadamard rotation
- **FP16**: Half precision
- **BF16**: Brain float 16
- **FP32**: Full precision
- **Scaled formats**: Per-tensor scaled quantization
- **Metadata tensors**: Embedded quantization configs

## Tips

- **Large Models**: Analysis is fast even for huge models (seconds)
- **File Path**: Use custom path for models outside ComfyUI folders
- **Report Storage**: All reports saved to `output/modelinfo/` for easy access
- **Text Output**: Connect `layers_info` to a text display node to view in ComfyUI
- **Comparison**: Run on multiple models and compare the .txt files

## Technical Details

### Metadata Parsing

The node reads:
- Safetensors metadata for quantization info
- Embedded `.comfy_quant` JSON configs
- Scale tensors (`_scale` suffixes)
- Tensor dtypes and shapes

### Statistics

Calculates:
- Total parameter count across all layers
- Size distribution by format
- Layer count by type
- Memory usage per layer

## Requirements

- **Input**: Safetensors format model file
- **Output Folder**: Write access to ComfyUI output directory
- **Memory**: Minimal (only loads metadata, not full model)

## Credits

Part of the Starnodes Model Converter suite for comprehensive model analysis and management.
