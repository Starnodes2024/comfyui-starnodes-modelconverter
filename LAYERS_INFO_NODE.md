# Star Model Layers Info Node

## Overview

New analysis node that inspects diffusion models and generates detailed reports about layer quantization, storage formats, and memory usage.

---

## Features

### 📊 **Comprehensive Analysis**

Analyzes every layer in a diffusion model:
- Layer name and path
- Tensor shape and dimensions
- Storage format (NVFP4, FP8, MXFP8, INT8, INT4, etc.)
- Parameter count
- Memory usage

### 📁 **Automatic Report Generation**

Creates a detailed text file report:
- **Location**: `ComfyUI/output/modelinfo/{model_name}.txt`
- **Format**: Human-readable formatted table
- **Content**: Complete layer-by-layer breakdown

### 🔌 **Dual Outputs**

1. **Status Output**: Summary with key statistics
2. **Layers Info Output**: Full multiline report text

### ⚙️ **Flexible Input**

- **Dropdown**: Select from `models/diffusion_models` folder
- **Custom Path**: Toggle to use any file path

---

## Inputs

### Required
- **model_name**: Dropdown of diffusion models

### Optional
- **use_file_path**: Boolean toggle (default: False)
- **file_path**: Custom path to .safetensors file

---

## Outputs

### 1. Status (STRING)
```
✅ Model analysis complete
Model: flux-dev-nvfp4
Total layers: 1,234
Total parameters: 12,345,678,901
File size: 11.92 GB
Analysis time: 2.3s
Report saved to: E:/ComfyUI/output/modelinfo/flux-dev-nvfp4.txt
```

### 2. Layers Info (STRING - Multiline)
```
Model: flux-dev-nvfp4
File: flux-dev-nvfp4.safetensors
Total size: 11.92 GB
Total parameters: 12,345,678,901
Total layers: 1,234

Layer Type Distribution:
  - nvfp4: 856 layers
  - bf16: 378 layers
  - scale_tensor: 856 layers

============================================================
Layer Details:
============================================================
double_blocks.0.img_attn.norm.key_norm.scale    | Shape: [3072]       | Storage: bf16              | Params:    3,072 | Size: 6.00 KB
double_blocks.0.img_attn.proj.weight            | Shape: [3072, 3072] | Storage: NVFP4 (4-bit)     | Params: 9,437,184 | Size: 4.50 MB
double_blocks.0.img_attn.proj.weight_scale      | Shape: []           | Storage: bf16 (scale)      | Params:        1 | Size: 2 bytes
...
```

---

## Detected Formats

The node automatically identifies:

| Format | Description |
|--------|-------------|
| **NVFP4** | 4-bit NVIDIA floating point |
| **FP8** | 8-bit floating point (e4m3fn) |
| **MXFP8** | OCP Microscaling 8-bit FP |
| **INT8** | 8-bit integer quantization |
| **INT8 ConvRot** | INT8 with Hadamard rotation |
| **INT4 ConvRot** | INT4 with Hadamard rotation |
| **FP16** | Half precision |
| **BF16** | Brain float 16 |
| **FP32** | Full precision |
| **Scaled** | Per-tensor scaled formats |

---

## Use Cases

### 1. Verify Quantization Results
```
Star Ultimate Model Converter
  ↓
Star Model Layers Info
  ↓
Verify which layers were quantized
```

### 2. Compare Models
```
Star Model Layers Info (original)
Star Model Layers Info (quantized)
  ↓
Compare formats and compression
```

### 3. Debug Conversion Issues
```
Star Model Layers Info
  ↓
Identify failed quantization layers
Check for unexpected formats
```

### 4. Generate Documentation
```
Star Model Layers Info
  ↓
Create technical documentation
Share layer details with team
```

---

## Example Workflow

```
┌─────────────────────────────┐
│ Star Model Layers Info      │
│  - model_name: flux-dev     │
│  - use_file_path: False     │
└──────────┬──────────────────┘
           │
           ├─ status ──────────► Display Text
           │
           └─ layers_info ─────► Save Text / Display
                                 
Output File: ComfyUI/output/modelinfo/flux-dev.txt
```

---

## Report Format

### Summary Section
```
Model: {model_name}
File: {filename}
Total size: {size in GB}
Total parameters: {count}
Total layers: {count}

Layer Type Distribution:
  - nvfp4: 856 layers
  - bf16: 378 layers
  - scale_tensor: 856 layers
```

### Layer Details Section
```
============================================================
Layer Details:
============================================================
{layer_name:<80} | Shape: {shape:<20} | Storage: {format:<30} | Params: {count:>12,} | Size: {size}
```

Each line shows:
- Full layer path/name (80 chars)
- Tensor shape (20 chars)
- Storage format with details (30 chars)
- Parameter count (right-aligned, comma-separated)
- Memory size (human-readable: bytes/KB/MB/GB)

---

## Technical Implementation

### Metadata Reading
```python
# Load safetensors metadata
with safetensors.safe_open(path, framework="pt") as f:
    metadata = f.metadata()
    keys = f.keys()

# Parse quantization metadata
quant_data = json.loads(metadata["_quantization_metadata"])
```

### Format Detection
1. Check for `_scale` suffix → Scaled quantization
2. Check quantization metadata → NVFP4, MXFP8, INT8/INT4 ConvRot
3. Check tensor dtype → FP16, BF16, FP32, FP8
4. Check for `.comfy_quant` → Embedded configs

### Statistics Calculation
```python
total_params = sum(tensor.numel() for tensor in sd.values())
layer_stats = Counter(format for layer in layers)
size_bytes = params * element_size
```

---

## Performance

- **Speed**: Very fast (2-5 seconds for large models)
- **Memory**: Minimal (only loads metadata, not full model)
- **Output**: Text file + in-memory string

---

## Files Created

1. **star_model_layers_info.py** (200 lines)
   - Node implementation
   - Analysis logic
   - Report generation

2. **web/docs/StarModelLayersInfo.md**
   - Complete help documentation
   - Usage examples
   - Technical details

3. **__init__.py** (Updated)
   - Added node registration

---

## Console Output

```
🔍 [Star Model Layers Info] Starting analysis...
📦 Loading model: flux-dev-nvfp4.safetensors
🔍 Analyzing layers...
💾 Saving layer info to: E:/ComfyUI/output/modelinfo/flux-dev-nvfp4.txt

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

---

## Benefits

✅ **Transparency**: See exactly how each layer is stored  
✅ **Verification**: Confirm quantization worked correctly  
✅ **Debugging**: Identify problematic layers  
✅ **Documentation**: Generate technical reports  
✅ **Comparison**: Compare different model versions  
✅ **Education**: Learn about model structure and compression  

---

## Integration

Works seamlessly with other Starnodes:
- **Star Ultimate Model Converter**: Verify conversion results
- **Star Ultimate AIO Converter**: Analyze AIO checkpoints
- **Star Ultimate Text-Encoder Converter**: Check text encoder formats
- **Star AIO Splitter**: Analyze extracted components

---

**Status**: ✅ Complete and ready to use  
**Category**: ⭐StarNodes/Model Tools  
**Display Name**: ⭐ Star Model Layers Info
