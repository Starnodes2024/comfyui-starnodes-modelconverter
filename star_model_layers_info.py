"""Star Model Layers Info - Analyze and report layer quantization information."""
import os
import time
import torch
import folder_paths
import safetensors.torch
from collections import Counter, OrderedDict
import json

# Inline utilities (no dependency on star_utils.py)
DTYPE_NAMES = {
    torch.float32: "fp32",
    torch.float16: "fp16",
    torch.bfloat16: "bf16",
    torch.float8_e4m3fn: "fp8_e4m3fn",
    torch.float8_e5m2: "fp8_e5m2",
    torch.int8: "int8",
}

def format_size(num_bytes):
    """Format bytes as human-readable size."""
    if num_bytes < 1024:
        return f"{num_bytes} bytes"
    elif num_bytes < 1024**2:
        return f"{num_bytes / 1024:.2f} KB"
    elif num_bytes < 1024**3:
        return f"{num_bytes / (1024**2):.2f} MB"
    else:
        return f"{num_bytes / (1024**3):.2f} GB"


class StarModelLayersInfo:
    """Analyze diffusion model layers and report quantization information."""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("diffusion_models"), {
                    "tooltip": "Select a diffusion model from your ComfyUI diffusion_models folder."
                }),
            },
            "optional": {
                "use_file_path": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable this to use a custom file path instead of selecting from the model list."
                }),
                "file_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Enter path to .safetensors file",
                    "tooltip": "Full path to a .safetensors file. Only used when 'Use File Path' is enabled."
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("status", "layers_info")
    FUNCTION = "analyze"
    CATEGORY = '⭐StarNodes/Model Tools'
    OUTPUT_NODE = True

    def analyze(self, model_name, use_file_path=False, file_path=""):
        start_time = time.time()
        
        print("🔍 [Star Model Layers Info] Starting analysis...")
        
        # Determine input path
        if use_file_path and file_path.strip():
            input_path = os.path.abspath(os.path.expanduser(file_path.strip().strip('"')))
            if not os.path.isfile(input_path):
                raise ValueError(f"File not found: {input_path}")
        else:
            input_path = folder_paths.get_full_path("diffusion_models", model_name)
        
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        input_bytes = os.path.getsize(input_path)
        
        print(f"📦 Loading model: {os.path.basename(input_path)}")
        
        # Load model with safetensors to access metadata
        with safetensors.safe_open(input_path, framework="pt") as f:
            metadata = f.metadata()
            keys = f.keys()
        
        # Load actual tensors
        sd = safetensors.torch.load_file(input_path)
        
        # Parse quantization metadata if present
        quant_metadata = {}
        if metadata and "_quantization_metadata" in metadata:
            try:
                quant_data = json.loads(metadata["_quantization_metadata"])
                quant_metadata = quant_data.get("layers", {})
            except:
                pass
        
        # Analyze layers
        layer_info = []
        layer_stats = Counter()
        total_params = 0
        
        print("🔍 Analyzing layers...")
        
        for key in sorted(keys):
            tensor = sd[key]
            
            # Get basic info
            dtype_name = DTYPE_NAMES.get(tensor.dtype, str(tensor.dtype))
            shape = list(tensor.shape)
            num_params = tensor.numel()
            total_params += num_params
            size_bytes = num_params * tensor.element_size()
            
            # Determine quantization info
            quant_info = "None"
            storage_format = dtype_name
            
            # Check if this is a quantized weight
            if key.endswith(".weight"):
                base_key = key[:-len(".weight")]
                
                # Check for scale tensors (FP8, INT8)
                if f"{key}_scale" in keys:
                    quant_info = "Per-tensor scaled"
                    storage_format = f"{dtype_name} + scale"
                    layer_stats["scaled"] += 1
                
                # Check for quantization metadata
                if base_key in quant_metadata:
                    meta = quant_metadata[base_key]
                    fmt = meta.get("format", "unknown")
                    quant_info = fmt
                    
                    if fmt == "nvfp4":
                        storage_format = "NVFP4 (4-bit)"
                        layer_stats["nvfp4"] += 1
                    elif fmt == "mxfp8":
                        storage_format = "MXFP8 (8-bit microscaling)"
                        layer_stats["mxfp8"] += 1
                    elif fmt == "int8_tensorwise":
                        if meta.get("convrot"):
                            storage_format = "INT8 ConvRot"
                            layer_stats["int8_convrot"] += 1
                        else:
                            storage_format = "INT8"
                            layer_stats["int8"] += 1
                    elif fmt == "convrot_w4a4":
                        storage_format = "INT4 ConvRot"
                        layer_stats["int4_convrot"] += 1
                    elif fmt == "float8_e4m3fn":
                        storage_format = "FP8 (e4m3fn)"
                        layer_stats["fp8"] += 1
                    else:
                        storage_format = fmt
                        layer_stats[fmt] += 1
                else:
                    layer_stats[dtype_name] += 1
            elif key.endswith("_scale"):
                # Scale tensor
                quant_info = "Scale tensor"
                storage_format = f"{dtype_name} (scale)"
                layer_stats["scale_tensor"] += 1
            elif key.endswith(".comfy_quant"):
                # Embedded quant config
                quant_info = "Quant config"
                storage_format = "JSON metadata"
                layer_stats["metadata"] += 1
            else:
                # Other tensors (biases, norms, etc.)
                layer_stats[dtype_name] += 1
            
            # Format layer info
            info_line = f"{key:<80} | Shape: {str(shape):<20} | Storage: {storage_format:<30} | Params: {num_params:>12,} | Size: {format_size(size_bytes)}"
            layer_info.append(info_line)
        
        # Build summary
        duration = time.time() - start_time
        
        summary_lines = [
            f"Model: {base_name}",
            f"File: {os.path.basename(input_path)}",
            f"Total size: {format_size(input_bytes)}",
            f"Total parameters: {total_params:,}",
            f"Total layers: {len(keys)}",
            "",
            "Layer Type Distribution:",
        ]
        
        for layer_type, count in layer_stats.most_common():
            summary_lines.append(f"  - {layer_type}: {count} layers")
        
        summary_lines.extend([
            "",
            "="*120,
            "Layer Details:",
            "="*120,
        ])
        
        # Combine summary and layer info
        full_info = "\n".join(summary_lines + layer_info)
        
        # Save to file
        output_dir = os.path.join(folder_paths.get_output_directory(), "modelinfo")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{base_name}.txt")
        
        print(f"💾 Saving layer info to: {output_file}")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(full_info)
        
        # Build status message
        status = "\n".join([
            f"✅ Model analysis complete",
            f"Model: {base_name}",
            f"Total layers: {len(keys)}",
            f"Total parameters: {total_params:,}",
            f"File size: {format_size(input_bytes)}",
            f"Analysis time: {duration:.1f}s",
            f"Report saved to: {output_file}",
        ])
        
        # Print status to console
        print("\n" + "="*60)
        print(status)
        print("="*60 + "\n")
        
        return (status, full_info)


NODE_CLASS_MAPPINGS = {"StarModelLayersInfo": StarModelLayersInfo}
NODE_DISPLAY_NAME_MAPPINGS = {"StarModelLayersInfo": "⭐ Star Model Layers Info"}
