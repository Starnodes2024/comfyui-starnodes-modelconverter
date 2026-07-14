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
                "view_mode": (["Normal View", "Tree View"], {
                    "default": "Normal View",
                    "tooltip": "Normal View: Flat list of all layers. Tree View: Hierarchical grouped view with layer ranges."
                }),
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
                "save_profile": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Save layer-by-layer quantization profile as JSON for use with Star Ultimate Model Converter Pro."
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("status", "layers_info")
    FUNCTION = "analyze"
    CATEGORY = '⭐StarNodes/Model Tools'
    OUTPUT_NODE = True

    def analyze(self, model_name, view_mode="Normal View", use_file_path=False, file_path="", save_profile=False):
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
        
        # Analyze layers and collect data
        layer_data = []
        layer_stats = Counter()
        total_params = 0
        
        print(f"🔍 Analyzing layers in {view_mode}...")
        
        for key in sorted(keys):
            tensor = sd[key]
            
            # Get basic info
            dtype_name = DTYPE_NAMES.get(tensor.dtype, str(tensor.dtype))
            shape = list(tensor.shape)
            num_params = tensor.numel()
            total_params += num_params
            size_bytes = num_params * tensor.element_size()
            
            # Determine quantization info
            storage_format = dtype_name
            
            # Check if this is a quantized weight
            if key.endswith(".weight"):
                base_key = key[:-len(".weight")]
                
                # Check for scale tensors (FP8, INT8)
                if f"{key}_scale" in keys:
                    storage_format = f"{dtype_name} + scale"
                    layer_stats["scaled"] += 1
                
                # Check for quantization metadata
                if base_key in quant_metadata:
                    meta = quant_metadata[base_key]
                    fmt = meta.get("format", "unknown")
                    
                    if fmt == "nvfp4":
                        storage_format = "NVFP4"
                        layer_stats["nvfp4"] += 1
                    elif fmt == "mxfp8":
                        storage_format = "MXFP8"
                        layer_stats["mxfp8"] += 1
                    elif fmt == "int8_tensorwise":
                        if meta.get("convrot"):
                            storage_format = "INT8_CONVROT"
                            layer_stats["int8_convrot"] += 1
                        else:
                            storage_format = "INT8"
                            layer_stats["int8"] += 1
                    elif fmt == "convrot_w4a4":
                        storage_format = "INT4_CONVROT"
                        layer_stats["int4_convrot"] += 1
                    elif fmt == "float8_e4m3fn":
                        storage_format = "F8_E4M3"
                        layer_stats["fp8"] += 1
                    else:
                        storage_format = fmt.upper()
                        layer_stats[fmt] += 1
                else:
                    layer_stats[dtype_name] += 1
            elif key.endswith("_scale"):
                storage_format = f"{dtype_name}_SCALE"
                layer_stats["scale_tensor"] += 1
            elif key.endswith(".comfy_quant"):
                storage_format = "METADATA"
                layer_stats["metadata"] += 1
            else:
                layer_stats[dtype_name] += 1
            
            # Store layer data
            layer_data.append({
                "key": key,
                "shape": shape,
                "format": storage_format.upper(),
                "params": num_params,
                "size": size_bytes
            })
        
        # Generate output based on view mode
        if view_mode == "Tree View":
            layer_info = self._build_tree_view(layer_data)
        else:
            layer_info = self._build_normal_view(layer_data)
        
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
        
        # Save to file with view mode suffix
        output_dir = os.path.join(folder_paths.get_output_directory(), "modelinfo")
        os.makedirs(output_dir, exist_ok=True)
        view_suffix = "_tree" if view_mode == "Tree View" else "_normal"
        output_file = os.path.join(output_dir, f"{base_name}{view_suffix}.txt")
        
        print(f"💾 Saving layer info to: {output_file}")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(full_info)
        
        # Save profile if requested
        profile_file = None
        if save_profile:
            profile_file = self._save_quantization_profile(layer_data, base_name, input_path)
        
        # Build status message
        status_lines = [
            f"✅ Model analysis complete ({view_mode})",
            f"Model: {base_name}",
            f"Total layers: {len(keys)}",
            f"Total parameters: {total_params:,}",
            f"File size: {format_size(input_bytes)}",
            f"Analysis time: {duration:.1f}s",
            f"Report saved to: {output_file}",
        ]
        
        if profile_file:
            status_lines.append(f"📋 Profile saved to: {profile_file}")
        
        status = "\n".join(status_lines)
        
        # Print status to console
        print("\n" + "="*60)
        print(status)
        print("="*60 + "\n")
        
        return (status, full_info)
    
    def _build_normal_view(self, layer_data):
        """Build flat list view of all layers."""
        lines = []
        for layer in layer_data:
            line = f"{layer['key']:<80} | Shape: {str(layer['shape']):<20} | Format: {layer['format']:<20} | Params: {layer['params']:>12,} | Size: {format_size(layer['size'])}"
            lines.append(line)
        return lines
    
    def _build_tree_view(self, layer_data):
        """Build hierarchical tree view with layer grouping."""
        from collections import defaultdict
        import re
        
        # Build tree structure
        tree = defaultdict(list)
        for layer in layer_data:
            parts = layer['key'].split('.')
            if len(parts) > 1:
                # Group by first level (e.g., "blocks", "double_blocks")
                prefix = parts[0]
                tree[prefix].append(layer)
            else:
                tree['_root'].append(layer)
        
        lines = []
        
        for prefix in sorted(tree.keys()):
            layers = tree[prefix]
            if not layers:
                continue
            
            # Group consecutive numbered layers
            grouped = self._group_consecutive_layers(layers)
            
            # Calculate total size and formats for this prefix
            total_size = sum(l['size'] for l in layers)
            formats = sorted(set(l['format'] for l in layers))
            formats_str = ", ".join(formats)
            
            # Print prefix header
            if prefix == '_root':
                lines.append(f"├── Root ({format_size(total_size)} | {formats_str})")
            else:
                lines.append(f"├── {prefix} ({format_size(total_size)} | {formats_str})")
            
            # Print grouped layers
            for group in grouped:
                if group['is_range']:
                    # Range of layers
                    total_group_size = sum(l['size'] for l in group['layers'])
                    group_formats = sorted(set(l['format'] for l in group['layers']))
                    group_formats_str = ", ".join(group_formats)
                    lines.append(f"│   ├── [{group['start']}-{group['end']}] ({format_size(total_group_size)} | {group_formats_str})")
                    
                    # Show sub-components of the range
                    subcomponents = defaultdict(list)
                    for layer in group['layers']:
                        # Extract component name (e.g., "attn", "mlp", "norm")
                        parts = layer['key'].split('.')
                        if len(parts) > 2:
                            component = '.'.join(parts[2:])  # Everything after the number
                            subcomponents[component].append(layer)
                    
                    for comp_name in sorted(subcomponents.keys()):
                        comp_layers = subcomponents[comp_name]
                        comp_size = sum(l['size'] for l in comp_layers)
                        comp_formats = sorted(set(l['format'] for l in comp_layers))
                        comp_formats_str = ", ".join(comp_formats)
                        lines.append(f"│   │   ├── {comp_name} ({format_size(comp_size)} | {comp_formats_str})")
                else:
                    # Single layer
                    layer = group['layers'][0]
                    lines.append(f"│   ├── {layer['key']} ({format_size(layer['size'])} | {layer['format']})")
        
        return lines
    
    def _group_consecutive_layers(self, layers):
        """Group consecutive numbered layers together."""
        import re
        
        # Extract layer numbers
        numbered = []
        unnumbered = []
        
        for layer in layers:
            match = re.search(r'\.(\d+)\.', layer['key'])
            if match:
                num = int(match.group(1))
                numbered.append((num, layer))
            else:
                unnumbered.append(layer)
        
        # Sort by number
        numbered.sort(key=lambda x: x[0])
        
        # Group consecutive numbers
        groups = []
        if numbered:
            current_group = [numbered[0]]
            
            for i in range(1, len(numbered)):
                if numbered[i][0] == current_group[-1][0] + 1:
                    current_group.append(numbered[i])
                else:
                    # Finish current group
                    if len(current_group) >= 3:  # Only group if 3+ consecutive
                        groups.append({
                            'is_range': True,
                            'start': current_group[0][0],
                            'end': current_group[-1][0],
                            'layers': [l for _, l in current_group]
                        })
                    else:
                        for _, layer in current_group:
                            groups.append({
                                'is_range': False,
                                'layers': [layer]
                            })
                    current_group = [numbered[i]]
            
            # Add last group
            if len(current_group) >= 3:
                groups.append({
                    'is_range': True,
                    'start': current_group[0][0],
                    'end': current_group[-1][0],
                    'layers': [l for _, l in current_group]
                })
            else:
                for _, layer in current_group:
                    groups.append({
                        'is_range': False,
                        'layers': [layer]
                    })
        
        # Add unnumbered layers
        for layer in unnumbered:
            groups.append({
                'is_range': False,
                'layers': [layer]
            })
        
        return groups
    
    def _save_quantization_profile(self, layer_data, model_name, model_path):
        """Save quantization profile as JSON for use with Model Converter Pro."""
        from datetime import datetime
        
        # Create profiles directory
        profiles_dir = os.path.join(os.path.dirname(__file__), "profiles")
        os.makedirs(profiles_dir, exist_ok=True)
        
        # Build profile structure
        profile = {
            "__metadata__": {
                "original_model_name": model_name,
                "original_model_path": model_path,
                "timestamp": datetime.now().isoformat(),
                "total_layers": len(layer_data),
                "created_by": "Star Model Layers Info"
            },
            "layers": {}
        }
        
        # Map each layer to its format
        for layer in layer_data:
            profile["layers"][layer["key"]] = layer["format"]
        
        # Save profile
        profile_file = os.path.join(profiles_dir, f"{model_name}.json")
        print(f"📋 Saving quantization profile to: {profile_file}")
        
        with open(profile_file, "w", encoding="utf-8") as f:
            json.dump(profile, f, indent=2)
        
        return profile_file


NODE_CLASS_MAPPINGS = {"StarModelLayersInfo": StarModelLayersInfo}
NODE_DISPLAY_NAME_MAPPINGS = {"StarModelLayersInfo": "⭐ Star Model Layers Info"}
