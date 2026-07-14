"""Star Ultimate Model Converter Pro - Profile-based mixed-precision quantization."""
import os
import time
import json
import torch
import folder_paths
import safetensors
import safetensors.torch
import comfy.utils
from collections import Counter, OrderedDict

try:
    import comfy_kitchen as ck
    from comfy_kitchen.registry import registry as ck_registry
    from comfy_kitchen.tensor import TensorCoreConvRotW4A4Layout, TensorCoreMXFP8Layout, TensorCoreNVFP4Layout, TensorWiseINT8Layout
    COMFY_KITCHEN_AVAILABLE = True
except ImportError:
    print("⚠️ [Star Ultimate Model Converter Pro] comfy-kitchen not found. Deep quantization will not be available.")
    COMFY_KITCHEN_AVAILABLE = False

NODE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_JSON = os.path.join(NODE_DIR, "models.json")

CONVROT_GROUPSIZE = 256
INT4_QUANT_GROUPSIZE = 64
FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)

# Inline utilities
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

# Dtype mapping from JSON strings to PyTorch dtypes
DTYPE_MAP = {
    "FP32": torch.float32,
    "F32": torch.float32,
    "FP16": torch.float16,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "FP8": torch.float8_e4m3fn,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E4M3FN": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
    "INT8": torch.int8,
}

# Normalization and embedding layer patterns
NORM_PATTERNS = ["norm", "ln", "layernorm", "groupnorm", "batchnorm"]
EMBEDDING_PATTERNS = ["embed", "embedding", "token", "position"]


def is_norm_or_embedding(layer_name):
    """Check if layer is normalization or embedding."""
    layer_lower = layer_name.lower()
    return any(pattern in layer_lower for pattern in NORM_PATTERNS + EMBEDDING_PATTERNS)


def load_model_configs():
    """Load model configurations from models.json."""
    with open(MODELS_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def get_blacklist(configs, model_type):
    """Get blacklist for a specific model type."""
    default = configs["default"]
    profile = configs["models"].get(model_type, default)
    return profile.get("blacklist", default["blacklist"])


class StarUltimateModelConverterPro:
    """Profile-based mixed-precision quantization converter."""
    
    @classmethod
    def INPUT_TYPES(s):
        # Get list of diffusion models
        try:
            model_files = folder_paths.get_filename_list("diffusion_models")
        except:
            model_files = []
        
        if not model_files:
            model_files = ["No models found"]
        
        # Get list of profiles
        profiles_dir = os.path.join(os.path.dirname(__file__), "profiles")
        os.makedirs(profiles_dir, exist_ok=True)
        
        profile_files = [f for f in os.listdir(profiles_dir) if f.endswith(".json")]
        if not profile_files:
            profile_files = ["No profiles found"]
        
        # Load model configs for blacklist
        try:
            configs = load_model_configs()
            model_types = list(configs["models"].keys())
        except:
            model_types = ["Unknown"]
        
        return {
            "required": {
                "model_name": (model_files, {
                    "tooltip": "Select a diffusion model from your models/diffusion_models folder to quantize."
                }),
                "profile": (profile_files, {
                    "tooltip": "Select a quantization profile created by Star Model Layers Info."
                }),
                "model_type": (model_types, {
                    "default": "Unknown",
                    "tooltip": "Select the model architecture to use the appropriate blacklist from models.json. Blacklisted layers will be preserved in higher precision."
                }),
                "use_blacklist": (["yes", "no"], {
                    "default": "yes",
                    "tooltip": "Use the blacklist from models.json to preserve critical layers (embeddings, norms, etc.) in BF16."
                }),
                "target_quant_format": (["NVFP4", "INT4_CONVROT", "FP8", "INT8", "INT8_CONVROT", "MXFP8"], {
                    "default": "NVFP4",
                    "tooltip": "Deep quantization format to apply to layers marked for heavy quantization in the profile."
                }),
                "manual_mode": (["disabled", "enabled"], {
                    "default": "disabled",
                    "tooltip": "Enable manual mode to manually control quantization for each source block type."
                }),
                "quantize_fp32": (["yes", "no"], {
                    "default": "no",
                    "tooltip": "[Manual Mode] YES = Convert FP32 layers from profile to FP8 | NO = Keep as FP32"
                }),
                "quantize_bf16": (["yes", "no"], {
                    "default": "yes",
                    "tooltip": "[Manual Mode] YES = Convert BF16 layers from profile to FP8 | NO = Keep as BF16"
                }),
                "quantize_fp16": (["yes", "no"], {
                    "default": "yes",
                    "tooltip": "[Manual Mode] YES = Convert FP16 layers from profile to FP8 | NO = Keep as FP16"
                }),
                "quantize_fp8_e4m3fn": (["yes", "no"], {
                    "default": "yes",
                    "tooltip": "[Manual Mode] YES = Convert FP8_E4M3FN layers from profile to FP8 | NO = Keep as-is"
                }),
                "quantize_fp8": (["yes", "no"], {
                    "default": "yes",
                    "tooltip": "[Manual Mode] YES = Convert FP8 layers from profile to FP8 | NO = Keep as-is"
                }),
                "device": (["cuda", "cpu"], {
                    "default": "cuda",
                    "tooltip": "Device for conversion. CUDA is much faster if you have an NVIDIA GPU."
                }),
            },
            "optional": {
                "output_name": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Leave blank for auto-naming",
                    "tooltip": "Custom name for the output model. If blank, will use original name with profile suffix."
                }),
            },
        }

    RETURN_TYPES = ("STRING", "MODEL")
    RETURN_NAMES = ("status", "model")
    FUNCTION = "convert"
    CATEGORY = '⭐StarNodes/Model Tools'
    OUTPUT_NODE = True

    def convert(self, model_name, profile, model_type, use_blacklist, target_quant_format, manual_mode, 
                quantize_fp32, quantize_bf16, quantize_fp16, quantize_fp8_e4m3fn, quantize_fp8, 
                device, output_name=""):
        start_time = time.time()
        
        print("🔄 [Star Ultimate Model Converter Pro] Starting profile-based conversion...")
        
        # Check model exists
        if model_name == "No models found":
            raise ValueError("No diffusion models found in models/diffusion_models folder.")
        
        # Load profile
        if profile == "No profiles found":
            raise ValueError("No quantization profiles found. Create one using Star Model Layers Info with 'Save Profile' enabled.")
        
        profiles_dir = os.path.join(os.path.dirname(__file__), "profiles")
        profile_path = os.path.join(profiles_dir, profile)
        
        print(f"📋 Loading profile: {profile}")
        with open(profile_path, "r", encoding="utf-8") as f:
            profile_data = json.load(f)
        
        metadata = profile_data.get("__metadata__", {})
        layer_profile = profile_data.get("layers", {})
        
        # Load blacklist from models.json
        blacklist_patterns = []
        if use_blacklist == "yes":
            try:
                configs = load_model_configs()
                blacklist_patterns = get_blacklist(configs, model_type)
                print(f"🚫 Blacklist loaded: {len(blacklist_patterns)} patterns from '{model_type}' profile")
                print(f"   Patterns: {', '.join(blacklist_patterns[:5])}{'...' if len(blacklist_patterns) > 5 else ''}")
            except Exception as e:
                print(f"⚠️ Failed to load blacklist: {e}")
        
        print(f"📦 Profile: {metadata.get('original_model_name', 'Unknown')}")
        print(f"📅 Created: {metadata.get('timestamp', 'Unknown')}")
        print(f"🔢 Profile layers: {len(layer_profile)}")
        print(f"🎮 Manual mode: {manual_mode}")
        
        # Build manual mode settings
        manual_settings = {
            "enabled": manual_mode == "enabled",
            "quantize_fp32": quantize_fp32 == "yes",
            "quantize_bf16": quantize_bf16 == "yes",
            "quantize_fp16": quantize_fp16 == "yes",
            "quantize_fp8_e4m3fn": quantize_fp8_e4m3fn == "yes",
            "quantize_fp8": quantize_fp8 == "yes",
        }
        
        if manual_settings["enabled"]:
            print("⚙️ Manual quantization settings:")
            print(f"  - FP32: {'✓' if manual_settings['quantize_fp32'] else '✗'}")
            print(f"  - BF16: {'✓' if manual_settings['quantize_bf16'] else '✗'}")
            print(f"  - FP16: {'✓' if manual_settings['quantize_fp16'] else '✗'}")
            print(f"  - FP8_E4M3FN: {'✓' if manual_settings['quantize_fp8_e4m3fn'] else '✗'}")
            print(f"  - FP8: {'✓' if manual_settings['quantize_fp8'] else '✗'}")
        
        # Load model from file
        model_path = folder_paths.get_full_path("diffusion_models", model_name)
        
        print(f"📦 Loading model: {model_name}")
        model_sd = safetensors.torch.load_file(model_path)
        
        # Convert model based on profile
        new_sd = OrderedDict()
        conversion_stats = Counter()
        quant_map = {"format_version": "1.0", "layers": {}}
        total_layers = len(model_sd)
        
        print(f"🔄 Converting {total_layers} layers using profile rules...")
        
        for i, (key, tensor) in enumerate(model_sd.items()):
            if (i + 1) % 100 == 0:
                print(f"  Progress: {i+1}/{total_layers}")
            
            # Get profile format for this layer
            profile_format = layer_profile.get(key, None)
            
            # Check if layer matches blacklist patterns (preserve in BF16)
            is_blacklisted = False
            if use_blacklist == "yes" and blacklist_patterns:
                is_blacklisted = any(pattern in key for pattern in blacklist_patterns)
            
            # Apply conversion rules
            result = self._apply_conversion_rules(
                key, tensor, profile_format, target_quant_format, device, conversion_stats, 
                manual_settings, is_blacklisted, quant_map
            )
            
            # Result can be a single tensor or a dict of tensors (for deep quant)
            if isinstance(result, dict):
                new_sd.update(result)
            else:
                new_sd[key] = result
        
        # Determine output name
        if output_name and output_name.strip():
            base_name = output_name.strip()
        else:
            model_base = os.path.splitext(model_name)[0]
            base_name = f"{model_base}_{target_quant_format.lower()}_pro"
        
        if not base_name.endswith(".safetensors"):
            base_name += ".safetensors"
        
        # Prepare metadata
        final_metadata = OrderedDict()
        if quant_map["layers"]:
            final_metadata["_quantization_metadata"] = json.dumps(quant_map)
        final_metadata["converted_by"] = "Star Ultimate Model Converter Pro"
        final_metadata["profile_used"] = profile
        final_metadata["model_type"] = model_type
        final_metadata["target_format"] = target_quant_format
        
        # Save converted model
        output_dir = folder_paths.get_folder_paths("diffusion_models")[0]
        output_path = os.path.join(output_dir, base_name)
        
        print(f"💾 Saving converted model to: {output_path}")
        safetensors.torch.save_file(new_sd, output_path, metadata=final_metadata)
        
        output_bytes = os.path.getsize(output_path)
        duration = time.time() - start_time
        
        # Build status
        status_lines = [
            "✅ Profile-based conversion complete",
            f"Profile: {metadata.get('original_model_name', 'Unknown')}",
            f"Target format: {target_quant_format}",
            f"Total layers: {total_layers}",
            "",
            "Conversion breakdown:",
        ]
        
        for fmt, count in conversion_stats.most_common():
            status_lines.append(f"  - {fmt}: {count} layers")
        
        status_lines.extend([
            "",
            f"Output size: {format_size(output_bytes)}",
            f"Time: {duration:.1f}s",
            f"Saved to: {output_path}",
        ])
        
        status = "\n".join(status_lines)
        
        # Print to console
        print("\n" + "="*60)
        print(status)
        print("="*60 + "\n")
        
        # Load converted model
        print("🔄 Loading converted model...")
        import comfy.sd
        model_out = comfy.sd.load_diffusion_model(output_path)
        
        return (status, model_out)
    
    def _apply_conversion_rules(self, key, tensor, profile_format, target_quant_format, device, stats, manual_settings, is_blacklisted, quant_map):
        """Apply strict conversion rules based on profile format."""
        
        # Non-floating point tensors pass through
        if not tensor.dtype.is_floating_point:
            stats["kept_non_float"] += 1
            return tensor
        
        # Blacklist override: preserve in BF16 (highest priority)
        if is_blacklisted:
            stats["blacklisted_to_bf16"] += 1
            return tensor.to(torch.bfloat16).contiguous()
        
        # Normalize profile format (remove extra info like "+ SCALE", "_SCALE", etc.)
        if profile_format:
            # Extract base format from strings like "FP8_E4M3FN + SCALE" or "TORCH.UINT8 + SCALE"
            profile_format_clean = profile_format.split("+")[0].split("_SCALE")[0].strip()
            # Handle special cases
            if "TORCH.UINT8" in profile_format_clean or "UINT8" in profile_format_clean:
                profile_format_clean = "INT4_CONVROT"  # Treat as deep quantization
            elif "METADATA" in profile_format_clean:
                stats["skipped_metadata"] += 1
                return tensor.contiguous()
        else:
            profile_format_clean = None
        
        # Rule 5 Exception: Missing layer that is norm/embedding → FP16/BF16
        if profile_format_clean is None:
            if is_norm_or_embedding(key):
                stats["missing_norm_to_fp16"] += 1
                return tensor.to(torch.float16).contiguous()
            else:
                # Rule 5: Missing layer → FP8
                stats["missing_to_fp8"] += 1
                return tensor.to(torch.float8_e4m3fn).contiguous()
        
        # Rule 1: FP32 in profile → Keep high precision (don't quantize) unless manual mode overrides
        if profile_format_clean in ["FP32", "F32"]:
            if manual_settings["enabled"]:
                if manual_settings["quantize_fp32"]:
                    # Manual mode YES: quantize FP32 to FP8
                    stats["manual_fp32_to_fp8"] += 1
                    return tensor.to(torch.float8_e4m3fn).contiguous()
                else:
                    # Manual mode NO: keep FP32
                    stats["manual_kept_fp32"] += 1
                    return tensor.contiguous()
            # Normal mode: keep FP32
            if tensor.dtype == torch.float32:
                stats["kept_fp32"] += 1
                return tensor.contiguous()
            else:
                stats["kept_lower_precision"] += 1
                return tensor.contiguous()
        
        # Rule 2: FP16/BF16 in profile → Convert to half precision unless manual mode overrides
        if profile_format_clean in ["FP16", "F16", "BF16"]:
            if manual_settings["enabled"]:
                # Manual mode: check if user wants to quantize this type
                if profile_format_clean in ["BF16"] and manual_settings["quantize_bf16"]:
                    # YES: quantize BF16 to FP8
                    stats["manual_bf16_to_fp8"] += 1
                    return tensor.to(torch.float8_e4m3fn).contiguous()
                elif profile_format_clean in ["FP16", "F16"] and manual_settings["quantize_fp16"]:
                    # YES: quantize FP16 to FP8
                    stats["manual_fp16_to_fp8"] += 1
                    return tensor.to(torch.float8_e4m3fn).contiguous()
                else:
                    # NO: keep original format
                    stats[f"manual_kept_{profile_format_clean.lower()}"] += 1
                    target_dtype = DTYPE_MAP.get(profile_format_clean, torch.float16)
                    return tensor.to(target_dtype).contiguous()
            
            # Normal mode: convert to half precision
            target_dtype = DTYPE_MAP.get(profile_format_clean, torch.float16)
            if tensor.dtype in [torch.float32]:
                stats[f"fp32_to_{profile_format_clean.lower()}"] += 1
                return tensor.to(target_dtype).contiguous()
            else:
                stats[f"kept_{profile_format_clean.lower()}"] += 1
                return tensor.contiguous()
        
        # Rule 3: FP8/INT8 in profile → Convert to 8-bit unless manual mode overrides
        if profile_format_clean in ["FP8", "F8_E4M3", "F8_E4M3FN", "FP8_E4M3FN", "INT8"]:
            if manual_settings["enabled"]:
                # Manual mode: check if user wants to quantize this type
                should_quantize = False
                if profile_format_clean in ["FP8_E4M3FN", "F8_E4M3FN"]:
                    should_quantize = manual_settings["quantize_fp8_e4m3fn"]
                elif profile_format_clean in ["FP8", "F8_E4M3"]:
                    should_quantize = manual_settings["quantize_fp8"]
                elif profile_format_clean == "INT8":
                    should_quantize = manual_settings["quantize_fp8"]  # Use FP8 setting for INT8
                
                if should_quantize:
                    # YES: convert to FP8
                    stats[f"manual_{profile_format_clean.lower()}_to_fp8"] += 1
                    return tensor.to(torch.float8_e4m3fn).contiguous()
                else:
                    # NO: keep as-is (don't quantize further)
                    stats[f"manual_kept_{profile_format_clean.lower()}"] += 1
                    return tensor.contiguous()
            
            # Normal mode: convert to 8-bit
            target_dtype = DTYPE_MAP.get(profile_format_clean, torch.float8_e4m3fn)
            if tensor.dtype in [torch.float32, torch.float16, torch.bfloat16]:
                stats[f"to_{profile_format_clean.lower()}"] += 1
                return tensor.to(target_dtype).contiguous()
            else:
                stats[f"kept_{profile_format_clean.lower()}"] += 1
                return tensor.contiguous()
        
        # Rule 4: Deep quantization (INT4, NVFP4, etc.) → Use target_quant_format
        if profile_format_clean in ["INT4", "INT4_CONVROT", "NVFP4", "MXFP8", "INT8_CONVROT", "TORCH.UINT8", "UINT8"]:
            # Manual mode doesn't apply to deep quantization - always convert
            # Only quantize 2D weight tensors
            if tensor.ndim == 2 and ".weight" in key:
                return self._apply_deep_quantization(key, tensor, target_quant_format, device, stats, quant_map)
            else:
                # Non-weight tensors: keep as BF16
                if tensor.dtype.is_floating_point:
                    stats["kept_bf16"] += 1
                    return tensor.to(torch.bfloat16).contiguous()
                else:
                    stats["kept"] += 1
                    return tensor.contiguous()
        
        # Default: Keep as-is
        stats["kept_default"] += 1
        return tensor.contiguous()
    
    def _apply_deep_quantization(self, key, tensor, target_quant_format, device, stats, quant_map):
        """Apply deep quantization using comfy-kitchen (NVFP4, INT4_CONVROT, etc.)."""
        
        if not COMFY_KITCHEN_AVAILABLE:
            print(f"⚠️ comfy-kitchen not available, keeping {key} as BF16")
            stats["kept_bf16_no_ck"] += 1
            return tensor.to(torch.bfloat16).contiguous()
        
        base_k_file = key.replace(".weight", "")
        base_k_meta = base_k_file
        
        # Move to device and convert to BF16 first
        v_tensor = tensor.to(device=device, dtype=torch.bfloat16)
        
        # Determine layout and format
        int8_convrot = target_quant_format.upper() == "INT8_CONVROT"
        int4_convrot = target_quant_format.upper() == "INT4_CONVROT"
        
        if target_quant_format.upper() in ("INT8", "INT8_CONVROT"):
            layout = TensorWiseINT8Layout
            fmt_name = "int8_tensorwise"
        elif target_quant_format.upper() == "INT4_CONVROT":
            layout = TensorCoreConvRotW4A4Layout
            fmt_name = "convrot_w4a4"
        elif target_quant_format.upper() == "MXFP8":
            layout = TensorCoreMXFP8Layout
            fmt_name = "mxfp8"
        elif target_quant_format.upper() == "FP8":
            # Use simple FP8 quantization
            weight_scale = (v_tensor.abs().max() / 448.0).clamp(min=1e-12).float()
            weight_quantized = ck.quantize_per_tensor_fp8(v_tensor, weight_scale)
            quant_map["layers"][base_k_meta] = {"format": "float8_e4m3fn"}
            stats["fp8"] += 1
            
            result_dict = {
                key: weight_quantized.cpu(),
                f"{base_k_file}.weight_scale": weight_scale.to(torch.bfloat16).cpu()
            }
            if device == "cuda":
                del v_tensor
            return result_dict
        else:  # NVFP4 (default)
            layout = TensorCoreNVFP4Layout
            fmt_name = "nvfp4"
        
        print(f"💎 {target_quant_format.upper()}: {key}")
        
        try:
            v_tensor_ready = v_tensor.float().contiguous()
            
            # Quantize based on format
            if int8_convrot:
                qdata, params = layout.quantize(v_tensor_ready, per_channel=True, convrot=True, convrot_groupsize=CONVROT_GROUPSIZE)
            elif int4_convrot:
                qdata, params = layout.quantize(v_tensor_ready, convrot_groupsize=CONVROT_GROUPSIZE, quant_group_size=INT4_QUANT_GROUPSIZE)
            else:
                qdata, params = layout.quantize(v_tensor_ready)
            
            tensors = layout.state_dict_tensors(qdata, params)
            
            # Build result dictionary
            result_dict = {}
            for suffix, t in tensors.items():
                if t.dtype == torch.float8_e8m0fnu:
                    result_dict[f"{base_k_file}.weight{suffix}"] = t.view(torch.uint8).cpu()
                elif t.dtype in FP8_DTYPES:
                    result_dict[f"{base_k_file}.weight{suffix}"] = t.view(torch.uint8).cpu().view(t.dtype)
                else:
                    result_dict[f"{base_k_file}.weight{suffix}"] = t.cpu()
            
            # Update quantization metadata
            layer_conf = {"format": fmt_name}
            if int8_convrot:
                layer_conf["convrot"] = True
                layer_conf["convrot_groupsize"] = CONVROT_GROUPSIZE
            elif int4_convrot:
                layer_conf["convrot_groupsize"] = CONVROT_GROUPSIZE
                layer_conf["quant_group_size"] = INT4_QUANT_GROUPSIZE
            quant_map["layers"][base_k_meta] = layer_conf
            stats[target_quant_format.lower()] += 1
            
            if device == "cuda":
                del v_tensor, v_tensor_ready
            
            return result_dict
            
        except Exception as e:
            print(f"⚠️ Quantization failed for {key}: {e}")
            if tensor.dtype.is_floating_point:
                stats["kept_bf16_quant_failed"] += 1
                return tensor.to(torch.bfloat16).contiguous()
            else:
                stats["kept_quant_failed"] += 1
                return tensor.contiguous()


NODE_CLASS_MAPPINGS = {"StarUltimateModelConverterPro": StarUltimateModelConverterPro}
NODE_DISPLAY_NAME_MAPPINGS = {"StarUltimateModelConverterPro": "⭐ Star Ultimate Model Converter Pro"}
