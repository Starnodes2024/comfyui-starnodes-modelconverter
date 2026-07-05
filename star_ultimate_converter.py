import os
import re
import json
import time
import glob
import torch
import folder_paths
import safetensors
import safetensors.torch
import comfy.utils
from collections import Counter, OrderedDict

try:
    import comfy_kitchen as ck
    from comfy_kitchen.tensor import TensorCoreNVFP4Layout, TensorWiseINT8Layout
except ImportError:
    print("⚠️ [Star Ultimate Model Converter] comfy-kitchen not found.")

NODE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_JSON = os.path.join(NODE_DIR, "models.json")

EXTENDED_METADATA_KEYS = ["config", "license", "encrypted_wandb_properties"]

AIO_MODEL_PREFIX = "model.diffusion_model."

TARGET_FORMATS = ["nvfp4", "fp8", "int8", "int8_convrot", "fp16", "fp32"]

CONVROT_GROUPSIZE = 256

PRECISION_RE = re.compile(r"[-_.](fp32|fp16|bf16|fp8(?:_e[45]m[23](?:fn)?)?(?:_scaled)?(?:_fast)?|int8(?:_convrot)?|nvfp4)(?=[-_.]|$)", re.IGNORECASE)

FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)

DTYPE_NAMES = {
    torch.float32: "fp32",
    torch.float16: "fp16",
    torch.bfloat16: "bf16",
    torch.float8_e4m3fn: "fp8_e4m3fn",
    torch.float8_e5m2: "fp8_e5m2",
    torch.int8: "int8",
}


def detect_input_format(sd, metadata):
    counts = Counter(DTYPE_NAMES.get(v.dtype, str(v.dtype)) for v in sd.values())
    parts = [f"{name} ({n} tensors)" for name, n in counts.most_common()]
    fmt = ", ".join(parts)
    if "scaled_fp8" in sd:
        fmt += " [ComfyUI scaled fp8]"
    elif metadata and "_quantization_metadata" in metadata:
        fmt += " [quantization metadata]"
    return fmt


def format_size(num_bytes):
    return f"{num_bytes / (1024**3):.2f} GB"


def load_model_configs():
    with open(MODELS_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def get_profile(configs, model_type):
    default = configs["default"]
    profile = configs["models"].get(model_type, default)
    return (
        profile.get("blacklist", default["blacklist"]),
        profile.get("fp8_layers", default["fp8_layers"]),
        profile.get("preserve_extended_metadata", default["preserve_extended_metadata"]),
    )


def resolve_input(model_name, custom_path, use_custom_path):
    """Return (list of safetensors files, output dir, base name for the output file)."""
    custom_path = (custom_path or "").strip().strip('"')
    if use_custom_path and custom_path:
        src = os.path.abspath(os.path.expanduser(custom_path))
        if os.path.isdir(src):
            files = sorted(glob.glob(os.path.join(src, "*.safetensors")))
            if not files:
                raise ValueError(f"No .safetensors files found in: {src}")
            return files, os.path.dirname(src), os.path.basename(src)
        if os.path.isfile(src):
            return [src], os.path.dirname(src), os.path.splitext(os.path.basename(src))[0]
        raise ValueError(f"Path not found: {src}")
    path = folder_paths.get_full_path("diffusion_models", model_name)
    return [path], os.path.dirname(path), os.path.splitext(os.path.basename(path))[0]


def diffusion_models_dir():
    """Prefer models/diffusion_models over the legacy models/unet path."""
    paths = folder_paths.get_folder_paths("diffusion_models")
    for p in paths:
        if os.path.basename(os.path.normpath(p)) == "diffusion_models":
            return p
    return paths[0]


def load_aio_model(checkpoint_name):
    """Load an AIO checkpoint and return only its diffusion model state dict."""
    ckpt_path = folder_paths.get_full_path("checkpoints", checkpoint_name)
    full_sd = comfy.utils.load_torch_file(ckpt_path, safe_load=True)
    sd = {k[len(AIO_MODEL_PREFIX):]: v for k, v in full_sd.items() if k.startswith(AIO_MODEL_PREFIX)}
    if not sd:
        raise ValueError(f"No '{AIO_MODEL_PREFIX}' keys found in {os.path.basename(ckpt_path)}. Is this an all-in-one checkpoint?")
    return sd, ckpt_path


def build_output_path(out_dir, base_name, target_format):
    stem = PRECISION_RE.sub("", base_name).rstrip("-_.")
    return os.path.join(out_dir, f"{stem}-{target_format}.safetensors")


def load_input(files):
    sd = {}
    for i, fp in enumerate(files):
        if len(files) > 1:
            print(f"📦 Loading shard {i + 1}/{len(files)}: {os.path.basename(fp)}")
        part = safetensors.torch.load_file(fp)
        for k in part:
            if k in sd:
                print(f"⚠️ Duplicate key '{k}' in {os.path.basename(fp)}, overwriting")
        sd.update(part)
    with safetensors.safe_open(files[0], framework="pt") as f:
        orig_meta = f.metadata()
    return sd, orig_meta


def dequantize_input(sd, metadata):
    """Unquantize fp8/int8 inputs back to bf16 so mixed-precision models re-quantize cleanly."""
    quant_layers = {}
    if metadata and "_quantization_metadata" in metadata:
        quant_layers = json.loads(metadata["_quantization_metadata"]).get("layers", {})
        if any(info.get("format") == "nvfp4" for info in quant_layers.values()):
            raise ValueError("Input model contains NVFP4 layers, which cannot be dequantized losslessly. Use a higher precision source model.")
        if any(info.get("convrot") for info in quant_layers.values()):
            raise ValueError("Input model contains ConvRot-rotated INT8 layers. Use a higher precision source model.")

    # ComfyUI scaled fp8 checkpoints: 'scaled_fp8' marker + per-layer '.scale_weight' keys
    if "scaled_fp8" in sd:
        sd.pop("scaled_fp8")
        for k in [k for k in sd if k.endswith(".scale_weight")]:
            scale = sd.pop(k)
            wk = k[: -len(".scale_weight")] + ".weight"
            if wk in sd:
                sd[wk] = (sd[wk].to(torch.float32) * scale.to(torch.float32)).to(torch.bfloat16)
        for k in [k for k in sd if k.endswith(".scale_input")]:
            sd.pop(k)

    # fp8/int8 weights with a '.weight_scale' sibling (comfy-kitchen style per-tensor scale)
    for k in list(sd.keys()):
        if k not in sd or not k.endswith(".weight"):
            continue
        v = sd[k]
        if v.dtype in FP8_DTYPES or v.dtype == torch.int8:
            scale = sd.pop(k + "_scale", None)
            if scale is not None:
                sd[k] = (v.to(torch.float32) * scale.to(torch.float32)).to(torch.bfloat16)
            elif v.dtype == torch.int8:
                raise ValueError(f"int8 weight '{k}' has no '{k}_scale' tensor, cannot dequantize.")

    # any leftover raw fp8 tensors: plain cast back to bf16
    for k, v in sd.items():
        if v.dtype in FP8_DTYPES:
            sd[k] = v.to(torch.bfloat16)

    return sd


class StarUltimateModelConverter:
    @classmethod
    def INPUT_TYPES(s):
        configs = load_model_configs()
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("diffusion_models"), {
                    "tooltip": "Select a model from your ComfyUI diffusion_models folder. This will be used unless 'Use Custom Path' or 'Use Model from AIO Checkpoint' is enabled."
                }),
                "model_type": (list(configs["models"].keys()), {
                    "tooltip": "Choose the model architecture profile. This determines which layers to keep in high precision and which can be quantized safely."
                }),
                "target_format": (TARGET_FORMATS, {
                    "default": "nvfp4",
                    "tooltip": "Target quantization format: nvfp4 (smallest, NVIDIA only), fp8 (small, good quality), int8 (compatible), int8_convrot (int8 with Hadamard rotation, better quality), fp16 (standard), fp32 (full precision)."
                }),
                "device": (["cuda", "cpu"], {
                    "default": "cuda",
                    "tooltip": "Device for conversion. CUDA is much faster if you have an NVIDIA GPU. CPU works for all systems but is slower."
                }),
            },
            "optional": {
                "use_custom_path": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable this to use a custom file path instead of selecting from the model list. When enabled, the path below will be used."
                }),
                "custom_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Enter path to .safetensors file or folder",
                    "tooltip": "Full path to a .safetensors file or a folder containing model shards. Only used when 'Use Custom Path' is enabled. Supports HuggingFace cache folders."
                }),
                "use_aio_checkpoint": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable this to convert directly from an all-in-one checkpoint. ONLY the diffusion model (UNet) is extracted and converted - text encoder and VAE are NOT included. The result is saved to models/diffusion_models."
                }),
                "checkpoint_name": (folder_paths.get_filename_list("checkpoints"), {
                    "tooltip": "Select an all-in-one checkpoint from your ComfyUI checkpoints folder. Only used when 'Use Model from AIO Checkpoint' is enabled. Only the diffusion model part will be converted."
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "convert"
    CATEGORY = "Star"
    OUTPUT_NODE = True

    def convert(self, model_name, model_type, target_format, device, use_custom_path=False, custom_path="", use_aio_checkpoint=False, checkpoint_name=None):
        configs = load_model_configs()
        blacklist, fp8_layers, preserve_extended = get_profile(configs, model_type)
        start_time = time.time()

        print(f"🚀 [Star Ultimate Model Converter] Profile: {model_type} | Target: {target_format}")

        if use_aio_checkpoint:
            if not checkpoint_name:
                raise ValueError("'Use Model from AIO Checkpoint' is enabled but no checkpoint is selected.")
            print(f"✂️ Extracting diffusion model from AIO checkpoint: {checkpoint_name}")
            sd, ckpt_path = load_aio_model(checkpoint_name)
            files = [ckpt_path]
            orig_meta = None
            base_name = os.path.splitext(os.path.basename(ckpt_path))[0]
            output_path = build_output_path(diffusion_models_dir(), base_name, target_format)
            input_bytes = sum(v.numel() * v.element_size() for v in sd.values())
        else:
            files, out_dir, base_name = resolve_input(model_name, custom_path, use_custom_path)
            output_path = build_output_path(out_dir, base_name, target_format)
            input_bytes = sum(os.path.getsize(f) for f in files)
            sd, orig_meta = load_input(files)

        temp_diffusers_meta = {}
        if orig_meta:
            # Always keep the format to avoid wait_stream errors
            if "format" in orig_meta:
                temp_diffusers_meta["format"] = orig_meta["format"]
            if "modelspec.architecture" in orig_meta:
                temp_diffusers_meta["modelspec.architecture"] = orig_meta["modelspec.architecture"]
            if preserve_extended:
                for key in EXTENDED_METADATA_KEYS:
                    if key in orig_meta:
                        temp_diffusers_meta[key] = orig_meta[key]

        input_format = detect_input_format(sd, orig_meta)
        sd = dequantize_input(sd, orig_meta)

        quant_map = {"format_version": "1.0", "layers": {}}
        new_sd = {}
        counts = Counter()

        pbar = comfy.utils.ProgressBar(len(sd))
        print(f"⚙️ Converting on: {device}")

        if target_format in ("fp16", "fp32"):
            target_dtype = torch.float16 if target_format == "fp16" else torch.float32
            for i, (k, v) in enumerate(sd.items()):
                pbar.update_absolute(i + 1)
                if v.dtype.is_floating_point:
                    new_sd[k] = v.to(target_dtype)
                    counts[target_format] += 1
                else:
                    new_sd[k] = v
                    counts["kept"] += 1
        else:
            for i, (k, v) in enumerate(sd.items()):
                pbar.update_absolute(i + 1)

                if any(name in k for name in blacklist):
                    new_sd[k] = v.to(dtype=torch.bfloat16)
                    counts["kept bf16"] += 1
                    continue

                if v.ndim == 2 and ".weight" in k:
                    base_k_file = k.replace(".weight", "")
                    if "model.diffusion_model." in base_k_file:
                        base_k_meta = base_k_file.split("model.diffusion_model.")[-1]
                    else:
                        base_k_meta = base_k_file

                    v_tensor = v.to(device=device, dtype=torch.bfloat16)

                    if target_format == "fp8" or (fp8_layers and any(name in k for name in fp8_layers)):
                        print(f"🌸 FP8: {k}")
                        weight_scale = (v_tensor.abs().max() / 448.0).clamp(min=1e-12).float()
                        weight_quantized = ck.quantize_per_tensor_fp8(v_tensor, weight_scale)
                        new_sd[k] = weight_quantized.cpu()
                        new_sd[f"{base_k_file}.weight_scale"] = weight_scale.to(torch.bfloat16).cpu()
                        quant_map["layers"][base_k_meta] = {"format": "float8_e4m3fn"}
                        counts["fp8"] += 1
                        if device == "cuda": del v_tensor
                        continue

                    convrot = target_format == "int8_convrot"
                    if target_format in ("int8", "int8_convrot"):
                        layout = TensorWiseINT8Layout
                        fmt_name = "int8_tensorwise"
                    else:
                        layout = TensorCoreNVFP4Layout
                        fmt_name = "nvfp4"
                    print(f"💎 {target_format.upper()}: {k}")
                    try:
                        if convrot:
                            qdata, params = layout.quantize(v_tensor.float(), per_channel=True, convrot=True, convrot_groupsize=CONVROT_GROUPSIZE)
                        else:
                            qdata, params = layout.quantize(v_tensor)
                        tensors = layout.state_dict_tensors(qdata, params)
                        for suffix, tensor in tensors.items():
                            new_sd[f"{base_k_file}.weight{suffix}"] = tensor.cpu()
                        layer_conf = {"format": fmt_name}
                        if convrot:
                            layer_conf["convrot"] = True
                            layer_conf["convrot_groupsize"] = CONVROT_GROUPSIZE
                        quant_map["layers"][base_k_meta] = layer_conf
                        counts[target_format] += 1
                    except Exception:
                        new_sd[k] = v.to(dtype=torch.bfloat16)
                        counts["kept bf16"] += 1

                    if device == "cuda": del v_tensor
                else:
                    new_sd[k] = v.to(dtype=torch.bfloat16)
                    counts["kept bf16"] += 1

        final_metadata = OrderedDict()
        if quant_map["layers"]:
            final_metadata["_quantization_metadata"] = json.dumps(quant_map)
        final_metadata["converted_by"] = "Star Ultimate Model Converter"

        for k, v in temp_diffusers_meta.items():
            final_metadata[k] = v

        print(f"💾 Saving | Type: {model_type} | Path: {output_path}")
        safetensors.torch.save_file(new_sd, output_path, metadata=final_metadata)

        output_bytes = os.path.getsize(output_path)
        duration = time.time() - start_time
        reduction = (1 - output_bytes / input_bytes) * 100 if input_bytes else 0
        print(f"✅ Done. Final size: {format_size(output_bytes)}")

        if use_aio_checkpoint:
            input_desc = f"diffusion model from AIO checkpoint {os.path.basename(files[0])}"
        elif len(files) > 1:
            input_desc = f"{len(files)} files from {os.path.basename(os.path.dirname(files[0]))}"
        else:
            input_desc = os.path.basename(files[0])
        layers_desc = ", ".join(f"{n} {name}" for name, n in counts.most_common())
        status = "\n".join([
            f"✅ Success ({model_type} → {target_format})",
            f"Input: {input_desc}",
            f"Original format: {input_format}",
            f"Original size: {format_size(input_bytes)}",
            f"New size: {format_size(output_bytes)} ({reduction:.1f}% smaller)",
            f"Layers: {layers_desc}",
            f"Device: {device} | Time: {duration:.1f}s",
            f"Saved to: {output_path}",
        ])
        return (status,)


NODE_CLASS_MAPPINGS = {"StarUltimateModelConverter": StarUltimateModelConverter}
NODE_DISPLAY_NAME_MAPPINGS = {"StarUltimateModelConverter": "⭐ Star Ultimate Model Converter"}
