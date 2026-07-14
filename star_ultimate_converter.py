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
    from comfy_kitchen.registry import registry as ck_registry
    from comfy_kitchen.tensor import TensorCoreConvRotW4A4Layout, TensorCoreMXFP8Layout, TensorCoreNVFP4Layout, TensorWiseINT8Layout
except ImportError:
    print("⚠️ [Star Ultimate Model Converter] comfy-kitchen not found.")

NODE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_JSON = os.path.join(NODE_DIR, "models.json")

EXTENDED_METADATA_KEYS = ["config", "license", "encrypted_wandb_properties"]

AIO_MODEL_PREFIX = "model.diffusion_model."

TARGET_FORMATS = ["nvfp4", "fp8", "mxfp8", "int8", "int8_convrot", "int4_convrot", "fp16", "fp32"]

CONVROT_GROUPSIZE = 256
INT4_QUANT_GROUPSIZE = 64

PRECISION_RE = re.compile(r"[-_.](fp32|fp16|bf16|mxfp8|fp8(?:_e[45]m[23](?:fn)?)?(?:_scaled)?(?:_fast)?|int[48](?:_convrot)?|nvfp4)(?=[-_.]|$)", re.IGNORECASE)

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


def resolve_input(mode, diffusion_model, checkpoint, text_encoder, custom_path):
    """Resolve the target path based on the selected mode."""
    if mode == "Custom Path":
        custom_path = (custom_path or "").strip().strip('"')
        if not custom_path:
            raise ValueError("Mode is 'Custom Path' but no custom path was provided.")
        src = os.path.abspath(os.path.expanduser(custom_path))
        if os.path.isdir(src):
            files = sorted(glob.glob(os.path.join(src, "*.safetensors")))
            if not files:
                raise ValueError(f"No .safetensors files found in: {src}")
            return files, os.path.dirname(src), os.path.basename(src)
        if os.path.isfile(src):
            return [src], os.path.dirname(src), os.path.splitext(os.path.basename(src))[0]
        raise ValueError(f"Path not found: {src}")

    elif mode == "Diffusion Model":
        if not diffusion_model or diffusion_model == "None":
            raise ValueError("No Diffusion Model selected. Please choose a model from the dropdown.")
        path = folder_paths.get_full_path("diffusion_models", diffusion_model)
        if not path:
            raise ValueError(f"Diffusion Model not found: {diffusion_model}")
        return [path], os.path.dirname(path), os.path.splitext(os.path.basename(path))[0]

    elif mode == "Text-Encoder":
        if not text_encoder or text_encoder == "None":
            raise ValueError("No Text-Encoder selected. Please choose a model from the dropdown.")
        path = folder_paths.get_full_path("text_encoders", text_encoder)
        if not path:  # Fallback to check the clip folder
            path = folder_paths.get_full_path("clip", text_encoder)
        if not path:
            raise ValueError(f"Text-Encoder not found: {text_encoder}")
        return [path], os.path.dirname(path), os.path.splitext(os.path.basename(path))[0]

    raise ValueError(f"Unknown mode: {mode}")


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
    if not ckpt_path:
        raise ValueError(f"Checkpoint not found: {checkpoint_name}")
    full_sd = comfy.utils.load_torch_file(ckpt_path, safe_load=True)
    sd = {k[len(AIO_MODEL_PREFIX):]: v for k, v in full_sd.items() if k.startswith(AIO_MODEL_PREFIX)}
    if not sd:
        raise ValueError(f"No '{AIO_MODEL_PREFIX}' keys found in {os.path.basename(ckpt_path)}. Is this an all-in-one checkpoint?")
    return sd, ckpt_path


def pick_mxfp8_backend(device):
    """Pick a working comfy_kitchen backend for MXFP8 quantization."""
    probe = torch.randn(32, 32, device=device, dtype=torch.float32)
    try:
        TensorCoreMXFP8Layout.quantize(probe)
        return None
    except Exception as e:
        print(f"\u26a0\ufe0f MXFP8 default backend failed ({e}). Trying fallback backends...")
    for backend in ("triton", "eager"):
        try:
            with ck_registry.use_backend(backend):
                TensorCoreMXFP8Layout.quantize(probe)
            print(f"\u2705 MXFP8: using '{backend}' backend")
            return backend
        except Exception:
            continue
    raise RuntimeError("MXFP8 quantization is not supported by any comfy_kitchen backend in this environment. Try updating comfy-kitchen and PyTorch.")


def build_output_path(out_dir, base_name, target_format):
    stem = PRECISION_RE.sub("", base_name).rstrip("-_.")
    return os.path.join(out_dir, f"{stem}-{target_format}.safetensors")


def load_input(files):
    sd = {}
    for i, fp in enumerate(files):
        if len(files) > 1:
            print(f"📦 Loading shard {i + 1}/{len(files)}: {os.path.basename(fp)}")
#        part = safetensors.torch.load_file(fp)
        # NOTE: previously this called safetensors.torch.load_file(fp) directly,
        # which access-violates on very large (60GB+) files on Windows due to
        # an mmap/materialization issue in the safetensors library. ComfyUI's
        # own loader (comfy.utils.load_torch_file) already handles these files
        # fine elsewhere in this same node (Checkpoint/AIO modes), so we route
        # through it here too instead of calling safetensors directly.
        part = comfy.utils.load_torch_file(fp, safe_load=True)
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

    for k in [k for k in sd if k.endswith(".comfy_quant")]:
        conf = sd.pop(k)
        layer = k[: -len(".comfy_quant")]
        if layer not in quant_layers:
            try:
                quant_layers[layer] = json.loads(bytes(conf.cpu().to(torch.uint8).tolist()))
            except Exception:
                print(f"⚠️ Could not parse embedded quant config for '{layer}', ignoring.")

    for layer, info in quant_layers.items():
        fmt = info.get("format")
        if fmt in ("nvfp4", "mxfp8", "convrot_w4a4"):
            raise ValueError(f"Input model contains {fmt} layers ('{layer}'), which cannot be dequantized losslessly. Use a higher precision source model.")
        if info.get("convrot") and fmt != "convrot_w4a4":
            raise ValueError(f"Input model contains ConvRot-rotated INT8 layers ('{layer}'). Use a higher precision source model.")

    if "scaled_fp8" in sd:
        sd.pop("scaled_fp8")
        for k in [k for k in sd if k.endswith(".scale_weight")]:
            scale = sd.pop(k)
            wk = k[: -len(".scale_weight")] + ".weight"
            if wk in sd:
                sd[wk] = (sd[wk].to(torch.float32) * scale.to(torch.float32)).to(torch.bfloat16)
        for k in [k for k in sd if k.endswith(".scale_input")]:
            sd.pop(k)

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

    for k, v in sd.items():
        if v.dtype in FP8_DTYPES:
            sd[k] = v.to(torch.bfloat16)

    return sd


class StarUltimateModelConverter:
    @classmethod
    def INPUT_TYPES(s):
        configs = load_model_configs()
        
        # Text-Encoders
        tenc_list = []
        if "text_encoders" in folder_paths.folder_names_and_paths:
            tenc_list.extend(folder_paths.get_filename_list("text_encoders") or [])
        if "clip" in folder_paths.folder_names_and_paths:
            tenc_list.extend(folder_paths.get_filename_list("clip") or [])
        tenc_list = sorted(list(set(tenc_list)))
        tenc_list.insert(0, "None") # Set None at the very top

        # Diffusion Models
        diff_list = folder_paths.get_filename_list("diffusion_models") or []
        diff_list = sorted(diff_list)
        diff_list.insert(0, "None") # Set None at the very top

        # Checkpoints
        ckpt_list = folder_paths.get_filename_list("checkpoints") or []
        ckpt_list = sorted(ckpt_list)
        ckpt_list.insert(0, "None") # Set None at the very top

        return {
            "required": {
                "mode": (["Diffusion Model", "Checkpoint", "AIO", "Text-Encoder", "Custom Path"], {
                    "default": "Diffusion Model",
                    "tooltip": "Select the source type. 'AIO' processes UNet + CLIP together and ignores VAE."
                }),
                "diffusion_model": (diff_list, {"tooltip": "Used if Mode is 'Diffusion Model'."}),
                "checkpoint": (ckpt_list, {"tooltip": "Used if Mode is 'Checkpoint' or 'AIO'."}),
                "text_encoder": (tenc_list, {"tooltip": "Used if Mode is 'Text-Encoder'."}),
                "model_type": (list(configs["models"].keys()), {
                    "tooltip": "Choose the model architecture profile."
                }),
                "target_format": (TARGET_FORMATS, {
                    "default": "nvfp4"
                }),
                "device": (["cuda", "cpu"], {
                    "default": "cpu"
                }),
            },
            "optional": {
                "custom_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Enter path to .safetensors file or folder",
                    "tooltip": "Used only if Mode is set to 'Custom Path'."
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "convert"
    CATEGORY = '⭐StarNodes/Model Tools'
    OUTPUT_NODE = True

    def convert(self, mode, diffusion_model, checkpoint, text_encoder, model_type, target_format, device, custom_path=""):
        configs = load_model_configs()
        blacklist, fp8_layers, preserve_extended = get_profile(configs, model_type)
        
        # Lade sicherheitshalber auch gleich das Text-Encoder Profil für den AIO Modus mit
        te_blacklist, te_fp8_layers, _ = get_profile(configs, "Text-Encoder")
        
        start_time = time.time()

        print(f"🚀 [Star Ultimate Model Converter] Mode: {mode} | Profile: {model_type} | Target: {target_format}")

        if mode in ("Checkpoint", "AIO"):
            if not checkpoint or checkpoint == "None":
                raise ValueError(f"Mode is '{mode}' but no checkpoint is selected. Please choose a model from the dropdown.")
            
            ckpt_path = folder_paths.get_full_path("checkpoints", checkpoint)
            base_name = os.path.splitext(os.path.basename(ckpt_path))[0]
            
            # Originale Metadaten aus dem Checkpoint retten
            orig_meta = None
            if ckpt_path.endswith(".safetensors"):
                with safetensors.safe_open(ckpt_path, framework="pt") as f:
                    orig_meta = f.metadata()
            
            full_sd = comfy.utils.load_torch_file(ckpt_path, safe_load=True)

            if mode == "Checkpoint":
                print(f"✂️ Extracting diffusion model from AIO checkpoint: {checkpoint}")
                sd = {k[len(AIO_MODEL_PREFIX):]: v for k, v in full_sd.items() if k.startswith(AIO_MODEL_PREFIX)}
                input_bytes = sum(v.numel() * v.element_size() for v in sd.values())
                output_path = build_output_path(diffusion_models_dir(), base_name, target_format)
                files = [ckpt_path]
                del full_sd # Speicher freigeben
            else:
                print(f"🔄 AIO Mode: Processing entire checkpoint intact: {checkpoint}")
                sd = full_sd # Wir behalten das volle Dictionary!
                input_bytes = os.path.getsize(ckpt_path)
                checkpoints_dir = folder_paths.get_folder_paths("checkpoints")[0]
                output_path = build_output_path(checkpoints_dir, f"{base_name}_AIO", target_format)
                files = [ckpt_path]
        else:
            files, out_dir, base_name = resolve_input(mode, diffusion_model, checkpoint, text_encoder, custom_path)
            output_path = build_output_path(out_dir, base_name, target_format)
            input_bytes = sum(os.path.getsize(f) for f in files)
            sd, orig_meta = load_input(files)

        temp_diffusers_meta = {}
        if orig_meta:
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
        mxfp8_backend = pick_mxfp8_backend(device) if target_format == "mxfp8" else None

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

                # --- Dynamische Blacklist-Zuweisung ---
                if mode == "AIO":
                    if k.startswith(AIO_MODEL_PREFIX):
                        # Es ist das Diffusion Model
                        active_blacklist = blacklist
                        active_fp8 = fp8_layers
                    elif k.startswith("cond_stage_model.") or k.startswith("conditioner.") or k.startswith("text_encoders."):
                        # Es ist ein Text-Encoder (CLIP/T5)
                        active_blacklist = te_blacklist
                        active_fp8 = te_fp8_layers
                    else:
                        # Es ist das VAE (first_stage_model) oder andere strukturelle Keys.
                        # Auf GAR KEINEN FALL quantisieren!
                        if v.dtype.is_floating_point:
                            new_sd[k] = v.to(dtype=torch.bfloat16)
                            counts["kept bf16 (VAE/Misc)"] += 1
                        else:
                            new_sd[k] = v
                            counts["kept (VAE/Misc)"] += 1
                        continue
                else:
                    # Normaler Modus (Stand-alone Modell)
                    active_blacklist = blacklist
                    active_fp8 = fp8_layers
                # ----------------------------------------

                if any(name in k for name in active_blacklist):
                    if v.dtype.is_floating_point:
                        new_sd[k] = v.to(dtype=torch.bfloat16)
                        counts["kept bf16"] += 1
                    else:
                        new_sd[k] = v
                        counts["kept"] += 1
                    continue

                if v.ndim == 2 and ".weight" in k:
                    base_k_file = k.replace(".weight", "")
                    # The metadata keys must exactly match the base keys
                    base_k_meta = base_k_file

                    # THIS LINE WAS LIKELY MISSING:
                    v_tensor = v.to(device=device, dtype=torch.bfloat16)

                    # Hier nutzen wir active_fp8 anstelle von fp8_layers
                    if target_format == "fp8" or (active_fp8 and any(name in k for name in active_fp8)):
                        print(f"🌸 FP8: {k}")
                        weight_scale = (v_tensor.abs().max() / 448.0).clamp(min=1e-12).float()
                        weight_quantized = ck.quantize_per_tensor_fp8(v_tensor, weight_scale)
                        new_sd[k] = weight_quantized.cpu()
                        new_sd[f"{base_k_file}.weight_scale"] = weight_scale.to(torch.bfloat16).cpu()
                        quant_map["layers"][base_k_meta] = {"format": "float8_e4m3fn"}
                        counts["fp8"] += 1
                        if device == "cuda": del v_tensor
                        continue

                    int8_convrot = target_format == "int8_convrot"
                    int4_convrot = target_format == "int4_convrot"
                    if target_format in ("int8", "int8_convrot"):
                        layout = TensorWiseINT8Layout
                        fmt_name = "int8_tensorwise"
                    elif target_format == "int4_convrot":
                        layout = TensorCoreConvRotW4A4Layout
                        fmt_name = "convrot_w4a4"
                    elif target_format == "mxfp8":
                        layout = TensorCoreMXFP8Layout
                        fmt_name = "mxfp8"
                    else:
                        layout = TensorCoreNVFP4Layout
                        fmt_name = "nvfp4"
                    print(f"💎 {target_format.upper()}: {k}")
                    
                    try:
                        v_tensor_ready = v_tensor.float().contiguous()

                        if int8_convrot:
                            qdata, params = layout.quantize(v_tensor_ready, per_channel=True, convrot=True, convrot_groupsize=CONVROT_GROUPSIZE)
                        elif int4_convrot:
                            qdata, params = layout.quantize(v_tensor_ready, convrot_groupsize=CONVROT_GROUPSIZE, quant_group_size=INT4_QUANT_GROUPSIZE)
                        elif mxfp8_backend is not None:
                            with ck_registry.use_backend(mxfp8_backend):
                                qdata, params = layout.quantize(v_tensor_ready)
                        else:
                            qdata, params = layout.quantize(v_tensor_ready)
                            
                        tensors = layout.state_dict_tensors(qdata, params)
                        
                        for suffix, tensor in tensors.items():
                            if tensor.dtype == torch.float8_e8m0fnu:
                                new_sd[f"{base_k_file}.weight{suffix}"] = tensor.view(torch.uint8).cpu()
                            elif tensor.dtype in FP8_DTYPES:
                                new_sd[f"{base_k_file}.weight{suffix}"] = tensor.view(torch.uint8).cpu().view(tensor.dtype)
                            else:
                                new_sd[f"{base_k_file}.weight{suffix}"] = tensor.cpu()
                                
                        layer_conf = {"format": fmt_name}
                        if int8_convrot:
                            layer_conf["convrot"] = True
                            layer_conf["convrot_groupsize"] = CONVROT_GROUPSIZE
                        elif int4_convrot:
                            layer_conf["convrot_groupsize"] = CONVROT_GROUPSIZE
                            layer_conf["quant_group_size"] = INT4_QUANT_GROUPSIZE
                        quant_map["layers"][base_k_meta] = layer_conf
                        counts[target_format] += 1
                        
                    except Exception as e:
                        print(f"⚠️ Quantization failed for {k}: {e}")
                        if v.dtype.is_floating_point:
                            new_sd[k] = v.to(dtype=torch.bfloat16)
                            counts["kept bf16"] += 1
                        else:
                            new_sd[k] = v
                            counts["kept"] += 1

                    if device == "cuda": del v_tensor
                else:
                    if v.dtype.is_floating_point:
                        new_sd[k] = v.to(dtype=torch.bfloat16)
                        counts["kept bf16"] += 1
                    else:
                        new_sd[k] = v
                        counts["kept"] += 1

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

        if mode == "AIO":
            input_desc = f"Full AIO Checkpoint {os.path.basename(files[0])}"
        elif mode == "Checkpoint":
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