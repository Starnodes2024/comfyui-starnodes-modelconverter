import os
import time
import torch
import folder_paths
import safetensors.torch
import comfy.utils

COMPONENTS = {
    "model": {
        "prefixes": ["model.diffusion_model."],
        "folder": "diffusion_models",
        "suffix": "_model",
    },
    "clip": {
        "prefixes": ["cond_stage_model.", "conditioner.", "text_encoders."],
        "folder": "text_encoders",
        "suffix": "_clip",
    },
    "vae": {
        "prefixes": ["first_stage_model.", "vae."],
        "folder": "vae",
        "suffix": "_vae",
    },
}


def format_size(num_bytes):
    return f"{num_bytes / (1024**3):.2f} GB"


def get_output_dir(folder_name):
    """Prefer the path whose basename matches the folder name (e.g. models/diffusion_models over legacy models/unet)."""
    paths = folder_paths.get_folder_paths(folder_name)
    for p in paths:
        if os.path.basename(os.path.normpath(p)) == folder_name:
            return p
    return paths[0]


def extract_component(sd, prefixes):
    """Return a state dict with matching keys, top-level prefix stripped."""
    out = {}
    for k, v in sd.items():
        for prefix in prefixes:
            if k.startswith(prefix):
                out[k[len(prefix):]] = v
                break
    return out


class StarnodesAIOSplitter:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "checkpoint_name": (folder_paths.get_filename_list("checkpoints"), {
                    "tooltip": "Select an all-in-one checkpoint from your ComfyUI checkpoints folder to split into its components."
                }),
                "save_model": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Extract the diffusion model and save it to models/diffusion_models with the '_model' suffix."
                }),
                "save_text_encoder": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Extract the text encoder (CLIP) and save it to models/text_encoders with the '_clip' suffix."
                }),
                "save_vae": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Extract the VAE and save it to models/vae with the '_vae' suffix."
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "split"
    CATEGORY = "Star"
    OUTPUT_NODE = True

    def split(self, checkpoint_name, save_model, save_text_encoder, save_vae):
        selected = {
            "model": save_model,
            "clip": save_text_encoder,
            "vae": save_vae,
        }
        if not any(selected.values()):
            raise ValueError("Nothing to save: enable at least one of Model, Text-Encoder or VAE.")

        ckpt_path = folder_paths.get_full_path("checkpoints", checkpoint_name)
        stem = os.path.splitext(os.path.basename(ckpt_path))[0]
        input_bytes = os.path.getsize(ckpt_path)
        start_time = time.time()

        print(f"✂️ [Starnodes AIO Splitter] Loading: {os.path.basename(ckpt_path)}")
        sd = comfy.utils.load_torch_file(ckpt_path, safe_load=True)

        lines = [f"✂️ Split: {os.path.basename(ckpt_path)} ({format_size(input_bytes)})"]
        saved_any = False

        for name, conf in COMPONENTS.items():
            if not selected[name]:
                continue
            part = extract_component(sd, conf["prefixes"])
            if not part:
                lines.append(f"⚠️ {name}: no matching keys found, skipped")
                continue
            out_dir = get_output_dir(conf["folder"])
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{stem}{conf['suffix']}.safetensors")
            part = {k: v.contiguous() for k, v in part.items()}
            print(f"💾 Saving {name}: {out_path}")
            safetensors.torch.save_file(part, out_path)
            lines.append(f"✅ {name}: {len(part)} tensors, {format_size(os.path.getsize(out_path))} → {out_path}")
            saved_any = True

        if not saved_any:
            raise ValueError("No components could be extracted. Is this a full all-in-one checkpoint?")

        lines.append(f"Time: {time.time() - start_time:.1f}s")
        return ("\n".join(lines),)


NODE_CLASS_MAPPINGS = {"StarnodesAIOSplitter": StarnodesAIOSplitter}
NODE_DISPLAY_NAME_MAPPINGS = {"StarnodesAIOSplitter": "⭐ Starnodes AIO Splitter"}
