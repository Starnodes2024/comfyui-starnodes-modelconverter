import os
import time
import folder_paths
from safetensors import safe_open
import safetensors.torch

def format_size(num_bytes):
    return f"{num_bytes / (1024**3):.2f} GB"

class StarnodesAIOSaver:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (["None"] + folder_paths.get_filename_list("diffusion_models"), {
                    "tooltip": "Choose a diffusion model."
                }),
                "clip_name": (["None"] + folder_paths.get_filename_list("text_encoders"), {
                    "tooltip": "Choose a Text Encoder (CLIP)."
                }),
                "vae_name": (["None"] + folder_paths.get_filename_list("vae"), {
                    "tooltip": "Choose a VAE."
                }),
                "output_name": ("STRING", {
                    "default": "Star_Merged_AIO",
                    "tooltip": "Name of the final model file in your checkpoints folder."
                }),
                "clip_type_setting": (
                    [
                        "stable_diffusion", "stable_cascade", "sd3", "stable_audio", "mochi", 
                        "ltxv", "pixart", "cosmos", "lumina2", "wan", "hidream", "chroma", 
                        "ace", "omnigen2", "qwen_image", "hunyuan_image", "flux2", "ovis", 
                        "longcat_image", "cogvideox", "lens", "pixeldit", "ideogram4", 
                        "boogu", "krea2"
                    ], 
                    {
                        "default": "ltxv", 
                        "tooltip": "Saves the clip type within the metadata of the model."
                    }
                ),
                "model_prefix": ("STRING", {"default": "model.diffusion_model."}),
                "clip_prefix": ("STRING", {"default": "cond_stage_model."}),
                "vae_prefix": ("STRING", {"default": "first_stage_model."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "merge"
    CATEGORY = "Star"
    OUTPUT_NODE = True

    def merge(self, model_name, clip_name, vae_name, output_name, clip_type_setting, model_prefix, clip_prefix, vae_prefix):
        merged_sd = {}
        merged_metadata = {}
        lines = []
        start_time = time.time()

        ckpt_dir = folder_paths.get_folder_paths("checkpoints")[0]

        # output_name is free text. Strip it down to a bare filename so it
        # can't be used to write outside ckpt_dir (e.g. "../../evil" or an
        # absolute path like "C:\\evil").
        stripped_name = (output_name or "").strip()
        safe_output_name = os.path.basename(stripped_name)
        if not safe_output_name or safe_output_name != stripped_name:
            raise ValueError(
                "output_name must be a plain filename without path separators or a drive letter."
            )
        output_name = safe_output_name

        out_path = os.path.join(ckpt_dir, f"{output_name}.safetensors")

        # Defense in depth: confirm the resolved path still lands inside
        # ckpt_dir before writing anything to disk.
        ckpt_dir_real = os.path.realpath(ckpt_dir)
        out_path_real = os.path.realpath(out_path)
        if os.path.commonpath([out_path_real, ckpt_dir_real]) != ckpt_dir_real:
            raise ValueError("Resolved output path escapes the checkpoints folder.")

        components = [
            ("Model", model_name, "diffusion_models", model_prefix),
            ("CLIP", clip_name, "text_encoders", clip_prefix),
            ("VAE", vae_name, "vae", vae_prefix)
        ]

        merged_any = False

        for name, filename, folder, prefix in components:
            if filename == "None":
                continue

            file_path = folder_paths.get_full_path(folder, filename)
            if not file_path:
                lines.append(f"⚠️ {name}: File not found, skipping.")
                continue

            print(f"🔄 [Starnodes AIO Saver] Processing {name}: {filename}")
            try:
                with safe_open(file_path, framework="pt", device="cpu") as f:
                    
                    file_meta = f.metadata() or {}
                    for mk, mv in file_meta.items():
                        if mk not in merged_metadata:
                            merged_metadata[mk] = mv
                        else:
                            merged_metadata[f"{name.lower()}_{mk}"] = mv

                    for k in f.keys():
                        tensor = f.get_tensor(k)
                        new_key = k
                        
                        # Smart Prefixing: Prevents double prefixes like "first_stage_model.first_stage_model."
                        if prefix and not k.startswith(prefix):
                            new_key = f"{prefix}{k}"
                            
                        merged_sd[new_key] = tensor.contiguous()

                lines.append(f"✅ {name} integrated.")
                merged_any = True
            except Exception as e:
                lines.append(f"❌ Error with {name}: {str(e)}")

        if not merged_any:
            raise ValueError("No components selected for merging!")

        if clip_type_setting and clip_type_setting.strip():
            merged_metadata["clip_type"] = clip_type_setting.strip()
            merged_metadata["modelspec.architecture"] = clip_type_setting.strip()

        print(f"💾 Saving AIO: {out_path}")
        safetensors.torch.save_file(merged_sd, out_path, metadata=merged_metadata)

        final_size = format_size(os.path.getsize(out_path))
        lines.append(f"✅ Saved: {output_name}.safetensors ({final_size})")
        lines.append(f"⏱️ Time: {time.time() - start_time:.1f}s")

        return ("\n".join(lines),)

NODE_CLASS_MAPPINGS = {"StarnodesAIOSaver": StarnodesAIOSaver}
NODE_DISPLAY_NAME_MAPPINGS = {"StarnodesAIOSaver": "⭐ Starnodes AIO Saver"}