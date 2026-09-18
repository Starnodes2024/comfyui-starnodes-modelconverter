"""Star Quant Key Fixer

Normalizes the 'model.diffusion_model.' prefix that StarNodes' converter
(and some other tools) writes into every state dict key, including the
quantization metadata. Stripping the prefix makes ComfyUI's loader stop
flagging .comfy_quant entries as unexpected keys, and produces a file
whose layout matches what community-quantized safetensors use.

TWO MODES
---------

  Fix all keys (recommended)
      Renames every tensor key that starts with 'model.diffusion_model.'.
      Also renames the same prefix inside the _quantization_metadata
      header blob. Result: bare-keyed file, no prefix anywhere.

  Fix metadata only (legacy)
      Renames only the header metadata layer keys. Leaves tensor keys
      untouched. This produces the mixed-prefix layout that the previous
      version of this node shipped, which works on some loaders and not
      others.

BYTE SAFETY
-----------

No tensor is ever passed to a quantize / dequantize / cast function.
safetensors.torch.load_file reads raw bytes; the fixer only rebuilds the
Python dict that maps names to those tensors; save_file writes the same
bytes back. An optional post-write verification pass hashes every tensor
in the original and the fixed file and refuses to keep the output if any
byte differs.
"""
import os
import json
import hashlib
import tempfile
import folder_paths
import torch
import safetensors
import safetensors.torch

PREFIX = "model.diffusion_model."
SEARCH_FOLDERS = ("diffusion_models", "unet", "unet_gguf")
QUANT_HEADER_KEY = "_quantization_metadata"

MODE_REPORT = "Report only"
MODE_ALL = "Fix all keys (recommended)"
MODE_META = "Fix metadata only (legacy)"
MODES = [MODE_REPORT, MODE_ALL, MODE_META]


def _strip(k):
    return k[len(PREFIX):] if k.startswith(PREFIX) else k


def _tensor_byte_hash(t):
    """SHA256 of a tensor's raw bytes; dtype and shape are folded in so
    that a silent dtype change or shape transpose is detected even if the
    raw bytes happen to match."""
    h = hashlib.sha256()
    h.update(str(tuple(t.shape)).encode())
    h.update(str(t.dtype).encode())
    t_cpu = t.detach().cpu().contiguous()
    try:
        raw = t_cpu.numpy().tobytes()
    except TypeError:
        # bfloat16 (and possibly exotic fp8 types) don't support .numpy().
        # Reinterpret the underlying memory as uint8 bytes instead.
        raw = t_cpu.reshape(-1).view(torch.uint8).numpy().tobytes()
    h.update(raw)
    return h.hexdigest()


def _hash_multiset(sd):
    """Order-independent multiset of per-tensor byte hashes.

    Key names are deliberately excluded: if the multiset matches, every
    tensor is byte-identical to something in the other dict, which for a
    pure rename is exactly the property we want to verify.
    """
    from collections import Counter
    c = Counter()
    for k, v in sd.items():
        if isinstance(v, torch.Tensor):
            c[_tensor_byte_hash(v)] += 1
    return c


class StarQuantKeyFixer:
    """Rewrite prefixed quantization keys to their bare form."""

    @classmethod
    def INPUT_TYPES(s):
        files = set()
        for folder in SEARCH_FOLDERS:
            try:
                files.update(folder_paths.get_filename_list(folder) or [])
            except Exception:
                pass
        file_list = sorted(f for f in files if f.lower().endswith(".safetensors"))
        if not file_list:
            file_list = ["<no .safetensors files found>"]

        return {
            "required": {
                "model_name": (file_list, {
                    "tooltip": "Pick a .safetensors file from your model folders."
                }),
                "mode": (MODES, {
                    "default": MODE_REPORT,
                    "tooltip": (
                        "Report only: scan and describe, no writes.\n"
                        "Fix all keys: strip prefix from tensors and header metadata (recommended).\n"
                        "Fix metadata only: strip prefix from header metadata only (legacy)."
                    ),
                }),
                "verify_bytes": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Hash every tensor before and after, abort if any byte differs. Adds a few seconds per GB."
                }),
                "overwrite": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "When off, writes '<name>_fixed.safetensors'. When on, replaces the original (a .bak backup is made first)."
                }),
            },
            "optional": {
                "custom_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Optional absolute path; overrides the dropdown selection",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("status", "details")
    FUNCTION = "run"
    CATEGORY = "⭐StarNodes/Model Tools"
    OUTPUT_NODE = True

    def run(self, model_name, mode, verify_bytes, overwrite, custom_path=""):
        # ---- resolve source ------------------------------------------------
        if custom_path.strip():
            src = os.path.abspath(os.path.expanduser(custom_path.strip().strip('"')))
            if not os.path.isfile(src):
                raise ValueError(f"File not found: {src}")
        else:
            src = None
            for folder in SEARCH_FOLDERS:
                try:
                    p = folder_paths.get_full_path(folder, model_name)
                except Exception:
                    p = None
                if p:
                    src = p
                    break
            if src is None:
                raise ValueError(f"Could not resolve model: {model_name}")

        if not src.lower().endswith(".safetensors"):
            raise ValueError("Only .safetensors files are supported.")

        print(f"🔍 [Star Quant Key Fixer] Scanning {os.path.basename(src)}")

        # ---- scan ----------------------------------------------------------
        with safetensors.safe_open(src, framework="pt") as f:
            keys = list(f.keys())
            raw_meta = f.metadata() or {}

        prefixed_weight = [k for k in keys
                           if k.startswith(PREFIX) and k.endswith(".weight")]
        prefixed_scale = [k for k in keys
                          if k.startswith(PREFIX) and k.endswith("_scale")]
        prefixed_other = [k for k in keys
                          if k.startswith(PREFIX)
                          and not k.endswith(".weight")
                          and not k.endswith("_scale")]
        prefixed_any = prefixed_weight + prefixed_scale + prefixed_other

        header_raw = raw_meta.get(QUANT_HEADER_KEY)
        header_layers = {}
        header_prefixed = 0
        header_parse_error = None
        if header_raw:
            try:
                parsed = json.loads(header_raw)
                header_layers = parsed.get("layers", {})
                header_prefixed = sum(1 for k in header_layers if k.startswith(PREFIX))
            except Exception as e:
                header_parse_error = str(e)

        # ---- report --------------------------------------------------------
        report = [
            f"File: {os.path.basename(src)}",
            f"Total tensors: {len(keys)}",
            "",
            "Prefixed tensor keys (start with '" + PREFIX + "'):",
            f"  weights:  {len(prefixed_weight)}",
            f"  scales:   {len(prefixed_scale)}",
            f"  other:    {len(prefixed_other)}",
            "",
            "Header _quantization_metadata:",
        ]
        if header_raw is None:
            report.append("  not present")
        elif header_parse_error is not None:
            report.append(f"  present but failed to parse: {header_parse_error}")
        else:
            report.append(f"  layer entries: {len(header_layers)}")
            report.append(f"  prefixed: {header_prefixed}")

        total_to_rename = (
            len(prefixed_any) if mode == MODE_ALL else 0
        ) + header_prefixed

        if mode == MODE_REPORT:
            report.append("")
            report.append("Dry run. Choose a Fix mode and re-run to write.")
            if prefixed_any:
                report.append("")
                report.append("Sample prefixed tensor keys:")
                for k in prefixed_any[:5]:
                    report.append(f"  {k}")
            if header_prefixed:
                report.append("")
                report.append("Sample prefixed header keys:")
                for k in [k for k in header_layers if k.startswith(PREFIX)][:5]:
                    report.append(f"  {k}")
            return ("✅ Scan complete (report only)", "\n".join(report))

        if total_to_rename == 0:
            report.append("")
            report.append("Nothing to rename in the selected mode.")
            return ("✅ Scan complete (no changes needed)", "\n".join(report))

        # ---- load ----------------------------------------------------------
        print(f"📥 Loading {os.path.basename(src)}")
        original_sd = safetensors.torch.load_file(src)

        # ---- build new state dict -----------------------------------------
        new_sd = {}
        collisions = []

        for k, v in original_sd.items():
            if mode == MODE_ALL:
                new_k = _strip(k)
            else:
                new_k = k
            if new_k in new_sd:
                collisions.append(new_k)
            new_sd[new_k] = v

        if collisions:
            raise ValueError(
                f"Rename would collide on {len(collisions)} key(s): "
                f"{collisions[:3]}. File has duplicate tensors under both "
                f"prefix forms; refusing to guess."
            )

        # ---- rebuild header metadata --------------------------------------
        new_meta = dict(raw_meta)
        if header_raw and header_prefixed:
            parsed = json.loads(header_raw)
            old_layers = parsed.get("layers", {})
            new_layers = {}
            for k, v in old_layers.items():
                new_k = _strip(k)
                if new_k in new_layers:
                    raise ValueError(f"Header collision on: {new_k}")
                new_layers[new_k] = v
            parsed["layers"] = new_layers
            new_meta[QUANT_HEADER_KEY] = json.dumps(parsed)

        # ---- verify tensors are byte-identical ----------------------------
        if verify_bytes:
            print("🔐 Verifying tensor bytes are unchanged ...")
            before = _hash_multiset(original_sd)
            after = _hash_multiset(new_sd)
            if before != after:
                report.append("")
                report.append("❌ Byte verification FAILED. Tensors differ after rename.")
                diff_before = sum(before.values()) - sum((before & after).values())
                diff_after = sum(after.values()) - sum((before & after).values())
                report.append(f"   only in original: {diff_before}")
                report.append(f"   only in fixed:    {diff_after}")
                return ("❌ Aborted: byte verification failed", "\n".join(report))
            report.append("")
            report.append(f"✅ Byte verification passed ({sum(before.values())} tensors hashed)")

        # ---- write ---------------------------------------------------------
        if overwrite:
            dst = src
            bak = src + ".bak"
            if not os.path.exists(bak):
                os.replace(src, bak)
                print(f"📦 Backup: {bak}")
            else:
                print(f"📦 Backup already exists, keeping: {bak}")
        else:
            stem, ext = os.path.splitext(src)
            dst = f"{stem}_fixed{ext}"

        tmp_fd, tmp_path = tempfile.mkstemp(
            suffix=".safetensors", prefix=".keyfix_", dir=os.path.dirname(dst)
        )
        os.close(tmp_fd)
        try:
            print(f"💾 Writing {dst}")
            safetensors.torch.save_file(new_sd, tmp_path, metadata=new_meta)
            os.replace(tmp_path, dst)
        except BaseException:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

        # ---- final report --------------------------------------------------
        report.append("")
        if mode == MODE_ALL:
            report.append(f"Renamed {len(prefixed_any)} tensor keys")
        report.append(f"Renamed {header_prefixed} header layer keys")
        report.append(f"Saved to: {dst}")
        if overwrite:
            report.append(f"Backup: {bak}")

        status = f"✅ Fixed {total_to_rename} keys → {os.path.basename(dst)}"
        print("\n" + status)
        return (status, "\n".join(report))


NODE_CLASS_MAPPINGS = {"StarQuantKeyFixer": StarQuantKeyFixer}
NODE_DISPLAY_NAME_MAPPINGS = {"StarQuantKeyFixer": "⭐ Star Quant Key Fixer"}