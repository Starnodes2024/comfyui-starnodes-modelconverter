"""Star Model Layers Info - Analyze and report layer quantization information.

Supports:
  * .safetensors (native, FP8, INT8, INT8 ConvRot, NVFP4, MXFP8, W4A4, AWQ, ...)
  * .gguf       (all GGML quant types, plus custom Q8_CR / Q4_CR_W4A4 metadata
                 written by the ComfyUI-GGUF fork)

The safetensors path parses the authoritative per-tensor ``.comfy_quant``
JSON blobs that ComfyUI's quantization tooling writes next to each quantized
weight. It only falls back to the legacy global ``_quantization_metadata``
header or the ``.weight_scale`` heuristic when per-tensor metadata is absent.

The GGUF path reads the GGUF header directly via the ``gguf`` package, so the
report reflects on-disk storage (Q4_K, Q6_K, IQ*, F16, BF16, ...) rather than
a dequantized torch dtype. Custom ``comfy.gguf.quant.*`` fields are also read
so Q8_CR / Q4_CR_W4A4 tensors are labelled correctly.
"""
import os
import re
import json
import time
import torch
import folder_paths
import safetensors
import safetensors.torch
from collections import Counter, OrderedDict, defaultdict

# ---------------------------------------------------------------------------
# Lookup tables
# ---------------------------------------------------------------------------

DTYPE_NAMES = {
    torch.float32: "fp32",
    torch.float16: "fp16",
    torch.bfloat16: "bf16",
    torch.float8_e4m3fn: "fp8_e4m3fn",
    torch.float8_e5m2: "fp8_e5m2",
    torch.int8: "int8",
    torch.uint8: "uint8",
}

GGML_QTYPE_LABELS = {
    "F32": "GGUF_F32",
    "F16": "GGUF_F16",
    "BF16": "GGUF_BF16",
    "Q4_0": "GGUF_Q4_0",
    "Q4_1": "GGUF_Q4_1",
    "Q5_0": "GGUF_Q5_0",
    "Q5_1": "GGUF_Q5_1",
    "Q8_0": "GGUF_Q8_0",
    "Q2_K": "GGUF_Q2_K",
    "Q3_K": "GGUF_Q3_K",
    "Q4_K": "GGUF_Q4_K",
    "Q5_K": "GGUF_Q5_K",
    "Q6_K": "GGUF_Q6_K",
    "IQ1_S": "GGUF_IQ1_S",
    "IQ1_M": "GGUF_IQ1_M",
    "IQ2_XXS": "GGUF_IQ2_XXS",
    "IQ2_XS": "GGUF_IQ2_XS",
    "IQ2_S": "GGUF_IQ2_S",
    "IQ2_M": "GGUF_IQ2_M",
    "IQ3_XXS": "GGUF_IQ3_XXS",
    "IQ3_XS": "GGUF_IQ3_XS",
    "IQ3_S": "GGUF_IQ3_S",
    "IQ3_M": "GGUF_IQ3_M",
    "IQ4_NL": "GGUF_IQ4_NL",
    "IQ4_XS": "GGUF_IQ4_XS",
    "TQ1_0": "GGUF_TQ1_0",
    "TQ2_0": "GGUF_TQ2_0",
    "MXFP4_MOE": "GGUF_MXFP4_MOE",
    "I8": "GGUF_I8",
    "COPY": "GGUF_COPY",
}

# Folders to search for models, in order of preference.
MODEL_FOLDERS = ("diffusion_models", "unet_gguf", "unet")


def format_size(num_bytes):
    """Format bytes as human-readable size."""
    if num_bytes < 1024:
        return f"{num_bytes} bytes"
    elif num_bytes < 1024 ** 2:
        return f"{num_bytes / 1024:.2f} KB"
    elif num_bytes < 1024 ** 3:
        return f"{num_bytes / (1024 ** 2):.2f} MB"
    else:
        return f"{num_bytes / (1024 ** 3):.2f} GB"


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class StarModelLayersInfo:
    """Analyze diffusion model layers and report quantization information."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (s._list_model_files(), {
                    "tooltip": "Diffusion model file. Both .safetensors and .gguf are listed."
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
                    "placeholder": "Enter path to .safetensors or .gguf file",
                    "tooltip": "Full path to a model file. Only used when 'Use File Path' is enabled."
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
    CATEGORY = "⭐StarNodes/Model Tools"
    OUTPUT_NODE = True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @classmethod
    def _list_model_files(cls):
        names = set()
        for key in MODEL_FOLDERS:
            try:
                for n in folder_paths.get_filename_list(key) or []:
                    names.add(n)
            except Exception:
                continue
        return sorted(names)

    def _resolve_model_path(self, model_name):
        for key in MODEL_FOLDERS:
            try:
                p = folder_paths.get_full_path(key, model_name)
                if p:
                    return p
            except Exception:
                continue
        raise ValueError(
            f"Model not found in any of {MODEL_FOLDERS}: {model_name}"
        )

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def analyze(self, model_name, view_mode="Normal View",
                use_file_path=False, file_path="", save_profile=False):
        start_time = time.time()

        print("🔍 [Star Model Layers Info] Starting analysis...")

        # -- Resolve input path -----------------------------------------
        if use_file_path and file_path.strip():
            input_path = os.path.abspath(
                os.path.expanduser(file_path.strip().strip('"'))
            )
            if not os.path.isfile(input_path):
                raise ValueError(f"File not found: {input_path}")
        else:
            input_path = self._resolve_model_path(model_name)

        base_name = os.path.splitext(os.path.basename(input_path))[0]
        input_bytes = os.path.getsize(input_path)
        is_gguf = input_path.lower().endswith(".gguf")

        kind = "GGUF" if is_gguf else "safetensors"
        print(f"📦 Loading {kind} model: {os.path.basename(input_path)}")

        # -- Scan -------------------------------------------------------
        if is_gguf:
            layer_data, layer_stats, total_params, extra_meta = self._scan_gguf(input_path)
        else:
            layer_data, layer_stats, total_params, extra_meta = \
                self._scan_safetensors(input_path)

        # -- Build the layer listing ------------------------------------
        if view_mode == "Tree View":
            layer_lines = self._build_tree_view(layer_data)
        else:
            layer_lines = self._build_normal_view(layer_data)

        # -- Summary ----------------------------------------------------
        duration = time.time() - start_time

        summary_lines = [
            f"Model: {base_name}",
            f"File: {os.path.basename(input_path)}",
            f"Format: {kind}",
            f"Total size: {format_size(input_bytes)}",
            f"Total parameters: {total_params:,}",
            f"Total tensors: {len(layer_data)}",
        ]

        if is_gguf:
            arch = extra_meta.get("general.architecture")
            if arch:
                summary_lines.append(f"GGUF architecture: {arch}")
            gname = extra_meta.get("general.name")
            if gname:
                summary_lines.append(f"GGUF model name: {gname}")
            ftype = extra_meta.get("general.file_type")
            if ftype is not None:
                summary_lines.append(f"GGUF file_type: {ftype}")
            qver = extra_meta.get("general.quantization_version")
            if qver is not None:
                summary_lines.append(f"GGUF quantization_version: {qver}")
            stored = extra_meta.get("_stored_tensor_bytes")
            if stored is not None:
                summary_lines.append(
                    f"Sum of tensor sizes (on-disk): {format_size(stored)}"
                )

        summary_lines.extend([
            "",
            "Layer Type Distribution:",
        ])
        for layer_type, count in layer_stats.most_common():
            summary_lines.append(f"  - {layer_type}: {count} layers")

        summary_lines.extend([
            "",
            "=" * 120,
            "Layer Details:",
            "=" * 120,
        ])

        full_info = "\n".join(summary_lines + layer_lines)

        # -- Save report ------------------------------------------------
        output_dir = os.path.join(folder_paths.get_output_directory(), "modelinfo")
        os.makedirs(output_dir, exist_ok=True)
        view_suffix = "_tree" if view_mode == "Tree View" else "_normal"
        ext_suffix = "_gguf" if is_gguf else ""
        output_file = os.path.join(
            output_dir, f"{base_name}{ext_suffix}{view_suffix}.txt"
        )

        print(f"💾 Saving layer info to: {output_file}")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(full_info)

        # -- Optional profile save -------------------------------------
        profile_file = None
        if save_profile:
            profile_file = self._save_quantization_profile(
                layer_data, base_name, input_path, source_kind=kind
            )

        # -- Status -----------------------------------------------------
        status_lines = [
            f"✅ Model analysis complete ({view_mode}, {kind})",
            f"Model: {base_name}",
            f"Total tensors: {len(layer_data)}",
            f"Total parameters: {total_params:,}",
            f"File size: {format_size(input_bytes)}",
            f"Analysis time: {duration:.1f}s",
            f"Report saved to: {output_file}",
        ]
        if profile_file:
            status_lines.append(f"📋 Profile saved to: {profile_file}")

        status = "\n".join(status_lines)

        print("\n" + "=" * 60)
        print(status)
        print("=" * 60 + "\n")

        return (status, full_info)

    # ------------------------------------------------------------------
    # Safetensors scan
    # ------------------------------------------------------------------

    def _scan_safetensors(self, input_path):
        with safetensors.safe_open(input_path, framework="pt") as f:
            metadata = f.metadata()
            keys = f.keys()

        # Load tensors
        sd = safetensors.torch.load_file(input_path)

        # -- Per-tensor comfy_quant metadata ----------------------------
        # ComfyUI's quantization tooling writes an authoritative JSON blob
        # named "<layer>.comfy_quant" next to every custom-quantized weight.
        # This is where ConvRot / NVFP4 / MXFP8 / W4A4 configs actually
        # live. The legacy global header may be absent.
        per_tensor_configs = {}
        for key in list(sd.keys()):
            if not key.endswith(".comfy_quant"):
                continue
            base_key = key[:-len(".comfy_quant")]
            tensor = sd[key]
            try:
                if isinstance(tensor, torch.Tensor):
                    raw = tensor.detach().cpu()
                    if raw.dtype == torch.uint8:
                        data = bytes(raw.tolist())
                    elif raw.dtype == torch.int8:
                        data = bytes(raw.to(torch.uint8).tolist())
                    else:
                        data = bytes(raw.to(torch.uint8).tolist())
                else:
                    continue
                per_tensor_configs[base_key] = json.loads(data.decode("utf-8"))
            except Exception as e:
                print(f"⚠️ Warning: Failed to parse {key}: {e}")

        # -- Legacy global _quantization_metadata (fallback) ------------
        legacy_metadata = {}
        if metadata and "_quantization_metadata" in metadata:
            try:
                quant_data = json.loads(metadata["_quantization_metadata"])
                legacy_metadata = quant_data.get("layers", {})
            except Exception:
                pass

        # -- Iterate ----------------------------------------------------
        layer_data = []
        layer_stats = Counter()
        total_params = 0

        for key in sorted(keys):
            tensor = sd[key]

            dtype_name = DTYPE_NAMES.get(tensor.dtype, str(tensor.dtype))
            shape = list(tensor.shape)
            num_params = tensor.numel()
            total_params += num_params
            size_bytes = num_params * tensor.element_size()

            storage_format = dtype_name

            if key.endswith(".weight"):
                base_key = key[:-len(".weight")]

                # 1. Per-tensor comfy_quant config (authoritative)
                if base_key in per_tensor_configs:
                    storage_format = self._label_from_comfy_quant(
                        per_tensor_configs[base_key], layer_stats
                    )

                # 2. Legacy global metadata (fallback)
                elif base_key in legacy_metadata:
                    storage_format = self._label_from_legacy_meta(
                        legacy_metadata[base_key], layer_stats
                    )

                # 3. Scale-tensor heuristic (older FP8 / INT8 files)
                elif f"{key}_scale" in keys:
                    storage_format = f"{dtype_name} + scale"
                    layer_stats["scaled"] += 1

                # 4. Default dtype
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

            layer_data.append({
                "key": key,
                "shape": shape,
                "format": storage_format.upper(),
                "params": num_params,
                "size": size_bytes,
            })

        return layer_data, layer_stats, total_params, {}

    def _label_from_comfy_quant(self, cfg, layer_stats):
        """Build a human-readable label from a per-tensor comfy_quant dict."""
        fmt = cfg.get("format", "unknown")

        if fmt == "int8_tensorwise":
            if cfg.get("convrot"):
                gs = cfg.get("convrot_groupsize", 256)
                layer_stats["int8_convrot"] += 1
                return f"INT8_CONVROT (GS:{gs})"
            layer_stats["int8"] += 1
            return "INT8"

        if fmt == "convrot_w4a4":
            gs = cfg.get("convrot_groupsize", 256)
            qgs = cfg.get("quant_group_size", 64)
            layer_stats["int4_convrot"] += 1
            return f"INT4_CONVROT (GS:{gs} QGS:{qgs})"

        if fmt == "nvfp4":
            layer_stats["nvfp4"] += 1
            gs = cfg.get("group_size")
            return f"NVFP4 (GS:{gs})" if gs else "NVFP4"

        if fmt == "mxfp8":
            layer_stats["mxfp8"] += 1
            return "MXFP8"

        if fmt == "float8_e4m3fn":
            layer_stats["fp8"] += 1
            return "F8_E4M3"

        if fmt == "awq_w4a16":
            layer_stats["awq_w4a16"] += 1
            gs = cfg.get("group_size")
            return f"AWQ_W4A16 (GS:{gs})" if gs else "AWQ_W4A16"

        if fmt == "int4_cr":
            if cfg.get("backing") == "w4a4":
                layer_stats["int4_cr_w4a4"] += 1
                return "INT4_CR_W4A4"
            layer_stats["int4_cr"] += 1
            return "INT4_CR"

        # Unknown custom format: expose its name verbatim
        layer_stats[fmt] += 1
        return fmt.upper()

    def _label_from_legacy_meta(self, meta, layer_stats):
        """Build a label from the older global _quantization_metadata dict."""
        fmt = meta.get("format", "unknown")

        if fmt == "int8_tensorwise":
            if meta.get("convrot"):
                layer_stats["int8_convrot"] += 1
                return "INT8_CONVROT"
            layer_stats["int8"] += 1
            return "INT8"

        if fmt == "convrot_w4a4":
            layer_stats["int4_convrot"] += 1
            return "INT4_CONVROT"

        if fmt == "nvfp4":
            layer_stats["nvfp4"] += 1
            return "NVFP4"

        if fmt == "mxfp8":
            layer_stats["mxfp8"] += 1
            return "MXFP8"

        if fmt == "float8_e4m3fn":
            layer_stats["fp8"] += 1
            return "F8_E4M3"

        layer_stats[fmt] += 1
        return fmt.upper()

    # ------------------------------------------------------------------
    # GGUF scan
    # ------------------------------------------------------------------

    def _scan_gguf(self, input_path):
        try:
            import gguf
        except ImportError:
            raise ImportError(
                "The 'gguf' package is required to inspect GGUF files. "
                "Install it with: pip install gguf"
            )

        reader = gguf.GGUFReader(input_path)

        # -- Metadata ---------------------------------------------------
        extra_meta = {}
        for field_name in reader.fields:
            try:
                field = reader.get_field(field_name)
                if field is None or len(field.types) != 1:
                    continue
                t = field.types[0]
                if t == gguf.GGUFValueType.STRING:
                    extra_meta[field_name] = str(field.parts[field.data[-1]], "utf-8")
                elif t in (
                    gguf.GGUFValueType.INT32, gguf.GGUFValueType.UINT32,
                    gguf.GGUFValueType.INT64, gguf.GGUFValueType.UINT64,
                ):
                    extra_meta[field_name] = int(field.parts[field.data[-1]])
                elif t in (gguf.GGUFValueType.FLOAT32, gguf.GGUFValueType.FLOAT64):
                    extra_meta[field_name] = float(field.parts[field.data[-1]])
                elif t == gguf.GGUFValueType.BOOL:
                    extra_meta[field_name] = bool(field.parts[field.data[-1]])
            except Exception:
                continue

        # -- Custom comfy.gguf.quant.* metadata (Q8_CR / Q4_CR_W4A4) -----
        custom_quant_configs = {}
        for field_name in reader.fields:
            if not field_name.startswith("comfy.gguf.quant."):
                continue
            key = field_name[len("comfy.gguf.quant."):]
            try:
                field = reader.get_field(field_name)
                custom_quant_configs[key] = json.loads(
                    str(field.parts[field.data[-1]], "utf-8")
                )
            except Exception:
                continue

        # -- Tensors ----------------------------------------------------
        layer_data = []
        layer_stats = Counter()
        total_params = 0
        total_stored_bytes = 0

        for tensor in reader.tensors:
            name = tensor.name
            qtype = tensor.tensor_type
            qtype_name = getattr(qtype, "name", str(qtype))

            # GGML stores ne[] with ne[0] as fastest-varying, i.e. reversed
            # relative to torch shape. ComfyUI's loader reverses it back.
            ggml_shape = tuple(int(v) for v in tensor.shape)
            torch_shape = tuple(reversed(ggml_shape))

            num_params = 1
            for d in torch_shape:
                num_params *= d

            size_bytes = int(tensor.n_bytes)
            total_stored_bytes += size_bytes
            total_params += num_params

            display = self._gguf_display_label(
                name, qtype_name, custom_quant_configs
            )
            layer_stats[display] += 1

            layer_data.append({
                "key": name,
                "shape": list(torch_shape),
                "format": display,
                "params": num_params,
                "size": size_bytes,
                "qtype": qtype_name,
            })

        extra_meta["_stored_tensor_bytes"] = total_stored_bytes
        return layer_data, layer_stats, total_params, extra_meta

    def _gguf_display_label(self, name, qtype_name, custom_quant_configs):
        """Best-effort label describing how a GGUF tensor is stored."""
        if name in custom_quant_configs:
            cfg = custom_quant_configs[name]
            fmt = cfg.get("format")
            if fmt == "int8_tensorwise":
                return "INT8_CONVROT (Q8_CR)" if cfg.get("convrot") else "INT8"
            if fmt == "int4_cr":
                if cfg.get("backing") == "w4a4":
                    return "INT4_CR_W4A4 (Q4_CR)"
                return "INT4_CR"
            if fmt:
                return f"CUSTOM:{fmt.upper()}"

        if name.endswith("_scale") and name[:-len("_scale")] in custom_quant_configs:
            cfg = custom_quant_configs[name[:-len("_scale")]]
            fmt = (cfg.get("format") or "").upper()
            return f"SCALE_{fmt}" if fmt else "SCALE"

        return GGML_QTYPE_LABELS.get(qtype_name, f"GGUF_{qtype_name}")

    # ------------------------------------------------------------------
    # Views
    # ------------------------------------------------------------------

    def _build_normal_view(self, layer_data):
        lines = []
        for layer in layer_data:
            line = (
                f"{layer['key']:<80} | Shape: {str(layer['shape']):<20} | "
                f"Format: {layer['format']:<28} | Params: {layer['params']:>12,} | "
                f"Size: {format_size(layer['size'])}"
            )
            lines.append(line)
        return lines

    def _build_tree_view(self, layer_data):
        tree = defaultdict(list)
        for layer in layer_data:
            parts = layer['key'].split('.')
            if len(parts) > 1:
                tree[parts[0]].append(layer)
            else:
                tree['_root'].append(layer)

        lines = []
        for prefix in sorted(tree.keys()):
            layers = tree[prefix]
            if not layers:
                continue

            grouped = self._group_consecutive_layers(layers)

            total_size = sum(l['size'] for l in layers)
            formats = sorted(set(l['format'] for l in layers))
            formats_str = ", ".join(formats)

            if prefix == '_root':
                lines.append(f"├── Root ({format_size(total_size)} | {formats_str})")
            else:
                lines.append(f"├── {prefix} ({format_size(total_size)} | {formats_str})")

            for group in grouped:
                if group['is_range']:
                    total_group_size = sum(l['size'] for l in group['layers'])
                    group_formats = sorted(set(l['format'] for l in group['layers']))
                    group_formats_str = ", ".join(group_formats)
                    lines.append(
                        f"│   ├── [{group['start']}-{group['end']}] "
                        f"({format_size(total_group_size)} | {group_formats_str})"
                    )

                    subcomponents = defaultdict(list)
                    for layer in group['layers']:
                        parts = layer['key'].split('.')
                        if len(parts) > 2:
                            component = '.'.join(parts[2:])
                            subcomponents[component].append(layer)

                    for comp_name in sorted(subcomponents.keys()):
                        comp_layers = subcomponents[comp_name]
                        comp_size = sum(l['size'] for l in comp_layers)
                        comp_formats = sorted(set(l['format'] for l in comp_layers))
                        comp_formats_str = ", ".join(comp_formats)
                        lines.append(
                            f"│   │   ├── {comp_name} "
                            f"({format_size(comp_size)} | {comp_formats_str})"
                        )
                else:
                    layer = group['layers'][0]
                    lines.append(
                        f"│   ├── {layer['key']} "
                        f"({format_size(layer['size'])} | {layer['format']})"
                    )

        return lines

    def _group_consecutive_layers(self, layers):
        numbered = []
        unnumbered = []

        for layer in layers:
            match = re.search(r'\.(\d+)\.', layer['key'])
            if match:
                num = int(match.group(1))
                numbered.append((num, layer))
            else:
                unnumbered.append(layer)

        numbered.sort(key=lambda x: x[0])

        groups = []
        if numbered:
            current_group = [numbered[0]]

            for i in range(1, len(numbered)):
                if numbered[i][0] == current_group[-1][0] + 1:
                    current_group.append(numbered[i])
                else:
                    if len(current_group) >= 3:
                        groups.append({
                            'is_range': True,
                            'start': current_group[0][0],
                            'end': current_group[-1][0],
                            'layers': [l for _, l in current_group],
                        })
                    else:
                        for _, layer in current_group:
                            groups.append({
                                'is_range': False,
                                'layers': [layer],
                            })
                    current_group = [numbered[i]]

            if len(current_group) >= 3:
                groups.append({
                    'is_range': True,
                    'start': current_group[0][0],
                    'end': current_group[-1][0],
                    'layers': [l for _, l in current_group],
                })
            else:
                for _, layer in current_group:
                    groups.append({
                        'is_range': False,
                        'layers': [layer],
                    })

        for layer in unnumbered:
            groups.append({
                'is_range': False,
                'layers': [layer],
            })

        return groups

    # ------------------------------------------------------------------
    # Profile save
    # ------------------------------------------------------------------

    def _save_quantization_profile(self, layer_data, model_name, model_path,
                                   source_kind="safetensors"):
        from datetime import datetime

        profiles_dir = os.path.join(os.path.dirname(__file__), "profiles")
        os.makedirs(profiles_dir, exist_ok=True)

        profile = {
            "__metadata__": {
                "original_model_name": model_name,
                "original_model_path": model_path,
                "source_format": source_kind,
                "timestamp": datetime.now().isoformat(),
                "total_layers": len(layer_data),
                "created_by": "Star Model Layers Info",
            },
            "layers": {},
        }

        for layer in layer_data:
            profile["layers"][layer["key"]] = layer["format"]

        profile_file = os.path.join(profiles_dir, f"{model_name}.json")
        print(f"📋 Saving quantization profile to: {profile_file}")

        with open(profile_file, "w", encoding="utf-8") as f:
            json.dump(profile, f, indent=2)

        return profile_file


NODE_CLASS_MAPPINGS = {"StarModelLayersInfo": StarModelLayersInfo}
NODE_DISPLAY_NAME_MAPPINGS = {"StarModelLayersInfo": "⭐ Star Model Layers Info"}