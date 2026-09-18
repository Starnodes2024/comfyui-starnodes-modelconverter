import os
import re
import json
import time
import glob
import math
import inspect
import subprocess
import tempfile
import shutil

import torch
import folder_paths
import safetensors
import safetensors.torch
import comfy.utils

from collections import Counter, OrderedDict

try:
    import gguf
    GGUF_AVAILABLE = True
except ImportError:
    gguf = None
    GGUF_AVAILABLE = False
    print("⚠️ [Star Ultimate Model Converter] gguf package not found. GGUF target formats will not be available (pip install gguf).")

try:
    import comfy_kitchen as ck
    from comfy_kitchen.registry import registry as ck_registry
    from comfy_kitchen.tensor import (
        TensorCoreConvRotW4A4Layout,
        TensorCoreMXFP8Layout,
        TensorCoreNVFP4Layout,
        TensorWiseINT8Layout,
    )
    KITCHEN_AVAILABLE = True
except ImportError:
    ck = None
    ck_registry = None
    TensorCoreConvRotW4A4Layout = None
    TensorCoreMXFP8Layout = None
    TensorCoreNVFP4Layout = None
    TensorWiseINT8Layout = None
    KITCHEN_AVAILABLE = False
    print("⚠️ [Star Ultimate Model Converter] comfy-kitchen not found.")

W4A8_LAYOUT = None
try:
    from comfy_kitchen.tensor import AsymW4A8Int8Layout as W4A8_LAYOUT
except ImportError:
    try:
        from comfy_kitchen.tensor.w4a8_int8 import AsymW4A8Int8Layout as W4A8_LAYOUT
    except ImportError:
        W4A8_LAYOUT = None

W4A8_AVAILABLE = W4A8_LAYOUT is not None

AWQ_W4A16_LAYOUT = None
try:
    from comfy_kitchen.tensor import TensorCoreAWQW4A16Layout as AWQ_W4A16_LAYOUT
except ImportError:
    try:
        from comfy_kitchen.tensor.awq_w4a16 import TensorCoreAWQW4A16Layout as AWQ_W4A16_LAYOUT
    except ImportError:
        AWQ_W4A16_LAYOUT = None

AWQ_W4A16_AVAILABLE = AWQ_W4A16_LAYOUT is not None

NODE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_JSON = os.path.join(NODE_DIR, "models.json")

EXTENDED_METADATA_KEYS = ["config", "license", "encrypted_wandb_properties"]
AIO_MODEL_PREFIX = "model.diffusion_model."

TARGET_FORMATS = [
    "nvfp4",
    "fp8",
    "mxfp8",
    "int8",
    "int8_convrot",
    "int4_convrot",
    "int8_convrot_pruned",
    "int4_convrot_pruned",
    "w4a8_convrot",
    "w4a8_convrot_pruned",
    "awq_w4a16",
    "w4a4",
    "minimax_h3_native_mix",
    "svdquant_w4a4",
    "fp16",
    "fp32",

    # GGUF legacy targets
    "gguf_f16",
    "gguf_bf16",
    "gguf_q8_0",
    "gguf_q5_1",
    "gguf_q5_0",
    "gguf_q4_1",
    "gguf_q4_0",

    # GGUF K-quant targets
    "gguf_q3_k_s",
    "gguf_q3_k_m",
    "gguf_q4_k_s",
    "gguf_q4_k_m",
    "gguf_q5_k_s",
    "gguf_q5_k_m",
    "gguf_q6_k",
]

GGUF_LEGACY_QUANT_MAP = {
    "gguf_f16": None,
    "gguf_bf16": None,
    "gguf_q8_0": "Q8_0",
    "gguf_q5_1": "Q5_1",
    "gguf_q5_0": "Q5_0",
    "gguf_q4_1": "Q4_1",
    "gguf_q4_0": "Q4_0",
}

GGUF_KQUANT_MAP = {
    "gguf_q3_k_s": "q3_k_s",
    "gguf_q3_k_m": "q3_k_m",
    "gguf_q4_k_s": "q4_k_s",
    "gguf_q4_k_m": "q4_k_m",
    "gguf_q5_k_s": "q5_k_s",
    "gguf_q5_k_m": "q5_k_m",
    "gguf_q6_k": "q6_k",
}

GGUF_TARGET_FORMATS = set(GGUF_LEGACY_QUANT_MAP) | set(GGUF_KQUANT_MAP)
GGUF_QUANTIZATION_THRESHOLD = 1024

GGUF_IMG_ARCH_LIST = {
    "flux", "sd1", "sdxl", "sd3", "aura", "hidream", "cosmos", "ltxv",
    "ltxv_upscaler", "hyvid", "wan", "lumina2", "qwen_image", "ideogram",
    "krea2", "minimax_h3", "minimax_h3_vae", "minimax_music3",
}

GGUF_TXT_ARCH_LIST = {
    "t5", "t5encoder", "llama", "qwen2vl", "qwen3", "qwen3vl", "qwen35",
    "gemma3", "gemma4", "minimax_music3",
}

MODEL_TYPE_TO_GGUF_ARCH = {
    "Chroma": "flux",
    "Flux1 / Flux2": "flux",
    "Flux2 Tight (W4A8 Full)": "flux",
    "Ideogram-4": "ideogram",
    "Krea-2": "krea2",
    "LTX-Video (All Versions)": "ltxv",
    "LTX-2.5": "ltxv",
    "Qwen-Image": "qwen_image",
    "Qwen-Image W4A8": "qwen_image",
    "SDXL (Not NVFP4)": "sdxl",
    "Wan (All Versions)": "wan",
    "minimax_h3": "minimax_h3",
    "minimax_h3_ref_nvfp4_fp8": "minimax_h3",
    "minimax_h3_ref_nvfp4_int8convrot": "minimax_h3",
    "minimax_h3_int4_tensorwise_experimental": "minimax_h3",
    "minimax_h3_int8convrot_int4fc2": "minimax_h3",
    "minimax_h3_int8convrot_int4mlp": "minimax_h3",
    "minimax_h3_vae": "minimax_h3_vae",
}

GGUF_MAX_TENSOR_NAME_LENGTH = 127
GGUF_MAX_TENSOR_DIMS = 4

MINIMAX_H3_BOUNDARY_BLOCKS = {0, 1, 47, 48, 49}
MINIMAX_H3_NVFP4_GROUPSIZE = 16

CONVROT_GROUPSIZE = 256
INT4_QUANT_GROUPSIZE = 64

SVDQ_OVERSAMPLE = 16
SVDQ_NITER = 2

AWQ_PRESCALE_ALPHA = 0.5

W4A8_GROUP_SIZE = 16
W4A8_CONVROT_GROUPSIZE = 256
W4A8_FORMAT_NAME = "asym_w4a8_int8"
W4A8_SYMMETRIC = True
W4A8_CODEBOOK = True

AWQ_W4A16_GROUP_SIZE = 64
LEARNED_ROUNDING_DEFAULT_ITERS = 200
LEARNED_ROUNDING_DEFAULT_LR = 0.01
LEARNED_ROUNDING_DEFAULT_TOPK_RATIO = 0.25
AWQ_W4A16_FORMAT_NAME = "awq_w4a16"

PRECISION_RE = re.compile(
    r"[-_.]("
    r"fp32|fp16|bf16|mxfp8|"
    r"fp8(?:_e[45]m[23](?:fn)?)(?:_scaled)?(?:_fast)?|"
    r"int[48](?:_convrot)?(?:_tensorwise)?(?:_pruned)?|"
    r"w4a8(?:_convrot)?(?:_pruned)?|"
    r"awq_w4a16|"
    r"nvfp4|svdquant_w4a4|w4a4"
    r")(?=[-_.]|$)",
    re.IGNORECASE,
)

FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)
E8M0_DTYPE = getattr(torch, "float8_e8m0fnu", None)

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
        profile.get("blacklist", default.get("blacklist", [])),
        profile.get("fp8_layers", default.get("fp8_layers", [])),
        profile.get("preserve_extended_metadata", default.get("preserve_extended_metadata", False)),
        profile.get("force_int8_layers", default.get("force_int8_layers", [])),
        profile.get("awq_prescale_layers", default.get("awq_prescale_layers", [])),
        profile.get("pruned_extra_blacklist", default.get("pruned_extra_blacklist", [])),
        profile.get("int8_convrot_layers", default.get("int8_convrot_layers", [])),
        profile.get("int4_convrot_layers", default.get("int4_convrot_layers", [])),
        profile.get("keep_fp32", default.get("keep_fp32", [])),
        profile.get("keep_fp16", default.get("keep_fp16", [])),
        profile.get("w4a8_blacklist", default.get("w4a8_blacklist", [])),
        profile.get("w4a8_keep_fp32", default.get("w4a8_keep_fp32", [])),
        profile.get("w4a8_keep_fp16", default.get("w4a8_keep_fp16", [])),
        profile.get("awq_w4a16_layers", default.get("awq_w4a16_layers", [])),
    )


def blacklisted_dtype(k: str, keep_fp32, keep_fp16) -> torch.dtype:
    if keep_fp32 and any(name in k for name in keep_fp32):
        return torch.float32
    if keep_fp16 and any(name in k for name in keep_fp16):
        return torch.float16
    return torch.bfloat16


def resolve_input(mode, diffusion_model, checkpoint, text_encoder, custom_path, vae="None"):
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

    elif mode == "VAE":
        if not vae or vae == "None":
            raise ValueError("No VAE selected. Please choose a model from the dropdown.")

        path = folder_paths.get_full_path("vae", vae)
        if not path:
            raise ValueError(f"VAE not found: {vae}")

        return [path], os.path.dirname(path), os.path.splitext(os.path.basename(path))[0]

    elif mode == "Text-Encoder":
        if not text_encoder or text_encoder == "None":
            raise ValueError("No Text-Encoder selected. Please choose a model from the dropdown.")

        path = folder_paths.get_full_path("text_encoders", text_encoder)
        if not path:
            path = folder_paths.get_full_path("clip", text_encoder)
        if not path:
            raise ValueError(f"Text-Encoder not found: {text_encoder}")

        return [path], os.path.dirname(path), os.path.splitext(os.path.basename(path))[0]

    raise ValueError(f"Unknown mode: {mode}")


def diffusion_models_dir():
    paths = folder_paths.get_folder_paths("diffusion_models")
    for p in paths:
        if os.path.basename(os.path.normpath(p)) == "diffusion_models":
            return p
    return paths[0]


def load_aio_model(checkpoint_name):
    ckpt_path = folder_paths.get_full_path("checkpoints", checkpoint_name)
    if not ckpt_path:
        raise ValueError(f"Checkpoint not found: {checkpoint_name}")

    full_sd = comfy.utils.load_torch_file(ckpt_path, safe_load=True)
    sd = {
        k[len(AIO_MODEL_PREFIX):]: v
        for k, v in full_sd.items()
        if k.startswith(AIO_MODEL_PREFIX)
    }

    if not sd:
        raise ValueError(
            f"No '{AIO_MODEL_PREFIX}' keys found in {os.path.basename(ckpt_path)}. "
            "Is this an all-in-one checkpoint?"
        )

    return sd, ckpt_path


def pick_mxfp8_backend(device):
    if not KITCHEN_AVAILABLE or TensorCoreMXFP8Layout is None:
        raise RuntimeError("MXFP8 requires comfy-kitchen TensorCoreMXFP8Layout.")

    probe = torch.randn(32, 32, device=device, dtype=torch.float32)

    try:
        TensorCoreMXFP8Layout.quantize(probe)
        return None
    except Exception as e:
        print(f"⚠️ MXFP8 default backend failed ({e}). Trying fallback backends...")

    for backend in ("triton", "eager"):
        try:
            with ck_registry.use_backend(backend):
                TensorCoreMXFP8Layout.quantize(probe)
            print(f"✅ MXFP8: using '{backend}' backend")
            return backend
        except Exception:
            continue

    raise RuntimeError(
        "MXFP8 quantization is not supported by any comfy_kitchen backend in this environment. "
        "Try updating comfy-kitchen and PyTorch."
    )


_HADAMARD_CACHE = {}


def build_hadamard(size: int, device="cpu", dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Normalized regular orthogonal Hadamard matrix (Kronecker/Theorem-3.3
    construction for power-of-4 sizes, scipy fallback otherwise). Ported
    verbatim from silveroxides/convert_to_quant's utils/convrot.py.
    """
    cache_key = (size, str(device), dtype)
    if cache_key in _HADAMARD_CACHE:
        return _HADAMARD_CACHE[cache_key]
    if size < 4 or (size & (size - 1)) != 0:
        raise ValueError(f"Hadamard size must be a power of 2, got {size}")

    is_power_of_4 = (math.log(size, 4) % 1 == 0)
    if not is_power_of_4:
        from scipy.linalg import hadamard as scipy_hadamard
        h_np = scipy_hadamard(size)
        h = torch.from_numpy(h_np).to(device=device, dtype=dtype) / (size ** 0.5)
        _HADAMARD_CACHE[cache_key] = h
        return h

    h4 = torch.tensor(
        [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
        dtype=dtype, device=device,
    )
    h = h4
    current_size = 4
    while current_size < size:
        h = torch.kron(h, h4)
        current_size *= 4
    h = h / (size ** 0.5)
    _HADAMARD_CACHE[cache_key] = h
    return h


def rotate_weight(weight: torch.Tensor, h: torch.Tensor, group_size: int) -> torch.Tensor:
    """W_rot = W @ H_block^T, applied group-wise along in_features. Ported verbatim from convrot.py."""
    out_f, in_f = weight.shape
    if in_f % group_size != 0:
        raise ValueError(f"in_features {in_f} not divisible by group_size {group_size}")
    n_groups = in_f // group_size
    w_grouped = weight.view(out_f, n_groups, group_size)
    h_t = h.T.to(dtype=weight.dtype, device=weight.device)
    return torch.matmul(w_grouped, h_t).reshape(out_f, in_f)


def resolve_int8_convrot_group_size(in_features: int, requested_group_size: int):
    """256 -> 64 -> None fallback, matching convrot.py exactly."""
    if in_features % requested_group_size == 0:
        return requested_group_size
    if requested_group_size == 256 and in_features % 64 == 0:
        return 64
    return None


def quantize_int8_rowwise_convrot(
    weight_f32: torch.Tensor,
    convrot_groupsize: int = CONVROT_GROUPSIZE,
    use_learned_rounding: bool = False,
    learned_rounding_iters: int = LEARNED_ROUNDING_DEFAULT_ITERS,
    learned_rounding_lr: float = LEARNED_ROUNDING_DEFAULT_LR,
    learned_rounding_topk_ratio: float = LEARNED_ROUNDING_DEFAULT_TOPK_RATIO,
):
    """
    Row-wise INT8 quantization with optional group-wise ConvRot rotation and
    optional learned-rounding refinement. Storage convention (weight.shape =
    (out_features, in_features)) and the exact metadata keys below were
    confirmed against silveroxides/convert_to_quant's own
    test_int8_convrot_serialization.py:
      qdata: int8, shape (out_features, in_features)
      scale: float32, shape (out_features, 1)
      quant_conf: {"format": "int8_tensorwise", "per_row": True,
                   "convrot": bool, "convrot_groupsize": int (only if rotated)}
    """
    out_f, in_f = weight_f32.shape
    resolved_gs = resolve_int8_convrot_group_size(in_f, convrot_groupsize)

    w = weight_f32
    convrot_used = False
    if resolved_gs is not None:
        try:
            h = build_hadamard(resolved_gs, device=weight_f32.device, dtype=torch.float32)
            w = rotate_weight(weight_f32, h, resolved_gs)
            convrot_used = True
        except Exception as e:
            print(f"⚠️ ConvRot rotation failed, falling back to unrotated INT8: {e}")
            w = weight_f32
            convrot_used = False

    scale = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-9) / 127.0

    if use_learned_rounding:
        qdata = learned_round_refine(
            w, 1.0 / scale, torch.int8,
            num_iter=learned_rounding_iters,
            lr=learned_rounding_lr,
            topk_ratio=learned_rounding_topk_ratio,
        )
    else:
        qdata = (w / scale).round().clamp(-127, 127).to(torch.int8)

    quant_conf = {"format": "int8_tensorwise", "per_row": True}
    if convrot_used:
        quant_conf["convrot"] = True
        quant_conf["convrot_groupsize"] = resolved_gs

    return qdata, scale.to(torch.float32), quant_conf


def learned_round_refine(
    weight_f32: torch.Tensor,
    scale: torch.Tensor,
    target_dtype: torch.dtype,
    num_iter: int = LEARNED_ROUNDING_DEFAULT_ITERS,
    lr: float = LEARNED_ROUNDING_DEFAULT_LR,
    topk_ratio: float = LEARNED_ROUNDING_DEFAULT_TOPK_RATIO,
):
    """
    Refine a naive round-to-nearest quantization by gradient descent on a
    continuous delta added to the rounded value, minimizing reconstruction
    error projected onto the weight's own top-k SVD subspace.

    Ported (leaner: AdamW only, no optimizer/scheduler variants, no
    early-stop heuristics) from silveroxides/convert_to_quant's
    learned_rounding.py. Calibration-free: U_k/Vh_k come from the weight's
    own SVD, no activation data. Output format is unchanged -- this only
    changes which grid value each weight rounds to, so it stays fully
    compatible with the existing dequantize/loader pipeline.

    weight_f32: the original float32 weight, shape (M, N)
    scale: per-tensor or per-row scale already used to produce the naive
           quantization (weight_f32 / scale must land in target_dtype's range)
    target_dtype: e.g. torch.float8_e4m3fn or torch.int8
    Returns: refined weight tensor already cast to target_dtype
    """
    device = weight_f32.device
    m, n = weight_f32.shape
    k = max(1, min(int(min(m, n) * topk_ratio), min(m, n)))

    with torch.no_grad():
        try:
            u, s, v = torch.svd_lowrank(weight_f32, q=min(k + 8, min(m, n)), niter=2)
            u_k, vh_k = u[:, :k].contiguous(), v[:, :k].transpose(-2, -1).contiguous()
        except Exception:
            # Degenerate (all-zero / non-finite) tensor: no SVD subspace to
            # project onto, so just return the naive rounding untouched.
            return (weight_f32 * scale).round().clamp(
                torch.finfo(target_dtype).min if target_dtype.is_floating_point else -128,
                torch.finfo(target_dtype).max if target_dtype.is_floating_point else 127,
            ).to(target_dtype)

        w_rounded = (weight_f32 * scale).round().to(target_dtype).to(torch.float32)

    delta = torch.zeros_like(w_rounded, requires_grad=True)
    optimizer = torch.optim.AdamW([delta], lr=lr)

    best_loss = float("inf")
    best_delta = torch.zeros_like(w_rounded)

    for _ in range(max(1, num_iter)):
        optimizer.zero_grad()
        dequant = (w_rounded + delta) / scale
        error = dequant - weight_f32
        projected_error = u_k.T @ error @ vh_k.T
        loss = torch.linalg.norm(projected_error)
        if not torch.isfinite(loss):
            break
        loss.backward()
        optimizer.step()
        loss_val = loss.item()
        if loss_val < best_loss:
            best_loss = loss_val
            best_delta = delta.detach().clone()

    with torch.no_grad():
        refined = (w_rounded + best_delta)
        clamp_min = torch.finfo(target_dtype).min if target_dtype.is_floating_point else -128
        clamp_max = torch.finfo(target_dtype).max if target_dtype.is_floating_point else 127
        return refined.clamp(clamp_min, clamp_max).to(target_dtype)


def compute_awq_prescale(weight: torch.Tensor, alpha: float = AWQ_PRESCALE_ALPHA) -> torch.Tensor:
    w = weight.float()
    channel_mag = w.abs().mean(dim=0).clamp(min=1e-5)
    s = channel_mag.pow(alpha)
    s = (s / s.mean()).clamp(min=1e-4)
    return s


def pack_int4_nibbles(q: torch.Tensor) -> torch.Tensor:
    if q.shape[1] % 2 != 0:
        raise ValueError(f"in_features {q.shape[1]} is odd, cannot pack 2-per-byte cleanly.")

    low = (q[:, 0::2] & 0x0F).to(torch.uint8)
    high = (q[:, 1::2] & 0x0F).to(torch.uint8)
    packed = low | (high << 4)
    return packed


def block_index_from_key(k: str):
    if not k.startswith("blocks."):
        return None

    parts = k.split(".", 2)
    try:
        return int(parts[1])
    except (IndexError, ValueError):
        return None


def svd_lowrank(weight: torch.Tensor, rank: int, oversample: int = SVDQ_OVERSAMPLE, niter: int = SVDQ_NITER):
    w = weight.float()
    min_dim = min(w.shape)
    rank = max(1, min(int(rank), min_dim))
    q = min(rank + max(0, int(oversample)), min_dim)

    if min_dim <= max(q, 32):
        u, s, vh = torch.linalg.svd(w, full_matrices=False)
        u_r, s_r, vh_r = u[:, :rank], s[:rank], vh[:rank, :]
    else:
        u, s, v = torch.svd_lowrank(w, q=q, niter=max(0, int(niter)))
        u_r, s_r, vh_r = u[:, :rank], s[:rank], v[:, :rank].transpose(-2, -1)

    return (u_r * s_r.unsqueeze(0)).contiguous(), vh_r.contiguous()


def svdquant_split(weight: torch.Tensor, rank: int, groupsize: int, refine_iters: int):
    if TensorCoreConvRotW4A4Layout is None:
        raise RuntimeError("SVDQuant W4A4 requires comfy-kitchen TensorCoreConvRotW4A4Layout.")

    w = weight.float()
    w_norm = torch.linalg.matrix_norm(w).item()
    if not math.isfinite(w_norm) or w_norm == 0.0:
        return None

    layout = TensorCoreConvRotW4A4Layout
    qw = torch.zeros((), device=w.device, dtype=torch.float32)

    best = None
    best_err = float("inf")

    for _ in range(max(1, refine_iters)):
        target = w - qw
        l1, l2 = svd_lowrank(target, rank, oversample=SVDQ_OVERSAMPLE, niter=SVDQ_NITER)

        l1 = l1.to(torch.bfloat16)
        l2 = l2.to(torch.bfloat16)

        lw = l1.float() @ l2.float()
        residual = (w - lw).to(torch.bfloat16)

        qdata, params = layout.quantize(residual.float().contiguous(), convrot_groupsize=groupsize)
        qw = layout.dequantize(qdata, params).float()
        del qdata, params

        left = w - (lw + qw)
        err = (torch.linalg.matrix_norm(left) / w_norm).item()
        del lw, left

        if not math.isfinite(err):
            break

        if best is not None and err >= best_err - 1e-6:
            break

        best_err, best = err, (residual, l1, l2)

    return best


def build_output_path(out_dir, base_name, target_format):
    stem = PRECISION_RE.sub("", base_name).rstrip("-_.")
    return os.path.join(out_dir, f"{stem}-{target_format.strip()}.safetensors")


def load_input(files):
    sd = {}

    for i, fp in enumerate(files):
        if len(files) > 1:
            print(f"📦 Loading shard {i + 1}/{len(files)}: {os.path.basename(fp)}")

        part = comfy.utils.load_torch_file(fp, safe_load=True)
        for k in part:
            if k in sd:
                print(f"⚠️ Duplicate key '{k}' in {os.path.basename(fp)}, overwriting")
        sd.update(part)

    with safetensors.safe_open(files[0], framework="pt") as f:
        orig_meta = f.metadata()

    return sd, orig_meta


def assign_quantized_tensor(new_sd, key, tensor):
    if E8M0_DTYPE is not None and tensor.dtype == E8M0_DTYPE:
        new_sd[key] = tensor.view(torch.uint8).cpu()
    elif tensor.dtype in FP8_DTYPES:
        new_sd[key] = tensor.view(torch.uint8).cpu().view(tensor.dtype)
    else:
        new_sd[key] = tensor.cpu()


def store_quantized_weight(new_sd, original_weight_key, tensors):
    for suffix, tensor in tensors.items():
        if suffix == ".comfy_quant":
            continue

        if not suffix:
            out_key = original_weight_key
        elif suffix.startswith(".") or suffix.startswith("_"):
            out_key = original_weight_key + suffix
        else:
            out_key = f"{original_weight_key}.{suffix}"

        assign_quantized_tensor(new_sd, out_key, tensor)


def store_w4a8_quantized_weight(new_sd, original_weight_key, tensors):
    seen_suffixes = list(tensors.keys())
    mapped = {}

    for suffix, tensor in tensors.items():
        if suffix == ".comfy_quant":
            continue

        if not suffix:
            canonical = ""
        else:
            key_norm = suffix.lstrip("._").lower()
            if key_norm in ("s_rel", "rel", "group_scale", "gscale", "scale"):
                canonical = "_s_rel"
            elif key_norm in ("s_channel", "channel_scale", "cscale"):
                canonical = "_s_channel"
            elif key_norm in ("codebook", "cb"):
                canonical = "_codebook"
            else:
                canonical = None

        if canonical is None:
            print(f"⚠️ W4A8: unrecognized tensor suffix '{suffix}' for {original_weight_key}, storing as-is")
            canonical = suffix if suffix.startswith((".", "_")) else f".{suffix}"

        mapped[canonical] = tensor

    if "_s_rel" not in mapped:
        raise RuntimeError(
            f"W4A8 quantization for '{original_weight_key}' did not produce a per-group scale tensor. "
            f"comfy-kitchen's AsymW4A8Int8Layout.state_dict_tensors() returned suffixes: {seen_suffixes}. "
            f"comfy/ops.py's asym_w4a8_int8 loader requires a 'weight_s_rel' tensor to load this layer. "
            f"Update comfy-kitchen to a build that emits it, or map the correct suffix above."
        )

    for canonical, tensor in mapped.items():
        out_key = original_weight_key if not canonical else original_weight_key + canonical
        assign_quantized_tensor(new_sd, out_key, tensor)


def store_awq_w4a16_quantized_weight(new_sd, original_weight_key, tensors):
    seen_suffixes = list(tensors.keys())
    mapped = {}

    for suffix, tensor in tensors.items():
        if suffix == ".comfy_quant":
            continue

        if not suffix:
            canonical = ""
        else:
            key_norm = suffix.lstrip("._").lower()
            if key_norm in ("scale", "weight_scale", "s", "scales"):
                canonical = "_scale"
            elif key_norm in ("zero", "zeros", "weight_zero", "z"):
                canonical = "_zero"
            else:
                canonical = None

        if canonical is None:
            print(f"⚠️ AWQ W4A16: unrecognized tensor suffix '{suffix}' for {original_weight_key}, storing as-is")
            canonical = suffix if suffix.startswith((".", "_")) else f".{suffix}"

        mapped[canonical] = tensor

    if "_scale" not in mapped or "_zero" not in mapped:
        raise RuntimeError(
            f"AWQ W4A16 quantization for '{original_weight_key}' did not produce both scale and zero-point "
            f"tensors. comfy-kitchen's TensorCoreAWQW4A16Layout.state_dict_tensors() returned suffixes: "
            f"{seen_suffixes}. ComfyUI's awq_w4a16 loader requires 'weight_scale' and 'weight_zero' tensors. "
            f"Update comfy-kitchen to a build that emits them, or map the correct suffixes above."
        )

    for canonical, tensor in mapped.items():
        out_key = original_weight_key if not canonical else original_weight_key + canonical
        assign_quantized_tensor(new_sd, out_key, tensor)


def quantize_w4a8_convrot(weight_f32: torch.Tensor):
    if not W4A8_AVAILABLE:
        raise RuntimeError(
            "W4A8 ConvRot requires comfy-kitchen AsymW4A8Int8Layout. "
            "Update comfy-kitchen to a build that contains it."
        )

    try:
        sig = inspect.signature(W4A8_LAYOUT.quantize)
        kwargs = {}

        if "group_size" in sig.parameters:
            kwargs["group_size"] = W4A8_GROUP_SIZE
        elif "quant_group_size" in sig.parameters:
            kwargs["quant_group_size"] = W4A8_GROUP_SIZE

        if "convrot_groupsize" in sig.parameters:
            kwargs["convrot_groupsize"] = W4A8_CONVROT_GROUPSIZE

        if "convrot" in sig.parameters:
            kwargs["convrot"] = True

        if "codebook" in sig.parameters:
            kwargs["codebook"] = W4A8_CODEBOOK

        if "symmetric" in sig.parameters:
            kwargs["symmetric"] = W4A8_SYMMETRIC

        return W4A8_LAYOUT.quantize(weight_f32, **kwargs)

    except TypeError:
        return W4A8_LAYOUT.quantize(
            weight_f32,
            group_size=W4A8_GROUP_SIZE,
            convrot_groupsize=W4A8_CONVROT_GROUPSIZE,
            symmetric=W4A8_SYMMETRIC,
            codebook=W4A8_CODEBOOK,
        )


def quantize_awq_w4a16(weight_f32: torch.Tensor, group_size: int = AWQ_W4A16_GROUP_SIZE):
    if not AWQ_W4A16_AVAILABLE:
        raise RuntimeError(
            "AWQ W4A16 requires comfy-kitchen TensorCoreAWQW4A16Layout. "
            "Update comfy-kitchen to a build that contains it."
        )

    try:
        sig = inspect.signature(AWQ_W4A16_LAYOUT.quantize)
        kwargs = {}

        if "group_size" in sig.parameters:
            kwargs["group_size"] = group_size
        elif "quant_group_size" in sig.parameters:
            kwargs["quant_group_size"] = group_size

        return AWQ_W4A16_LAYOUT.quantize(weight_f32, **kwargs)

    except TypeError:
        return AWQ_W4A16_LAYOUT.quantize(weight_f32, group_size=group_size)


def dequantize_input(sd, metadata):
    quant_layers = {}

    if metadata and "_quantization_metadata" in metadata:
        quant_layers = json.loads(metadata["_quantization_metadata"]).get("layers", {})

    for k in [k for k in sd if k.endswith(".comfy_quant")]:
        conf = sd.pop(k)
        layer = k[:-len(".comfy_quant")]

        if layer not in quant_layers:
            try:
                quant_layers[layer] = json.loads(bytes(conf.cpu().to(torch.uint8).tolist()))
            except Exception:
                print(f"⚠️ Could not parse embedded quant config for '{layer}', ignoring.")

    for layer, info in quant_layers.items():
        fmt = info.get("format")

        if fmt in ("nvfp4", "mxfp8", "convrot_w4a4", "w4a4", "asym_w4a8_int8", "awq_w4a16"):
            raise ValueError(
                f"Input model contains {fmt} layers ('{layer}'), "
                "which cannot be dequantized losslessly. Use a higher precision source model."
            )

        if info.get("convrot") and fmt not in ("convrot_w4a4", "w4a4", "asym_w4a8_int8"):
            raise ValueError(
                f"Input model contains ConvRot-rotated INT8 layers ('{layer}'). "
                "Use a higher precision source model."
            )

    if "scaled_fp8" in sd:
        sd.pop("scaled_fp8")

        for k in [k for k in sd if k.endswith(".scale_weight")]:
            scale = sd.pop(k)
            wk = k[:-len(".scale_weight")] + ".weight"
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


def find_llama_quantize_binary(explicit_path=None):
    candidates = []

    if explicit_path:
        candidates.append(explicit_path)

    env_path = os.environ.get("LLAMA_QUANTIZE_BIN")
    if env_path:
        candidates.append(env_path)

    for name in ("llama-quantize", "llama-quantize.exe"):
        found = shutil.which(name)
        if found:
            candidates.append(found)

    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return candidate

    raise FileNotFoundError(
        "Could not locate the 'llama-quantize' binary. GGUF K-quant target formats "
        f"({', '.join(sorted(GGUF_KQUANT_MAP))}) require llama.cpp's compiled "
        "llama-quantize tool -- gguf-py's own Python quantizer only implements "
        "F16/BF16/Q4_0/Q4_1/Q5_0/Q5_1/Q8_0; K-quant types are dequantize-only there. "
        "Build llama.cpp (https://github.com/ggml-org/llama.cpp) or download a release "
        "binary, then either put it on PATH, set the LLAMA_QUANTIZE_BIN environment "
        "variable, or fill in the node's 'llama_quantize_bin' field."
    )


def _gguf_default_qtype(torch_dtype):
    return gguf.GGMLQuantizationType.BF16 if torch_dtype == torch.bfloat16 else gguf.GGMLQuantizationType.F16


def build_gguf_output_path(out_dir, base_name, target_format):
    stem = PRECISION_RE.sub("", base_name).rstrip("-_.")
    suffix = target_format[len("gguf_"):].upper()
    return os.path.join(out_dir, f"{stem}-{suffix}.gguf")


def write_gguf_tensors(writer, sd, blacklist, keep_fp32, keep_fp16, gguf_quant_type, verbose=True):
    counts = Counter()

    for key, tensor in sd.items():
        if tensor.dim() == 0:
            counts["skipped_scalar"] += 1
            continue

        if key.endswith(".comfy_quant") or (key.endswith("_scale") and tensor.dim() == 0):
            counts["skipped_scale_meta"] += 1
            continue

        old_dtype = tensor.dtype

        if tensor.dtype == torch.bfloat16:
            data = tensor.to(torch.float32).numpy()
        elif tensor.dtype in FP8_DTYPES:
            data = tensor.to(torch.float32).numpy()
        else:
            data = tensor.numpy()

        n_dims = data.ndim

        if n_dims > GGUF_MAX_TENSOR_DIMS:
            raise ValueError(
                f"Tensor '{key}' has {n_dims} dims (shape {tensor.shape}); GGUF supports "
                f"at most {GGUF_MAX_TENSOR_DIMS}. This node does not reshape/flatten "
                "oversized tensors for GGUF export."
            )

        if len(key) > GGUF_MAX_TENSOR_NAME_LENGTH:
            raise ValueError(
                f"Tensor name '{key}' is {len(key)} chars; GGUF limits names to "
                f"{GGUF_MAX_TENSOR_NAME_LENGTH}."
            )

        n_params = data.size
        data_qtype = _gguf_default_qtype(old_dtype)

        if n_dims == 1 or n_params <= GGUF_QUANTIZATION_THRESHOLD:
            data_qtype = gguf.GGMLQuantizationType.F32
            counts["f32_protected"] += 1
        elif keep_fp32 and any(name in key for name in keep_fp32):
            data_qtype = gguf.GGMLQuantizationType.F32
            counts["f32_protected"] += 1
        elif keep_fp16 and any(name in key for name in keep_fp16):
            data_qtype = gguf.GGMLQuantizationType.F16
            counts["f16_protected"] += 1
        elif blacklist and any(name in key for name in blacklist):
            counts["kept_default"] += 1
        elif n_dims == 4 and "conv" in key.lower():
            data_qtype = gguf.GGMLQuantizationType.F16
            counts["f16_conv"] += 1
        elif gguf_quant_type is not None and n_dims >= 2:
            data_qtype = gguf_quant_type
            counts[gguf_quant_type.name] += 1
        else:
            counts["kept_default"] += 1

        try:
            out_data = gguf.quants.quantize(data, data_qtype)
        except (AttributeError, gguf.QuantError) as e:
            if verbose:
                print(f"⚠️ GGUF: '{key}' falling back to F16 ({e})")
            data_qtype = gguf.GGMLQuantizationType.F16
            out_data = gguf.quants.quantize(data, data_qtype)
            counts["f16_fallback"] += 1

        writer.add_tensor(key, out_data, raw_dtype=data_qtype)

    return counts


class StarUltimateModelConverter:
    @classmethod
    def INPUT_TYPES(s):
        configs = load_model_configs()

        tenc_list = []
        if "text_encoders" in folder_paths.folder_names_and_paths:
            tenc_list.extend(folder_paths.get_filename_list("text_encoders") or [])
        if "clip" in folder_paths.folder_names_and_paths:
            tenc_list.extend(folder_paths.get_filename_list("clip") or [])
        tenc_list = sorted(list(set(tenc_list)))
        tenc_list.insert(0, "None")

        diff_list = folder_paths.get_filename_list("diffusion_models") or []
        diff_list = sorted(diff_list)
        diff_list.insert(0, "None")

        ckpt_list = folder_paths.get_filename_list("checkpoints") or []
        ckpt_list = sorted(ckpt_list)
        ckpt_list.insert(0, "None")

        vae_list = folder_paths.get_filename_list("vae") or []
        vae_list = sorted(vae_list)
        vae_list.insert(0, "None")

        return {
            "required": {
                "mode": (
                    ["Diffusion Model", "Checkpoint", "AIO", "Text-Encoder", "VAE", "Custom Path"],
                    {
                        "default": "Diffusion Model",
                        "tooltip": "Select the source type. 'AIO' processes UNet + CLIP together and ignores VAE. 'VAE' targets the models/vae/ folder.",
                    },
                ),
                "diffusion_model": (diff_list, {"tooltip": "Used if Mode is 'Diffusion Model'."}),
                "checkpoint": (ckpt_list, {"tooltip": "Used if Mode is 'Checkpoint' or 'AIO'."}),
                "text_encoder": (tenc_list, {"tooltip": "Used if Mode is 'Text-Encoder'."}),
                "vae": (vae_list, {"tooltip": "Used if Mode is 'VAE'."}),
                "model_type": (
                    list(configs["models"].keys()),
                    {"tooltip": "Choose the model architecture profile."},
                ),
                "target_format": (TARGET_FORMATS, {"default": "nvfp4"}),
                "device": (["cuda", "cpu"], {"default": "cpu"}),
            },
            "optional": {
                "custom_path": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "placeholder": "Enter path to .safetensors file or folder",
                        "tooltip": "Used only if Mode is set to 'Custom Path'.",
                    },
                ),
                "svdquant_rank": (
                    "INT",
                    {
                        "default": 64,
                        "min": 8,
                        "max": 512,
                        "step": 8,
                        "tooltip": "[svdquant_w4a4 only] Rank of the low-rank bf16 branch. Higher = better fidelity, larger file.",
                    },
                ),
                "svdquant_refine_iters": (
                    "INT",
                    {
                        "default": 10,
                        "min": 0,
                        "max": 200,
                        "tooltip": "[svdquant_w4a4 only] Refine the low-rank branch against the quantization error.",
                    },
                ),
                "gguf_arch": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "placeholder": "e.g. flux, sd3, wan, qwen_image, sdxl, ltxv, krea2, ideogram",
                        "tooltip": (
                            "[gguf_* targets only] Architecture string written into the GGUF header. "
                            "ComfyUI-GGUF's loader hard-rejects values outside its whitelist."
                        ),
                    },
                ),
                "llama_quantize_bin": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "placeholder": "path to llama-quantize (K-quant targets only)",
                        "tooltip": (
                            "[gguf_q*_k_* / gguf_q6_k targets only] Path to llama.cpp's compiled "
                            "llama-quantize binary. Leave blank to use LLAMA_QUANTIZE_BIN or PATH."
                        ),
                    },
                ),
                "keep_intermediate_gguf": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "[K-quant targets only] Keep the intermediate F16/BF16 GGUF written "
                            "before llama-quantize runs."
                        ),
                    },
                ),
                "learned_rounding": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "[fp8 and int8_convrot targets, phase 1] Refine each weight's "
                            "quantized rounding via gradient descent instead of one-shot "
                            "round-to-nearest, minimizing reconstruction error in the weight's "
                            "own top-k SVD subspace. Also applies to any layer opted into "
                            "int8_convrot via a model_type profile's int8_convrot_layers "
                            "override, regardless of the selected target_format. "
                            "Calibration-free (no activation data needed), but MUCH slower -- "
                            "runs learned_rounding_iters gradient steps per eligible weight "
                            "matrix instead of a single round() call. Output format is "
                            "unchanged, fully compatible with the normal loader."
                        ),
                    },
                ),
                "learned_rounding_iters": (
                    "INT",
                    {
                        "default": LEARNED_ROUNDING_DEFAULT_ITERS,
                        "min": 10,
                        "max": 5000,
                        "tooltip": "[learned_rounding only] Gradient-descent steps per weight matrix. Higher = better fidelity, much slower.",
                    },
                ),
                "learned_rounding_lr": (
                    "FLOAT",
                    {
                        "default": LEARNED_ROUNDING_DEFAULT_LR,
                        "min": 0.0001,
                        "max": 1.0,
                        "step": 0.0001,
                        "tooltip": "[learned_rounding only] AdamW learning rate for the rounding-refinement delta.",
                    },
                ),
                "learned_rounding_topk_ratio": (
                    "FLOAT",
                    {
                        "default": LEARNED_ROUNDING_DEFAULT_TOPK_RATIO,
                        "min": 0.01,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "[learned_rounding only] Fraction of singular vectors used to weight the reconstruction loss. Higher = considers more of the weight's structure, slower.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "convert"
    CATEGORY = "⭐StarNodes/Model Tools"
    OUTPUT_NODE = True

    def convert(
        self,
        mode,
        diffusion_model,
        checkpoint,
        text_encoder,
        model_type,
        target_format,
        device,
        custom_path="",
        vae="None",
        svdquant_rank=64,
        svdquant_refine_iters=10,
        gguf_arch="",
        llama_quantize_bin="",
        keep_intermediate_gguf=False,
        learned_rounding=False,
        learned_rounding_iters=LEARNED_ROUNDING_DEFAULT_ITERS,
        learned_rounding_lr=LEARNED_ROUNDING_DEFAULT_LR,
        learned_rounding_topk_ratio=LEARNED_ROUNDING_DEFAULT_TOPK_RATIO,
    ):
        target_format = target_format.strip()

        configs = load_model_configs()

        (
            blacklist,
            fp8_layers,
            preserve_extended,
            force_int8_layers,
            awq_prescale_layers,
            pruned_extra_blacklist,
            int8_convrot_layers,
            int4_convrot_layers,
            keep_fp32,
            keep_fp16,
            w4a8_blacklist,
            w4a8_keep_fp32,
            w4a8_keep_fp16,
            awq_w4a16_layers,
        ) = get_profile(configs, model_type)

        (
            te_blacklist,
            te_fp8_layers,
            _,
            te_force_int8_layers,
            te_awq_prescale_layers,
            te_pruned_extra_blacklist,
            te_int8_convrot_layers,
            te_int4_convrot_layers,
            te_keep_fp32,
            te_keep_fp16,
            te_w4a8_blacklist,
            te_w4a8_keep_fp32,
            te_w4a8_keep_fp16,
            te_awq_w4a16_layers,
        ) = get_profile(configs, "Text-Encoder")

        is_pruned_format = target_format.endswith("_pruned")
        is_w4a8 = target_format in ("w4a8_convrot", "w4a8_convrot_pruned")
        is_awq_w4a16 = target_format == "awq_w4a16"
        is_w4a4 = target_format == "w4a4"
        is_gguf = target_format in GGUF_TARGET_FORMATS
        is_gguf_kquant = target_format in GGUF_KQUANT_MAP

        if target_format not in ("fp16", "fp32"):
            if is_gguf:
                if not GGUF_AVAILABLE:
                    raise ValueError(
                        "GGUF target formats require the 'gguf' Python package. "
                        "Install it in ComfyUI's environment with: pip install gguf"
                    )

                if mode == "AIO":
                    raise ValueError(
                        "GGUF target formats do not support AIO mode. ComfyUI-GGUF loads the "
                        "UNet and CLIP as separate .gguf files -- convert the diffusion model "
                        "and text encoder separately."
                    )

                if mode == "Text-Encoder":
                    raise ValueError(
                        "GGUF export for Text-Encoder mode is not supported by this node. "
                        "Use one of the safetensors target formats for text encoders instead."
                    )

                if is_gguf_kquant:
                    find_llama_quantize_binary(llama_quantize_bin or None)
                    print(
                        "⚠️ K-quant GGUF targets are not reliably supported by ComfyUI-GGUF's "
                        "diffusion-model loader. Verify before relying on this output."
                    )

            elif is_w4a8:
                if not W4A8_AVAILABLE:
                    raise ValueError(
                        "W4A8 ConvRot requires comfy-kitchen with AsymW4A8Int8Layout. "
                        "Update comfy-kitchen or use a build that includes the W4A8 layout."
                    )

            elif is_awq_w4a16:
                if not AWQ_W4A16_AVAILABLE:
                    raise ValueError(
                        "AWQ W4A16 requires comfy-kitchen with TensorCoreAWQW4A16Layout. "
                        "Update comfy-kitchen to a build that includes the AWQ W4A16 layout."
                    )

            elif is_w4a4:
                if not KITCHEN_AVAILABLE or TensorCoreConvRotW4A4Layout is None:
                    raise ValueError(
                        "w4a4 requires comfy-kitchen TensorCoreConvRotW4A4Layout. "
                        "Update comfy-kitchen to a build that contains it."
                    )

            elif target_format == "svdquant_w4a4":
                if not KITCHEN_AVAILABLE or TensorCoreConvRotW4A4Layout is None:
                    raise ValueError(
                        "svdquant_w4a4 requires comfy-kitchen TensorCoreConvRotW4A4Layout. "
                        "Update comfy-kitchen to a build that contains it."
                    )

            elif not KITCHEN_AVAILABLE:
                raise ValueError("comfy-kitchen is required for this target format.")

        start_time = time.time()
        print(f"🚀 [Star Ultimate Model Converter] Mode: {mode} | Profile: {model_type} | Target: {target_format}")

        if target_format == "svdquant_w4a4":
            print(f"🧬 SVDQuant rank: {svdquant_rank}  refine_iters: {svdquant_refine_iters}")

        if mode in ("Checkpoint", "AIO"):
            if not checkpoint or checkpoint == "None":
                raise ValueError(f"Mode is '{mode}' but no checkpoint is selected. Please choose a model from the dropdown.")

            ckpt_path = folder_paths.get_full_path("checkpoints", checkpoint)
            base_name = os.path.splitext(os.path.basename(ckpt_path))[0]
            orig_meta = None

            if ckpt_path.endswith(".safetensors"):
                with safetensors.safe_open(ckpt_path, framework="pt") as f:
                    orig_meta = f.metadata()

            full_sd = comfy.utils.load_torch_file(ckpt_path, safe_load=True)

            if mode == "Checkpoint":
                print(f"✂️ Extracting diffusion model from AIO checkpoint: {checkpoint}")

                sd = {
                    k[len(AIO_MODEL_PREFIX):]: v
                    for k, v in full_sd.items()
                    if k.startswith(AIO_MODEL_PREFIX)
                }

                input_bytes = sum(v.numel() * v.element_size() for v in sd.values())
                output_path = (
                    build_gguf_output_path(diffusion_models_dir(), base_name, target_format)
                    if is_gguf else
                    build_output_path(diffusion_models_dir(), base_name, target_format)
                )

                files = [ckpt_path]
                del full_sd

            else:
                print(f"🔄 AIO Mode: Processing entire checkpoint intact: {checkpoint}")

                sd = full_sd
                input_bytes = os.path.getsize(ckpt_path)

                checkpoints_dir = folder_paths.get_folder_paths("checkpoints")[0]
                output_path = build_output_path(checkpoints_dir, f"{base_name}_AIO", target_format)

                files = [ckpt_path]

        else:
            files, out_dir, base_name = resolve_input(
                mode,
                diffusion_model,
                checkpoint,
                text_encoder,
                custom_path,
                vae,
            )

            output_path = (
                build_gguf_output_path(out_dir, base_name, target_format)
                if is_gguf else
                build_output_path(out_dir, base_name, target_format)
            )

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

        if is_gguf:
            gguf_blacklist, gguf_keep_fp32, gguf_keep_fp16 = blacklist, keep_fp32, keep_fp16

            resolved_arch = (
                gguf_arch.strip()
                or configs["models"].get(model_type, {}).get("gguf_arch")
                or MODEL_TYPE_TO_GGUF_ARCH.get(model_type)
            )

            if resolved_arch not in GGUF_IMG_ARCH_LIST:
                raise ValueError(
                    f"No confirmed GGUF architecture mapping for model_type '{model_type}'. "
                    f"ComfyUI-GGUF's loader hard-rejects any GGUF whose 'general.architecture' "
                    f"header isn't one of: {', '.join(sorted(GGUF_IMG_ARCH_LIST))}. "
                    "Fill in the node's 'gguf_arch' field if you know this model is compatible."
                )

            return self._convert_to_gguf(
                sd=sd,
                target_format=target_format,
                output_path=output_path,
                blacklist=gguf_blacklist,
                keep_fp32=gguf_keep_fp32,
                keep_fp16=gguf_keep_fp16,
                arch=resolved_arch,
                model_type=model_type,
                mode=mode,
                input_bytes=input_bytes,
                input_format=input_format,
                files=files,
                start_time=start_time,
                llama_quantize_bin=llama_quantize_bin,
                keep_intermediate_gguf=keep_intermediate_gguf,
            )

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

                if mode == "AIO":
                    if k.startswith(AIO_MODEL_PREFIX):
                        active_blacklist = blacklist
                        active_fp8 = fp8_layers
                        active_force_int8 = force_int8_layers
                        active_awq_prescale = awq_prescale_layers
                        active_pruned_extra = pruned_extra_blacklist
                        active_int8_convrot = int8_convrot_layers
                        active_int4_convrot = int4_convrot_layers
                        active_keep_fp32 = keep_fp32
                        active_keep_fp16 = keep_fp16
                        active_w4a8_blacklist = w4a8_blacklist
                        active_w4a8_keep_fp32 = w4a8_keep_fp32
                        active_w4a8_keep_fp16 = w4a8_keep_fp16
                        active_awq_w4a16_layers = awq_w4a16_layers

                    elif (
                        k.startswith("cond_stage_model.")
                        or k.startswith("conditioner.")
                        or k.startswith("text_encoders.")
                    ):
                        active_blacklist = te_blacklist
                        active_fp8 = te_fp8_layers
                        active_force_int8 = te_force_int8_layers
                        active_awq_prescale = te_awq_prescale_layers
                        active_pruned_extra = te_pruned_extra_blacklist
                        active_int8_convrot = te_int8_convrot_layers
                        active_int4_convrot = te_int4_convrot_layers
                        active_keep_fp32 = te_keep_fp32
                        active_keep_fp16 = te_keep_fp16
                        active_w4a8_blacklist = te_w4a8_blacklist
                        active_w4a8_keep_fp32 = te_w4a8_keep_fp32
                        active_w4a8_keep_fp16 = te_w4a8_keep_fp16
                        active_awq_w4a16_layers = te_awq_w4a16_layers

                    else:
                        if v.dtype.is_floating_point:
                            new_sd[k] = v.to(dtype=torch.bfloat16)
                            counts["kept bf16 (VAE/Misc)"] += 1
                        else:
                            new_sd[k] = v
                            counts["kept (VAE/Misc)"] += 1
                        continue

                else:
                    active_blacklist = blacklist
                    active_fp8 = fp8_layers
                    active_force_int8 = force_int8_layers
                    active_awq_prescale = awq_prescale_layers
                    active_pruned_extra = pruned_extra_blacklist
                    active_int8_convrot = int8_convrot_layers
                    active_int4_convrot = int4_convrot_layers
                    active_keep_fp32 = keep_fp32
                    active_keep_fp16 = keep_fp16
                    active_w4a8_blacklist = w4a8_blacklist
                    active_w4a8_keep_fp32 = w4a8_keep_fp32
                    active_w4a8_keep_fp16 = w4a8_keep_fp16
                    active_awq_w4a16_layers = awq_w4a16_layers

                if is_pruned_format and active_pruned_extra:
                    if any(name in k for name in active_pruned_extra) and not any(name in k for name in active_blacklist):
                        active_blacklist = list(active_blacklist) + [n for n in active_pruned_extra if n in k]

                if any(name in k for name in active_blacklist):
                    if v.dtype.is_floating_point:
                        new_sd[k] = v.to(dtype=blacklisted_dtype(k, active_keep_fp32, active_keep_fp16))
                        counts["kept bf16/f16/f32"] += 1
                    else:
                        new_sd[k] = v
                        counts["kept"] += 1
                    continue

                if v.ndim == 2 and ".weight" in k:
                    base_k_file = k.replace(".weight", "")
                    # Strip the AIO prefix from the metadata key so ComfyUI's
                    # loader finds the .comfy_quant entry under the model's
                    # internal (unprefixed) module path. Tensor keys below are
                    # left untouched -- the loader strips those prefixes itself.
                    if base_k_file.startswith(AIO_MODEL_PREFIX):
                        base_k_meta = base_k_file[len(AIO_MODEL_PREFIX):]
                    else:
                        base_k_meta = base_k_file

                    v_tensor = v.to(device=device, dtype=torch.bfloat16)

                    if (
                        active_force_int8
                        and any(name in k for name in active_force_int8)
                        and target_format not in ("int4_tensorwise", "int4_tensorwise_pruned")
                    ):
                        print(f"🔒 FORCE-INT8: {k}")

                        try:
                            v_tensor_ready = v_tensor.float().contiguous()
                            qdata, params = TensorWiseINT8Layout.quantize(v_tensor_ready)
                            tensors = TensorWiseINT8Layout.state_dict_tensors(qdata, params)

                            store_quantized_weight(new_sd, k, tensors)
                            quant_map["layers"][base_k_meta] = {"format": "int8_tensorwise"}
                            counts["forced_int8"] += 1

                            if device == "cuda":
                                del v_tensor, v_tensor_ready

                        except Exception as e:
                            print(f"⚠️ Forced INT8 failed for {k}: {e}")

                            if v.dtype.is_floating_point:
                                new_sd[k] = v.to(dtype=torch.bfloat16)
                                counts["kept bf16"] += 1
                            else:
                                new_sd[k] = v
                                counts["kept"] += 1

                            if device == "cuda":
                                del v_tensor

                        continue

                    if active_int4_convrot and any(name in k for name in active_int4_convrot):
                        print(f"💎 OVERRIDE int4_convrot: {k}")

                        try:
                            v_tensor_ready = v_tensor.float().contiguous()

                            qdata, params = TensorCoreConvRotW4A4Layout.quantize(
                                v_tensor_ready,
                                convrot_groupsize=CONVROT_GROUPSIZE,
                                quant_group_size=INT4_QUANT_GROUPSIZE,
                            )

                            tensors = TensorCoreConvRotW4A4Layout.state_dict_tensors(qdata, params)

                            store_quantized_weight(new_sd, k, tensors)

                            quant_map["layers"][base_k_meta] = {
                                "format": "convrot_w4a4",
                                "convrot_groupsize": CONVROT_GROUPSIZE,
                                "quant_group_size": INT4_QUANT_GROUPSIZE,
                            }

                            counts["override_int4_convrot"] += 1

                            if device == "cuda":
                                del v_tensor, v_tensor_ready

                        except Exception as e:
                            print(f"⚠️ OVERRIDE int4_convrot failed for {k}: {e}")

                            if v.dtype.is_floating_point:
                                new_sd[k] = v.to(dtype=torch.bfloat16)
                                counts["kept bf16"] += 1
                            else:
                                new_sd[k] = v
                                counts["kept"] += 1

                            if device == "cuda":
                                del v_tensor

                        continue

                    if active_int8_convrot and any(name in k for name in active_int8_convrot):
                        print(f"💎 OVERRIDE int8_convrot{' + learned rounding' if learned_rounding else ''}: {k}")

                        try:
                            v_tensor_ready = v_tensor.float().contiguous()

                            if learned_rounding:
                                qdata, scale, quant_conf = quantize_int8_rowwise_convrot(
                                    v_tensor_ready,
                                    convrot_groupsize=CONVROT_GROUPSIZE,
                                    use_learned_rounding=True,
                                    learned_rounding_iters=learned_rounding_iters,
                                    learned_rounding_lr=learned_rounding_lr,
                                    learned_rounding_topk_ratio=learned_rounding_topk_ratio,
                                )
                                new_sd[k] = qdata.cpu()
                                new_sd[f"{base_k_file}.weight_scale"] = scale.to(torch.bfloat16).cpu()
                                quant_map["layers"][base_k_meta] = quant_conf
                            else:
                                qdata, params = TensorWiseINT8Layout.quantize(
                                    v_tensor_ready,
                                    per_channel=True,
                                    convrot=True,
                                    convrot_groupsize=CONVROT_GROUPSIZE,
                                )
                                tensors = TensorWiseINT8Layout.state_dict_tensors(qdata, params)
                                store_quantized_weight(new_sd, k, tensors)
                                quant_map["layers"][base_k_meta] = {
                                    "format": "int8_tensorwise",
                                    "convrot": True,
                                    "convrot_groupsize": CONVROT_GROUPSIZE,
                                }

                            counts["override_int8_convrot"] += 1

                            if device == "cuda":
                                del v_tensor, v_tensor_ready

                        except Exception as e:
                            print(f"⚠️ OVERRIDE int8_convrot failed for {k}: {e}")

                            if v.dtype.is_floating_point:
                                new_sd[k] = v.to(dtype=torch.bfloat16)
                                counts["kept bf16"] += 1
                            else:
                                new_sd[k] = v
                                counts["kept"] += 1

                            if device == "cuda":
                                del v_tensor

                        continue

                    if (
                        active_awq_w4a16_layers
                        and any(name in k for name in active_awq_w4a16_layers)
                        and target_format != "awq_w4a16"
                    ):
                        print(f"💎 OVERRIDE awq_w4a16: {k}")

                        try:
                            v_tensor_ready = v_tensor.float().contiguous()
                            qdata, params = quantize_awq_w4a16(v_tensor_ready)
                            tensors = AWQ_W4A16_LAYOUT.state_dict_tensors(qdata, params)

                            store_awq_w4a16_quantized_weight(new_sd, k, tensors)

                            quant_map["layers"][base_k_meta] = {
                                "format": AWQ_W4A16_FORMAT_NAME,
                                "group_size": AWQ_W4A16_GROUP_SIZE,
                            }

                            counts["override_awq_w4a16"] += 1

                            if device == "cuda":
                                del v_tensor, v_tensor_ready

                        except Exception as e:
                            print(f"⚠️ OVERRIDE awq_w4a16 failed for {k}: {e}")

                            if v.dtype.is_floating_point:
                                new_sd[k] = v.to(dtype=torch.bfloat16)
                                counts["kept bf16"] += 1
                            else:
                                new_sd[k] = v
                                counts["kept"] += 1

                            if device == "cuda":
                                del v_tensor

                        continue

                    if target_format == "fp8" or (active_fp8 and any(name in k for name in active_fp8)):
                        weight_scale = (v_tensor.abs().max() / 448.0).clamp(min=1e-12).float()

                        if learned_rounding and v_tensor.dim() == 2:
                            print(f"🎯 FP8 (learned rounding, {learned_rounding_iters} iters): {k}")
                            try:
                                v_tensor_ready = v_tensor.float().contiguous()
                                inv_scale = 1.0 / weight_scale
                                weight_quantized = learned_round_refine(
                                    v_tensor_ready,
                                    inv_scale,
                                    torch.float8_e4m3fn,
                                    num_iter=learned_rounding_iters,
                                    lr=learned_rounding_lr,
                                    topk_ratio=learned_rounding_topk_ratio,
                                )
                            except Exception as e:
                                print(f"⚠️ Learned rounding failed for {k}, falling back to naive round: {e}")
                                weight_quantized = ck.quantize_per_tensor_fp8(v_tensor, weight_scale)
                        else:
                            print(f"🌸 FP8: {k}")
                            weight_quantized = ck.quantize_per_tensor_fp8(v_tensor, weight_scale)

                        new_sd[k] = weight_quantized.cpu()
                        new_sd[f"{base_k_file}.weight_scale"] = weight_scale.to(torch.bfloat16).cpu()

                        quant_map["layers"][base_k_meta] = {"format": "float8_e4m3fn"}
                        counts["fp8"] += 1

                        if device == "cuda":
                            del v_tensor

                        continue

                    if target_format == "svdquant_w4a4":
                        print(f"💎 SVDQUANT_W4A4 (rank {svdquant_rank}): {k}")

                        try:
                            v_tensor_ready = v_tensor.float().contiguous()

                            split = svdquant_split(
                                v_tensor_ready,
                                svdquant_rank,
                                CONVROT_GROUPSIZE,
                                svdquant_refine_iters,
                            )

                            if split is None:
                                print(
                                    f"  warning: {k} is degenerate (zero or non-finite); "
                                    f"quantizing without a low-rank branch"
                                )
                                residual = v_tensor_ready.to(torch.bfloat16)
                                l1 = l2 = None
                            else:
                                residual, l1, l2 = split

                            layout = TensorCoreConvRotW4A4Layout

                            qdata, params = layout.quantize(
                                residual.float().contiguous(),
                                convrot_groupsize=CONVROT_GROUPSIZE,
                            )

                            tensors = layout.state_dict_tensors(qdata, params)
                            store_quantized_weight(new_sd, k, tensors)

                            layer_conf = {
                                "format": "convrot_w4a4",
                                "convrot_groupsize": CONVROT_GROUPSIZE,
                                "svdquant": l1 is not None,
                                "svdquant_rank": svdquant_rank,
                            }

                            if l1 is not None:
                                new_sd[f"{base_k_file}.svdq_l1"] = l1.cpu()
                                new_sd[f"{base_k_file}.svdq_l2"] = l2.cpu()

                            quant_map["layers"][base_k_meta] = layer_conf
                            counts["svdquant_w4a4"] += 1

                            if device == "cuda":
                                del v_tensor, v_tensor_ready

                        except Exception as e:
                            print(f"⚠️ SVDQuant failed for {k}: {e}")

                            if v.dtype.is_floating_point:
                                new_sd[k] = v.to(dtype=torch.bfloat16)
                                counts["kept bf16"] += 1
                            else:
                                new_sd[k] = v
                                counts["kept"] += 1

                            if device == "cuda":
                                del v_tensor

                        continue

                    if target_format == "minimax_h3_native_mix":
                        blk_idx = block_index_from_key(k)

                        if "attn.out_proj" in k:
                            new_sd[k] = v.to(dtype=torch.bfloat16)
                            counts["kept bf16 (out_proj)"] += 1
                            continue

                        if blk_idx in MINIMAX_H3_BOUNDARY_BLOCKS:
                            new_sd[k] = v.to(dtype=torch.bfloat16)
                            counts["kept bf16 (boundary block)"] += 1
                            continue

                        if "attn.qkv_proj" in k:
                            print(f"💎 NATIVE_MIX int8_convrot: {k}")

                            try:
                                v_tensor_ready = v_tensor.float().contiguous()

                                qdata, params = TensorWiseINT8Layout.quantize(
                                    v_tensor_ready,
                                    per_channel=True,
                                    convrot=True,
                                    convrot_groupsize=CONVROT_GROUPSIZE,
                                )

                                tensors = TensorWiseINT8Layout.state_dict_tensors(qdata, params)
                                store_quantized_weight(new_sd, k, tensors)

                                quant_map["layers"][base_k_meta] = {
                                    "format": "int8_tensorwise",
                                    "convrot": True,
                                    "convrot_groupsize": CONVROT_GROUPSIZE,
                                }

                                counts["native_mix_qkv_int8convrot"] += 1

                                if device == "cuda":
                                    del v_tensor, v_tensor_ready

                            except Exception as e:
                                print(f"⚠️ NATIVE_MIX qkv_proj failed for {k}: {e}")
                                new_sd[k] = v.to(dtype=torch.bfloat16)
                                counts["kept bf16"] += 1

                                if device == "cuda":
                                    del v_tensor

                            continue

                        if "mlp.fc1" in k or "mlp.fc2" in k:
                            print(f"💎 NATIVE_MIX nvfp4: {k}")

                            try:
                                v_tensor_ready = v_tensor.float().contiguous()

                                qdata, params = TensorCoreNVFP4Layout.quantize(v_tensor_ready)
                                tensors = TensorCoreNVFP4Layout.state_dict_tensors(qdata, params)

                                store_quantized_weight(new_sd, k, tensors)

                                quant_map["layers"][base_k_meta] = {
                                    "format": "nvfp4",
                                    "group_size": MINIMAX_H3_NVFP4_GROUPSIZE,
                                }

                                counts["native_mix_mlp_nvfp4"] += 1

                                if device == "cuda":
                                    del v_tensor, v_tensor_ready

                            except Exception as e:
                                print(f"⚠️ NATIVE_MIX mlp failed for {k}: {e}")
                                new_sd[k] = v.to(dtype=torch.bfloat16)
                                counts["kept bf16"] += 1

                                if device == "cuda":
                                    del v_tensor

                            continue

                        new_sd[k] = v.to(dtype=torch.bfloat16)
                        counts["kept bf16 (native_mix other)"] += 1
                        continue

                    if is_w4a8:
                        if active_w4a8_blacklist and any(name in k for name in active_w4a8_blacklist):
                            target_dtype = blacklisted_dtype(
                                k,
                                active_w4a8_keep_fp32 or active_keep_fp32,
                                active_w4a8_keep_fp16 or active_keep_fp16,
                            )
                            new_sd[k] = v.to(dtype=target_dtype)
                            counts["w4a8_blacklisted"] += 1
                            continue

                        print(f"💎 W4A8_CONVROT: {k}")

                        try:
                            v_tensor_ready = v_tensor.float().contiguous()
                            qdata, params = quantize_w4a8_convrot(v_tensor_ready)
                            tensors = W4A8_LAYOUT.state_dict_tensors(qdata, params)

                            store_w4a8_quantized_weight(new_sd, k, tensors)

                            quant_map["layers"][base_k_meta] = {
                                "format": W4A8_FORMAT_NAME,
                                "group_size": W4A8_GROUP_SIZE,
                                "convrot": True,
                                "convrot_groupsize": W4A8_CONVROT_GROUPSIZE,
                            }

                            counts[target_format] += 1

                            if device == "cuda":
                                del v_tensor, v_tensor_ready

                        except Exception as e:
                            print(f"⚠️ W4A8 ConvRot failed for {k}: {e}")

                            if v.dtype.is_floating_point:
                                new_sd[k] = v.to(dtype=torch.bfloat16)
                                counts["w4a8_failed_bf16"] += 1
                            else:
                                new_sd[k] = v
                                counts["kept"] += 1

                            if device == "cuda":
                                del v_tensor

                        continue

                    if is_awq_w4a16:
                        print(f"💎 AWQ_W4A16: {k}")

                        try:
                            v_tensor_ready = v_tensor.float().contiguous()
                            qdata, params = quantize_awq_w4a16(v_tensor_ready)
                            tensors = AWQ_W4A16_LAYOUT.state_dict_tensors(qdata, params)

                            store_awq_w4a16_quantized_weight(new_sd, k, tensors)

                            quant_map["layers"][base_k_meta] = {
                                "format": AWQ_W4A16_FORMAT_NAME,
                                "group_size": AWQ_W4A16_GROUP_SIZE,
                            }

                            counts[target_format] += 1

                            if device == "cuda":
                                del v_tensor, v_tensor_ready

                        except Exception as e:
                            print(f"⚠️ AWQ W4A16 failed for {k}: {e}")

                            if v.dtype.is_floating_point:
                                new_sd[k] = v.to(dtype=torch.bfloat16)
                                counts["awq_w4a16_failed_bf16"] += 1
                            else:
                                new_sd[k] = v
                                counts["kept"] += 1

                            if device == "cuda":
                                del v_tensor

                        continue

                    int8_convrot = target_format in ("int8_convrot", "int8_convrot_pruned")
                    int4_convrot = target_format in ("int4_convrot", "int4_convrot_pruned", "w4a4")

                    if int8_convrot and learned_rounding:
                        print(f"🎯 {target_format.upper()} (learned rounding, {learned_rounding_iters} iters): {k}")
                        try:
                            v_tensor_ready = v_tensor.float().contiguous()
                            qdata, scale, quant_conf = quantize_int8_rowwise_convrot(
                                v_tensor_ready,
                                convrot_groupsize=CONVROT_GROUPSIZE,
                                use_learned_rounding=True,
                                learned_rounding_iters=learned_rounding_iters,
                                learned_rounding_lr=learned_rounding_lr,
                                learned_rounding_topk_ratio=learned_rounding_topk_ratio,
                            )
                            new_sd[k] = qdata.cpu()
                            new_sd[f"{base_k_file}.weight_scale"] = scale.to(torch.bfloat16).cpu()
                            quant_map["layers"][base_k_meta] = quant_conf
                            counts[target_format] += 1
                            if device == "cuda":
                                del v_tensor, v_tensor_ready
                        except Exception as e:
                            print(f"⚠️ Learned-rounding {target_format} failed for {k}, falling back to naive: {e}")
                            try:
                                v_tensor_ready = v_tensor.float().contiguous()
                                qdata, params = TensorWiseINT8Layout.quantize(
                                    v_tensor_ready, per_channel=True, convrot=True,
                                    convrot_groupsize=CONVROT_GROUPSIZE,
                                )
                                tensors = TensorWiseINT8Layout.state_dict_tensors(qdata, params)
                                store_quantized_weight(new_sd, k, tensors)
                                quant_map["layers"][base_k_meta] = {
                                    "format": "int8_tensorwise", "convrot": True,
                                    "convrot_groupsize": CONVROT_GROUPSIZE,
                                }
                                counts[target_format] += 1
                            except Exception as e2:
                                print(f"⚠️ Naive fallback also failed for {k}: {e2}")
                                if v.dtype.is_floating_point:
                                    new_sd[k] = v.to(dtype=torch.bfloat16)
                                    counts["kept bf16"] += 1
                                else:
                                    new_sd[k] = v
                                    counts["kept"] += 1
                            if device == "cuda":
                                del v_tensor
                        continue

                    if target_format in ("int8", "int8_convrot", "int8_convrot_pruned"):
                        layout = TensorWiseINT8Layout
                        fmt_name = "int8_tensorwise"
                    elif target_format in ("int4_convrot", "int4_convrot_pruned", "w4a4"):
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
                        pre_quant_scale = None

                        if (
                            target_format == "nvfp4"
                            and active_awq_prescale
                            and any(name in k for name in active_awq_prescale)
                        ):
                            pre_quant_scale = compute_awq_prescale(v_tensor_ready)
                            v_tensor_ready = v_tensor_ready * pre_quant_scale.unsqueeze(0)

                        if int8_convrot:
                            qdata, params = layout.quantize(
                                v_tensor_ready,
                                per_channel=True,
                                convrot=True,
                                convrot_groupsize=CONVROT_GROUPSIZE,
                            )
                        elif int4_convrot:
                            qdata, params = layout.quantize(
                                v_tensor_ready,
                                convrot_groupsize=CONVROT_GROUPSIZE,
                                quant_group_size=INT4_QUANT_GROUPSIZE,
                            )
                        elif target_format == "mxfp8" and mxfp8_backend is not None:
                            with ck_registry.use_backend(mxfp8_backend):
                                qdata, params = layout.quantize(v_tensor_ready)
                        else:
                            qdata, params = layout.quantize(v_tensor_ready)

                        tensors = layout.state_dict_tensors(qdata, params)
                        store_quantized_weight(new_sd, k, tensors)

                        if pre_quant_scale is not None:
                            new_sd[f"{base_k_file}.pre_quant_scale"] = pre_quant_scale.to(torch.bfloat16).cpu()

                        layer_conf = {"format": fmt_name}

                        if int8_convrot:
                            layer_conf["convrot"] = True
                            layer_conf["convrot_groupsize"] = CONVROT_GROUPSIZE
                        elif int4_convrot:
                            layer_conf["convrot_groupsize"] = CONVROT_GROUPSIZE
                            layer_conf["quant_group_size"] = INT4_QUANT_GROUPSIZE

                        if pre_quant_scale is not None:
                            layer_conf["awq_prescale"] = True

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

                    if device == "cuda":
                        del v_tensor

                else:
                    if v.dtype.is_floating_point:
                        new_sd[k] = v.to(dtype=torch.bfloat16)
                        counts["kept bf16"] += 1
                    else:
                        new_sd[k] = v
                        counts["kept"] += 1

        new_sd = {k: v for k, v in new_sd.items() if not k.endswith(".comfy_quant")}

        final_metadata = OrderedDict()

        if quant_map["layers"]:
            final_metadata["_quantization_metadata"] = json.dumps(quant_map)

        final_metadata["converted_by"] = "Star Ultimate Model Converter"

        if target_format == "svdquant_w4a4":
            final_metadata["svdquant_rank"] = str(svdquant_rank)
            final_metadata["svdquant_refine_iters"] = str(svdquant_refine_iters)

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

        status = "\n".join(
            [
                f"✅ Success ({model_type} → {target_format})",
                f"Input: {input_desc}",
                f"Original format: {input_format}",
                f"Original size: {format_size(input_bytes)}",
                f"New size: {format_size(output_bytes)} ({reduction:.1f}% smaller)",
                f"Layers: {layers_desc}",
                f"Device: {device} | Time: {duration:.1f}s",
                f"Saved to: {output_path}",
            ]
        )

        return (status,)

    def _convert_to_gguf(
        self,
        sd,
        target_format,
        output_path,
        blacklist,
        keep_fp32,
        keep_fp16,
        arch,
        model_type,
        mode,
        input_bytes,
        input_format,
        files,
        start_time,
        llama_quantize_bin,
        keep_intermediate_gguf,
    ):
        is_kquant = target_format in GGUF_KQUANT_MAP
        gguf_type_name = GGUF_LEGACY_QUANT_MAP.get(target_format)
        gguf_quant_type = getattr(gguf.GGMLQuantizationType, gguf_type_name) if gguf_type_name else None

        print(f"🧱 GGUF arch: '{arch}'")

        def write_gguf(dst_path, quant_type):
            writer = gguf.GGUFWriter(path=None, arch=arch)
            writer.add_quantization_version(gguf.GGML_QUANT_VERSION)

            counts = write_gguf_tensors(writer, sd, blacklist, keep_fp32, keep_fp16, quant_type)

            writer.write_header_to_file(path=dst_path)
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file(progress=True)
            writer.close()

            return counts

        if not is_kquant:
            counts = write_gguf(output_path, gguf_quant_type)
        else:
            llama_bin = find_llama_quantize_binary(llama_quantize_bin or None)

            out_dir = os.path.dirname(os.path.abspath(output_path)) or "."
            os.makedirs(out_dir, exist_ok=True)

            fd, intermediate_path = tempfile.mkstemp(suffix=".gguf", prefix="kquant_stage_", dir=out_dir)
            os.close(fd)
            os.remove(intermediate_path)

            try:
                print(f"* Stage 1/2: writing intermediate F16/BF16 GGUF -> {intermediate_path}")
                counts = write_gguf(intermediate_path, None)

                cli_type = GGUF_KQUANT_MAP[target_format]
                print(f"* Stage 2/2: llama-quantize --pure {cli_type} -> {output_path}")

                command = [llama_bin, "--pure", intermediate_path, output_path, cli_type]
                result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

                if result.stdout:
                    for line in result.stdout.splitlines():
                        print(f"[llama-quantize] {line}")

                if result.returncode != 0:
                    raise RuntimeError(
                        f"llama-quantize exited with code {result.returncode} while producing "
                        f"{target_format}. See the log above for details."
                    )

                if not os.path.isfile(output_path):
                    raise RuntimeError(
                        f"llama-quantize reported success but no output file was found at {output_path}."
                    )

            finally:
                if os.path.isfile(intermediate_path):
                    if keep_intermediate_gguf:
                        print(f"Kept intermediate GGUF at {intermediate_path}")
                    else:
                        os.remove(intermediate_path)

        output_bytes = os.path.getsize(output_path)
        duration = time.time() - start_time
        reduction = (1 - output_bytes / input_bytes) * 100 if input_bytes else 0

        print(f"✅ Done. Final size: {format_size(output_bytes)}")

        if mode == "Checkpoint":
            input_desc = f"diffusion model from AIO checkpoint {os.path.basename(files[0])}"
        elif len(files) > 1:
            input_desc = f"{len(files)} files from {os.path.basename(os.path.dirname(files[0]))}"
        else:
            input_desc = os.path.basename(files[0])

        layers_desc = ", ".join(f"{n} {name}" for name, n in counts.most_common())

        status = "\n".join(
            [
                f"✅ Success ({model_type} → {target_format})",
                f"Input: {input_desc}",
                f"Original format: {input_format}",
                f"Original size: {format_size(input_bytes)}",
                f"New size: {format_size(output_bytes)} ({reduction:.1f}% smaller)",
                f"GGUF arch: {arch}",
                f"Layers: {layers_desc}",
                f"Time: {duration:.1f}s",
                f"Saved to: {output_path}",
            ]
        )

        return (status,)


NODE_CLASS_MAPPINGS = {
    "StarUltimateModelConverter": StarUltimateModelConverter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarUltimateModelConverter": "⭐ Star Ultimate Model Converter",
}