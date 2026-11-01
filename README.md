# ⭐ StarNodes Model Converter v1.7

**Quantize, convert, split and inspect diffusion models — inside ComfyUI.**

Profiles for 45 model families, formats from FP32 down to 4-bit, an analyzer that tells you
exactly what ended up in the file, and tools to take an all-in-one checkpoint apart or put one
back together.

---

## 🎉 v1.7 — W8A16 targets (INT8 weights, 16-bit compute)

ComfyUI has never stored activations, so "W8A16" isn't a file format — it's a **compute choice**
attached to INT8 weights. v1.7 makes that choice a first-class target format:

| Target | Weights | Matmul | Use when |
|---|---|---|---|
| `int8_convrot` | INT8 + per-row scale, Hadamard-rotated | fused INT8 GEMM (activations quantized per row inside the kernel) | default: fastest |
| **`w8a16_convrot`** | **identical bytes** | **weight dequantized to bf16, plain bf16 matmul** | activation-quantization error matters |
| `w8a16` | INT8 + scalar scale | same bf16 matmul | same, unrotated (worse quality — prefer `_convrot`) |
| `w8a16_convrot_pruned` | as `w8a16_convrot`, pruned-profile blacklists applied | same | pruned models |

New targets: **`w8a16`**, **`w8a16_convrot`**, **`w8a16_convrot_pruned`**. Pick one in
`target_format` like any other format — everything else about the conversion is unchanged.

**What it writes.** Exactly what `int8` / `int8_convrot` write (an `int8_tensorwise` weight plus
`weight_scale`, optionally ConvRot GS:256), and adds one flag to every quantized module:

```json
{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256,
 "full_precision_matrix_mult": true}
```

ComfyUI reads `full_precision_matrix_mult` and dequantizes the weight for a bf16 matmul instead of
running `comfy-kitchen`'s `int8_linear`, which quantizes activations per row on the fly. So the
weights stay 8-bit on disk **and in VRAM**, while the arithmetic is 16-bit.

**What it costs.** File size is unchanged — this is not a size win.

- Matmul runs ~2× slower than `int8_convrot`: the INT8 tensor cores sit idle.
- Each weight is materialised as bf16 for the duration of its layer, so peak VRAM rises above the
  resident INT8 size (the resident weight is still INT8).
- In exchange, activation-quantization error is gone — the output is perceptually bf16-level.

**Verified** on ROCm (RX 7800 XT, gfx1101): a 9.47 GB bf16 MOSS-Image source → **5.04 GB**
`w8a16_convrot` in **15 s**, all **168/168** block linears flagged, size identical to the INT8
ConvRot twin. Through ComfyUI's loader the module comes back as an INT8 `QuantizedTensor` with
`_full_precision_mm = True`, and its forward is **bit-identical** to a bf16 matmul on the
dequantized weight.

The Layers Info node labels these layers `INT8_CONVROT (GS:256) [W8A16]` so you can confirm what a
file actually is.

---

## Included nodes

| Node | What it does |
|---|---|
| ⭐ **Star Ultimate Model Converter** | Convert a diffusion model / text encoder / VAE / AIO checkpoint to any supported format, with architecture-aware layer protection |
| ⭐ **Star Model Layers Info** | Per-layer report of storage format, quantization config, parameter count and size; optional profile export |
| ⭐ **Starnodes AIO Splitter** | Split an all-in-one checkpoint into diffusion model + text encoder + VAE |
| ⭐ **Star Ultimate Model Converter Pro** | Profile-driven mixed-precision quantization with strict conversion rules |
| ⭐ **Star Quant Key Fixer** | Repair quantization metadata / tensor-key prefixes in converted files |
| ⭐ **Star Checkpoint Saver (AIO)** | Save model + CLIP + VAE as one AIO checkpoint (mixed quantizations supported) |
| ⭐ **Starnodes AIO Saver** | Assemble an AIO checkpoint from files already on disk |

<img width="854" height="296" alt="Star Ultimate Model Converter" src="https://github.com/user-attachments/assets/a290dc38-7f4f-4977-a34d-ec8859531dc7" />

---

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Starnodes2024/comfyui-starnodes-modelconverter
pip install -r comfyui-starnodes-modelconverter/requirements.txt
```

Restart ComfyUI. All nodes appear under the **⭐StarNodes** category.

## Requirements

- **ComfyUI** — recent version. ConvRot formats (`int8_convrot`, `int4_convrot`, `w8a16_convrot`)
  need ComfyUI with `comfy_quant` per-module support.
- **comfy-kitchen** — the quantization library. Ships with recent ComfyUI; otherwise
  `pip install comfy-kitchen`. `w6a8_convrot` additionally needs a build containing
  [comfy-kitchen#191](https://github.com/Comfy-Org/comfy-kitchen) (the `bits` parameter); the node
  probes for it and refuses with a clear message rather than misquantizing.
- **PyTorch 2.0+**.
- **GPU** — CUDA or ROCm. `device: cuda` is 10–50× faster than `cpu`; every format also converts on
  `cpu` if VRAM is the bottleneck. Native NVFP4 *compute* is NVIDIA-oriented, but NVFP4
  *conversion* works on AMD (fixed in v1.5.8).

### Memory

The converter holds the full input state dict **and** the accumulating output in system RAM, so
budget roughly **1.6× the source file size**:

| Source | RAM needed |
|---|---|
| 10 GB bf16 | ~16 GB |
| 20 GB bf16 | ~32 GB |
| 40 GB bf16 | ~64 GB |

64 GB+ RAM and a large pagefile (100 GB+ on Windows) are recommended for 20 GB+ models. If VRAM is
short, convert on `cpu` — slower, same result.

---

## Quick start

1. Add **⭐ Star Ultimate Model Converter**.
2. `mode` → `Diffusion Model`, `diffusion_model` → your file, `model_type` → the architecture profile.
3. `target_format` → e.g. `int8_convrot` or `w8a16_convrot`, `device` → `cuda`.
4. Run. The converted file lands next to the source.
5. Add **⭐ Star Model Layers Info** on the result to confirm the layer formats.

---

## ⭐ Star Ultimate Model Converter

### Inputs

| Widget | Meaning |
|---|---|
| `mode` | `Diffusion Model` · `Checkpoint` · `AIO` · `Text-Encoder` · `VAE` · `Custom Path` |
| `diffusion_model` | File from `models/diffusion_models` (used when `mode` = Diffusion Model) |
| `checkpoint` | File from `models/checkpoints` (Checkpoint / AIO modes) |
| `text_encoder` | File from `models/text_encoders` (Text-Encoder mode) |
| `vae` | File from `models/vae` (VAE mode) |
| `model_type` | Architecture profile — 45 available, see [Supported models](#supported-models) |
| `target_format` | See [Target formats](#target-formats) |
| `device` | `cuda` (fast) or `cpu` (works with any VRAM budget) |
| `custom_path` | *(optional)* File, shard folder or HF snapshot dir, used with `mode` = Custom Path. Must be inside a ComfyUI model folder (path-traversal guard) |
| `svdquant_rank` / `svdquant_refine_iters` | *(optional)* `svdquant_w4a4` only — low-rank branch size and refinement passes |
| `gguf_arch` | *(optional)* Architecture string written into the GGUF header (must be in ComfyUI-GGUF's whitelist) |
| `llama_quantize_bin` | *(optional)* Path to `llama-quantize` for K-quant GGUF targets |
| `keep_intermediate_gguf` | *(optional)* Keep the F16/BF16 intermediate GGUF |
| `learned_rounding` | *(optional)* Refine FP8 / INT8 ConvRot rounding by gradient descent on the weight's own top-k SVD subspace (calibration-free, **much** slower) |
| `learned_rounding_iters` / `_lr` / `_topk_ratio` | *(optional)* Learned-rounding controls (10–5000 steps, AdamW lr, subspace ratio) |

### Modes

- **Diffusion Model** — converts `models/diffusion_models/<file>` in place (output beside it).
- **Checkpoint** — extracts the diffusion model from an AIO checkpoint *without* the text encoder/VAE.
- **AIO** — converts UNet + CLIP together, ignores the VAE, saves to `models/checkpoints`.
- **Text-Encoder** — converts a CLIP/T5/Qwen text encoder.
- **VAE** — converts a VAE.
- **Custom Path** — file, shard folder (`*.safetensors` + optional `model.safetensors.index.json`) or
  HuggingFace snapshot directory.

Sharded inputs are validated against their index before merging, so an incomplete download or a
folder holding two different models is refused instead of silently producing a broken file.

### Target formats

| `target_format` | Bits | Storage | Notes |
|---|---|---|---|
| `nvfp4` | 4 | FP4 E2M1 + FP8 block scales (16) | smallest; NVIDIA-native compute |
| `fp8` | 8 | FP8 E4M3FN + scalar scale | broad support |
| `mxfp8` | 8 | FP8 + E8M0 block scales (32) | OCP microscaling, near-FP16 quality |
| `int8` | 8 | INT8 + scalar scale | most compatible 8-bit |
| `int8_convrot` | 8 | INT8 + per-row scale + Hadamard GS:256 | best 8-bit quality/speed balance |
| **`w8a16`** | 8 | INT8 + scalar scale, bf16 matmul | weight-only 8-bit |
| **`w8a16_convrot`** | 8 | INT8 + per-row scale + GS:256, bf16 matmul | weight-only 8-bit, high quality |
| `int4_convrot` | 4 | INT4 + group scales + rotation GS:256/QGS:64 | ~25% size |
| `w4a8_convrot` | 4 | packed 4-bit codes + FP8 per-group scale | needs `AsymW4A8Int8Layout` |
| `w6a8_convrot` | 6 | packed 6-bit codes + FP8 per-group scale | needs comfy-kitchen#191 |
| `awq_w4a16` | 4 | 4-bit + scale/zero (group 64) | weight-only 4-bit |
| `w4a4` | 4 | INT4 weights + INT4 activations | fastest 4-bit path |
| `svdquant_w4a4` | 4 + bf16 | low-rank bf16 branch + INT4 residual | best 4-bit quality, slowest |
| `minimax_h3_native_mix` | mixed | INT8 ConvRot `attn.qkv_proj`, NVFP4 MLP, BF16 boundaries | MiniMax-H3 official recipe |
| `fp16` / `fp32` | 16 / 32 | plain float | no compression |
| `gguf_*` | — | `gguf_f16/bf16/q8_0/q5_1/q5_0/q4_1/q4_0` and `q3_k_s/m`, `q4_k_s/m`, `q5_k_s/m`, `q6_k` | K-quants need `llama-quantize` |

`int8_convrot_pruned`, `int4_convrot_pruned`, `w4a8_convrot_pruned`, `w6a8_convrot_pruned` and
`w8a16_convrot_pruned` apply the profile's extra "pruned model" blacklist.

### Output

Saved beside the source with the format appended, existing precision suffixes stripped:

```
flux-dev-fp16.safetensors   →   flux-dev-int8_convrot.safetensors
moss_image_sft_bf16.safetensors → moss_image_sft-w8a16_convrot.safetensors
```

The status output lists input/output sizes, compression ratio, per-format layer counts, device,
duration and the output path.

---

## ⭐ Star Model Layers Info

Reads a `.safetensors` or `.gguf` file and reports every layer: storage format, quantization config
(including ConvRot group size and the W8A16 flag), parameter count and size.

- **Normal View** — flat list.
- **Tree View** — grouped by layer ranges, e.g. `[0-27]`.
- **save_profile** — writes `profiles/<model>.json`, usable with Converter Pro.

Writes `output/modelinfo/<model>_<view>.txt`. Labels look like `INT8_CONVROT (GS:256)`,
`INT8_CONVROT (GS:256) [W8A16]`, `NVFP4 (GS:16)`, `W4A8 (QGS:16)`, `BF16`.

## ⭐ Starnodes AIO Splitter

Splits an all-in-one checkpoint into ComfyUI's folders:

| Widget | Result |
|---|---|
| `save_model` | `models/diffusion_models/<name>_model.safetensors` |
| `save_text_encoder` | `models/text_encoders/<name>_clip.safetensors` |
| `save_vae` | `models/vae/<name>_vae.safetensors` |

Missing components are skipped with a warning rather than written empty.

## ⭐ Star Quant Key Fixer

Repairs converted files whose tensor keys or quantization metadata don't match what ComfyUI's
loader expects. Modes: report-only, fix keys, fix metadata; `verify_bytes` checks the rewrite
tensor-by-tensor and `overwrite` controls whether the original is replaced.

## ⭐ Star Checkpoint Saver (AIO) / Starnodes AIO Saver

`Star Checkpoint Saver (AIO)` takes a loaded `MODEL` + `CLIP` + `VAE` (and optional `CLIP_VISION`)
and writes a single AIO checkpoint — mixed quantizations are fine (NVFP4 model + NVFP4 text encoder
+ BF16 VAE). `Starnodes AIO Saver` assembles one from files already in your model folders, with
configurable key prefixes.

## ⭐ Star Ultimate Model Converter Pro

Profile-driven mixed-precision conversion with explicit rules instead of a single target format:
**blacklist → manual mode → profile → fallback**. Use it when you want some layers in BF16, some in
NVFP4 and some in INT4 within one file. `target_quant_format` accepts `NVFP4`, `INT4_CONVROT`,
`FP8`, `INT8`, `INT8_CONVROT`, `MXFP8`.

---

## Supported models

45 profiles ship in `models.json`:

- **Flux**: Flux1 / Flux2, Flux 2 Tight, Flux 3 Klein, Flux 3 Klein MaxQ
- **Qwen**: Qwen-Image, Qwen-Image W4A8, Qwen-Image-2.1, Qwen-Image-2.1 MaxQ
- **Z-Image**: Z-Image, Z-Image MaxQ
- **Krea**: Krea-2 · **Wan**: Wan (All Versions) · **LTX**: LTX-Video, LTX-2.5, `ltx25_gemma4_te`, `ltx25_vae`
- **MiniMax-H3**: `minimax_h3`, `minimax_h3_ref_nvfp4_fp8`, `minimax_h3_ref_nvfp4_int8convrot`,
  `minimax_h3_int4_tensorwise_experimental`, `minimax_h3_int8convrot_int4fc2`,
  `minimax_h3_int8convrot_int4mlp`, `minimax_h3_te`, `minimax_h3_vae`
- **MOSS**: MOSS-Image, MOSS-Image MaxQ · **Fibo**: Fibo-2.0-edit-turbo, HQ, MaxQ
- **SeedVR**: SeedVR (All Versions), SeedVR MaxQ · **Ideogram-4** · **Chroma** · **Anima**
- **Ming-Image** · **ERNIE-Image** · **Boogu-Image** · **Lens** · **ACE-Step** · **YuE2**
- **SDXL (Not NVFP4)**
- **Text encoders**: Text-Encoder, Text-Encoder HQ, Text-Encoder MaxQ
- **Unknown** — generic profile, quantizes every 2D `.weight` it safely can

Each profile is a blacklist: embedders, norms, modulators, final layers and architecture-specific
critical paths stay BF16 while the rest is quantized.

---

## Format comparison

| Format | Size | Quality | Compatibility | Speed |
|--------|------|---------|---------------|-------|
| NVFP4 | ★★★★★ | ★★★★☆ | NVIDIA-native compute | ★★★★★ |
| FP8 | ★★★★☆ | ★★★★☆ | Broad | ★★★★☆ |
| MXFP8 | ★★★★☆ | ★★★★★ | Growing (OCP) | ★★★★★ |
| INT8 | ★★★☆☆ | ★★★☆☆ | Most hardware | ★★★☆☆ |
| INT8 ConvRot | ★★★☆☆ | ★★★★☆ | ComfyUI with `comfy_quant` | ★★★☆☆ |
| W8A16 ConvRot | ★★★☆☆ | ★★★★☆ | ComfyUI with `comfy_quant` | ★★☆☆☆ |
| INT4 ConvRot | ★★★★★ | ★★★★☆ | SM 8.0+ / ROCm | ★★★★☆ |
| W4A8 / W6A8 ConvRot | ★★★★★ | ★★★★☆ | comfy-kitchen with `AsymW4A8Int8Layout` | ★★★★☆ |
| SVDQuant W4A4 | ★★★★★ | ★★★★★ | comfy-kitchen, needs `TensorCoreConvRotW4A4Layout` | ★★☆☆☆ |
| FP16 | ★★☆☆☆ | ★★★★★ | All hardware | ★★★★☆ |
| FP32 | ★☆☆☆☆ | ★★★★★ | All hardware | ★★☆☆☆ |

---

## How the quantization is stored

Converted files are ordinary safetensors whose tensor names are unchanged. Each quantized module
carries a `<module>.comfy_quant` tensor holding UTF-8 JSON, plus its scale tensors:

```
layers.0.proj.qkv.weight          int8   [N, K]
layers.0.proj.qkv.weight_scale    bf16   [N, 1]   (per-row) or scalar
layers.0.proj.qkv.comfy_quant     uint8  {"format": "int8_tensorwise", "convrot": true,
                                          "convrot_groupsize": 256,
                                          "full_precision_matrix_mult": true}
```

The older file-level `_quantization_metadata` header is written alongside for tools that predate
per-module entries; ComfyUI prefers the per-module ones. See ComfyUI's
[QUANTIZATION.md](https://github.com/Comfy-Org/ComfyUI/blob/master/QUANTIZATION.md) for the full
format specification.

**Re-conversion safety.** Loading an already-quantized model triggers automatic dequantization, but
formats that cannot be dequantized losslessly (NVFP4, MXFP8, W4A4, AWQ W4A16, ConvRot-rotated INT8)
are refused, so you get an error instead of a silently double-quantized file.

**File naming.** Existing precision suffixes are stripped before the new format is appended, so
re-converting `model-w8a16_convrot.safetensors` to INT8 gives `model-int8_convrot.safetensors`, not a
stacked chain.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `comfy-kitchen is required for this target format` | `pip install comfy-kitchen`, restart ComfyUI |
| `W6A8 requires a comfy-kitchen build whose ... accepts a 'bits' parameter` | Update comfy-kitchen to a build containing comfy-kitchen#191 |
| `Refusing to convert a partial model` | The shard set doesn't match its index — re-download the missing shard |
| `cannot be dequantized losslessly` | The source is already quantized; convert from a bf16/fp16 source instead |
| Out of memory during conversion | `device: cpu`, more pagefile/swap, or a smaller source |
| Output looks worse than expected | Check `model_type` matches the architecture; try `int8_convrot` → `w8a16_convrot` → `fp16` in that order |
| Converting on AMD | NVFP4 conversion is fixed for AMD since v1.5.8; INT8/ConvRot use comfy-kitchen's HIP backend |

## Performance tips

1. `device: cuda` — 10–50× faster than CPU.
2. Close ComfyUI's loaded models (and any local LLM) during big conversions; RAM is the usual limit, not VRAM.
3. Convert straight from a shard folder — no need to merge first.
4. Use Layers Info to verify one converted model per format before batch-converting a library.

---

## Contributing

1. Test the model with the **Unknown** profile and note which layers must stay high precision.
2. Add a profile to `models.json` with a tuned blacklist.
3. Verify with Layers Info (`BF16` on the sensitive rows, target format elsewhere).
4. Open a PR with the profile and a short note on what you checked.

## Credits

- **[Comfy-Org](https://github.com/Comfy-Org)** for
  [comfy-kitchen](https://github.com/Comfy-Org/comfy-kitchen) — NVFP4, FP8, MXFP8, INT8/INT4 ConvRot,
  W4A8/W6A8, AWQ W4A16 and SVDQuant layouts, plus the HIP backend used on AMD.
- Community PRs from **[@aksugat](https://github.com/aksugat)** — ConvRot layer display fix, quant key
  fixer, and the W8A16 targets in v1.7.
- The ComfyUI community for testing and feedback.

## License

MIT — see [LICENSE](LICENSE). Respect the licenses of the models you convert.

---

## Changelog

### v1.7
- 🆕 **W8A16 targets** — `w8a16`, `w8a16_convrot`, `w8a16_convrot_pruned`: INT8 weights, bf16 matmul
  via `full_precision_matrix_mult`. Same size as INT8 ConvRot, no activation-quantization error,
  ~2× matmul cost. Verified on ROCm and through ComfyUI's loader.
- 🔍 Layers Info labels W8A16 layers with a `[W8A16]` marker.
- 🛡️ Hardened sharded-input merging: shard sets are validated against their index, and folders
  holding several models or an incomplete download are refused instead of merged.
- 📖 README rebuilt: accurate widget reference for the current node, complete format table, real
  memory guidance.

### v1.6
- Fibo-2.0-edit-turbo profiles (standard, HQ, MaxQ).
- MOSS-Image profiles; NVFP4 conversion fixed on AMD GPUs.
- Flux 3 Klein and Flux 3 Klein MaxQ profiles.

### v1.5
- W6A8 ConvRot support (needs comfy-kitchen#191) incl. pruned profile.
- Profiles: Ming-Image, Qwen-Image-2.1 MaxQ, Z-Image MaxQ, Flux 3, plus two maximum-quality
  text-encoder profiles.
- Star Checkpoint Saver (AIO); learned-rounding fixes; AMD NVFP4 fix; W4A8/SVDQuant/AWQ W4A16 and
  Minimax-H3 native-mix targets; layers-info metadata prefix matching.

### v1.3.0 (2026-07-14)
- Pro converter enhanced: `models.json` blacklist integration with 15+ profiles.
- Manual mode for per-dtype quantization control.
- Real NVFP4 / INT4 ConvRot / MXFP8 quantization through comfy-kitchen.
- Priority system: blacklist → manual mode → profile → fallback.
- Format normalization for complex profile values (`FP8_E4M3FN + SCALE`).

### v1.2.0 (2025-01-13)
- New nodes: Star Model Layers Info, Star Ultimate Model Converter Pro.
- Profile system: extract a blueprint from an optimized model and apply it to others.
- Normal/Tree layer views, hover tooltips, profile metadata API endpoint.

### v1.1.0 (2025-01-10)
- INT4 ConvRot support (~25% size), requires SM 8.0+.
- RAM/pagefile guidance for large models; updated format comparison.

### v1.0.0
- Initial release: NVFP4, FP8, MXFP8, INT8, INT8 ConvRot, AIO splitter, architecture profiles,
  automatic dequantization, metadata preservation.
