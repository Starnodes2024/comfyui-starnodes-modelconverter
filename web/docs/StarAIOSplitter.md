# ⭐ Star AIO Splitter

Split all-in-one checkpoints into separate diffusion model, text encoder, and VAE files.

## Purpose

The Star AIO Splitter extracts individual components from all-in-one checkpoints and saves them to the appropriate ComfyUI model folders. This is useful for:
- Using only specific components
- Converting individual components separately
- Managing model components independently
- Reducing memory usage by loading only needed parts

## Inputs

### Required

- **checkpoint_name**: Select an AIO checkpoint from your `models/checkpoints` folder

### Optional

- **save_model**: Extract and save the diffusion model (default: True)
- **save_text_encoder**: Extract and save the text encoder/CLIP (default: False)
- **save_vae**: Extract and save the VAE (default: False)

## Outputs

- **status**: Summary of extracted components with sizes and locations
- **MODEL** (optional): Loaded diffusion model if extracted
- **CLIP** (optional): Loaded text encoder if extracted
- **VAE** (optional): Loaded VAE if extracted

## How It Works

1. **Load Checkpoint**: Reads the all-in-one checkpoint file
2. **Detect Components**: Identifies model, CLIP, and VAE tensors by key prefixes
3. **Extract**: Separates tensors for each selected component
4. **Save**: Writes components to appropriate folders:
   - Diffusion model → `models/diffusion_models/`
   - Text encoder → `models/text_encoders/`
   - VAE → `models/vae/`
5. **Load**: Optionally loads extracted components for output connectors

## Component Detection

The splitter uses key prefixes to identify components:

- **Diffusion Model**: `model.diffusion_model.*`, `double_blocks.*`, `single_blocks.*`
- **Text Encoder**: `cond_stage_model.*`, `conditioner.*`, `text_encoders.*`
- **VAE**: `first_stage_model.*`, `vae.*`

## Output Files

Files are saved with component suffixes:

```
Input:  my-checkpoint.safetensors
Output: my-checkpoint_model.safetensors  (diffusion model)
        my-checkpoint_clip.safetensors   (text encoder)
        my-checkpoint_vae.safetensors    (VAE)
```

## Example Workflow

### Extract All Components

```
[Star AIO Splitter]
  - checkpoint: SDXL_Pony_godiva_v10.safetensors
  - save_model: ✓ True
  - save_text_encoder: ✓ True
  - save_vae: ✓ True
  
Output:
  → models/diffusion_models/SDXL_Pony_godiva_v10_model.safetensors (5.12 GB)
  → models/text_encoders/SDXL_Pony_godiva_v10_clip.safetensors (492 MB)
  → models/vae/SDXL_Pony_godiva_v10_vae.safetensors (335 MB)
```

### Extract Model Only

```
[Star AIO Splitter]
  - checkpoint: flux-dev-aio.safetensors
  - save_model: ✓ True
  - save_text_encoder: ☐ False
  - save_vae: ☐ False
  
Output:
  → models/diffusion_models/flux-dev-aio_model.safetensors
```

## Use Cases

### 1. Convert Model Only
```
[Star AIO Splitter] → [Star Ultimate Model Converter]
  - save_model: True     - model: extracted_model
  - others: False        - target_format: nvfp4
```

### 2. Use Components Separately
```
[Star AIO Splitter]
  - save_model: True
  - save_text_encoder: True
  - save_vae: True
  ↓
[Load Diffusion Model] + [Load CLIP] + [Load VAE]
```

### 3. Replace VAE
```
[Star AIO Splitter] → Extract model + CLIP
[Load VAE] → Use custom VAE
[Star AIO Saver] → Merge with new VAE
```

## Tips

- **Selective Extraction**: Only enable components you need to save time and disk space
- **Output Connectors**: Use the MODEL/CLIP/VAE outputs to immediately use extracted components
- **Disk Space**: Ensure sufficient space (AIO checkpoint size + component sizes)
- **Component Detection**: If a component isn't found, it won't be saved (warning shown)

## Component Sizes

Typical sizes for different model types:

**SDXL Models**:
- Diffusion Model: ~5-6 GB
- Text Encoder: ~400-500 MB
- VAE: ~300-400 MB

**Flux Models**:
- Diffusion Model: ~12-24 GB
- Text Encoder: ~8-10 GB
- VAE: ~300-400 MB

## Status Output

The node provides detailed status:

```
✂️ Split: SDXL_Pony_godiva_v10.safetensors (5.98 GB)
✅ model: 1690 tensors, 5.12 GB → E:\...\diffusion_models\..._model.safetensors
✅ clip: 198 tensors, 492.34 MB → E:\...\text_encoders\..._clip.safetensors
✅ vae: 248 tensors, 334.64 MB → E:\...\vae\..._vae.safetensors
Time: 12.3s
```

## Troubleshooting

### "No components could be extracted"
- Verify the checkpoint is a valid AIO checkpoint
- Check if the file contains model/CLIP/VAE tensors
- Try a different checkpoint

### Component not found
- Some checkpoints may not include all components
- Check the status message for warnings
- Verify the checkpoint structure

### Out of memory
- Close other applications
- Extract components one at a time
- Use a system with more RAM

## Requirements

- ComfyUI (any recent version)
- Sufficient disk space for extracted components
- RAM: 16GB+ recommended for large checkpoints

## Credits

Part of the Starnodes Model Converter suite for comprehensive model management.
