# Helios Action-Interactive World Model

This document describes the migration of YUME 1.5's action-interactive world model capabilities to the Helios codebase, enabling keyboard (WASD) and mouse control for real-time video generation.

## Overview

This implementation adds action-conditioned video generation to Helios, allowing the model to generate first-person exploration videos controlled by discrete keyboard and mouse inputs. The approach leverages Helios's existing cross-attention mechanism by encoding actions as T5 embeddings and concatenating them with text prompt embeddings -- requiring zero model architecture changes.

### Key Design Decisions

- **Action conditioning via T5 embeddings**: Discrete actions (9 keyboard x 9 mouse = 81 combinations) are mapped to natural language descriptions and pre-encoded through T5. This reuses YUME's validated vocabulary approach.
- **Separate action/scene embeddings**: Actions are concatenated to `encoder_hidden_states` alongside the scene prompt, keeping them decoupled. This allows per-chunk action switching at inference time without re-encoding T5.
- **Zero architecture changes**: Helios's Guidance Cross-Attention naturally handles variable-length `encoder_hidden_states` and restricts text/action influence to the noisy context only (not history).

## Architecture

```
Input: Scene Description + Keyboard (WASD) + Mouse (arrows)
  |
  |  [Offline Cache: action_embeds_cache.pt]
  |  81 pre-encoded T5 embeddings for all action combos
  |
  v
torch.cat([scene_prompt_embed, action_embed], dim=seq_len)
  |
  v
encoder_hidden_states  -->  Guidance Cross-Attention
                              |
                              v  (only QNoisy, not QHist)
                          Denoised output
```

## File Changes Summary

### New Files

| File | Description |
|------|-------------|
| `tools/offload_data/convert_sekai_to_helios.py` | Converts YUME Sekai dataset (mp4 + txt) to Helios .pt format with action labels |
| `tools/offload_data/precompute_action_embeds.py` | Pre-encodes all 81 action T5 embeddings into a cache file |
| `scripts/training/configs/stage_1_action_init.yaml` | Stage 1 training config for action world model with LoRA |
| `example/action_interactive.csv` | Example CSV with per-chunk action labels for interactive inference |

### Modified Files

| File | Change |
|------|--------|
| `helios/dataset/dataloader_history_latents_dist.py` | Added `action_keys` and `action_mouse` fields to output dict |
| `helios/utils/train_config.py` | Added `action_embeds_cache_path` to `DataConfig` |
| `train_helios.py` | Loads action cache at startup; concatenates action embeds to `prompt_embeds` during Stage 1 training |
| `helios/pipelines/pipeline_helios.py` | Added `action_embeds_list` parameter for per-chunk action conditioning in generation loop |
| `infer_helios.py` | Added `--action_embeds_cache`, `--action_keys`, `--action_mouse` CLI args; parses action columns from interactive CSV |

## Usage Guide

### Step 1: Prepare Action Embedding Cache

Run once to create the 81-entry T5 embedding cache:

```bash
python tools/offload_data/precompute_action_embeds.py \
    --pretrained_model_name_or_path BestWishYsh/Helios-Base \
    --output_path data/action_embeds_cache.pt
```

### Step 2: Convert YUME Sekai Data to Helios Format

Convert the YUME mp4_frame dataset to Helios .pt files:

```bash
torchrun --nproc_per_node 8 tools/offload_data/convert_sekai_to_helios.py \
    --sekai_root /path/to/mp4_frame \
    --output_dir data/sekai_helios_latents \
    --pretrained_model_name_or_path BestWishYsh/Helios-Base \
    --target_height 384 \
    --target_width 640 \
    --max_frames 121
```

**Input format** (YUME Sekai):
```
mp4_frame/
├── Keys_W_Mouse_→/
│   ├── video_001_frames_00000-00048.mp4
│   ├── video_001_frames_00000-00048.txt    # Keys: W\nMouse: →
│   └── video_001_frames_00000-00048.npy    # (optional) camera c2w
├── Keys_W+A_Mouse_·/
│   └── ...
```

**Output format** (Helios .pt):
```
sekai_helios_latents/
├── video_001_frames_00000-00048_121_384_640.pt
│   ├── vae_latent:   (num_chunks, 16, 9, 48, 80)
│   ├── prompt_embed: (seq_len, 4096)
│   ├── prompt_raw:   str
│   ├── action_keys:  "W"
│   └── action_mouse: "→"
```

### Step 3: Train

```bash
accelerate launch --config_file scripts/accelerate_configs/zero2.json \
    train_helios.py \
    --config scripts/training/configs/stage_1_action_init.yaml
```

Key config options in `stage_1_action_init.yaml`:
- `data_config.instance_data_root`: Path to converted .pt files
- `data_config.action_embeds_cache_path`: Path to action_embeds_cache.pt
- `training_config.learning_rate`: 5e-5 (LoRA fine-tuning)
- `training_config.max_train_steps`: 10000

### Step 4: Inference

**Single action mode** (same action for all chunks):

```bash
python infer_helios.py \
    --base_model_path BestWishYsh/Helios-Base \
    --transformer_path /path/to/action_checkpoint \
    --sample_type i2v \
    --image_path example/input.jpg \
    --prompt "A first-person view of walking through a forest." \
    --action_embeds_cache data/action_embeds_cache.pt \
    --action_keys W \
    --action_mouse → \
    --num_frames 99 \
    --output_folder output_action
```

**Per-chunk action mode** (different actions per chunk, via CSV):

```bash
python infer_helios.py \
    --base_model_path BestWishYsh/Helios-Base \
    --transformer_path /path/to/action_checkpoint \
    --interactive_prompt_csv_path example/action_interactive.csv \
    --action_embeds_cache data/action_embeds_cache.pt \
    --num_frames 297 \
    --interpolate_time 3 \
    --output_folder output_action_interactive
```

**CSV format** (`example/action_interactive.csv`):

```csv
id,prompt_index,prompt,action_keys,action_mouse
1,0,"Scene description...",W,→
1,1,"Scene description...",W+A,·
1,2,"Scene description...",None,←
```

## Action Vocabulary

### Keyboard (9 options)

| Key | Description |
|-----|-------------|
| `W` | Person moves forward (W). |
| `A` | Person moves left (A). |
| `S` | Person moves backward (S). |
| `D` | Person moves right (D). |
| `W+A` | Person moves forward and left (W+A). |
| `W+D` | Person moves forward and right (W+D). |
| `S+A` | Person moves backward and left (S+A). |
| `S+D` | Person moves backward and right (S+D). |
| `None` | Person stands still. |

### Mouse (9 options)

| Key | Description |
|-----|-------------|
| `→` | Camera turns right (→). |
| `←` | Camera turns left (←). |
| `↑` | Camera tilts up (↑). |
| `↓` | Camera tilts down (↓). |
| `↑→` | Camera tilts up and turns right (↑→). |
| `↑←` | Camera tilts up and turns left (↑←). |
| `↓→` | Camera tilts down and turns right (↓→). |
| `↓←` | Camera tilts down and turns left (↓←). |
| `·` | Camera remains still (·). |

## Technical Details

### How Action Conditioning Flows Through the Model

1. **Pre-compute**: All 81 action combos are T5-encoded and cached offline
2. **Training**: For each batch, action embeds are looked up from cache and concatenated to `prompt_embeds` along the sequence dimension before the transformer forward pass
3. **Inference**: Per-chunk action embeds are selected from cache and concatenated to `prompt_embeds` inside the generation loop
4. **Guidance Cross-Attention**: The concatenated `[scene_embed | action_embed]` serves as KText/VText, attending only to QNoisy (current chunk), not QHist (history) -- action only influences the currently generated frames

### Why This Works Without Architecture Changes

Helios's `HeliosTransformer3DModel` accepts `encoder_hidden_states` with variable sequence length. The cross-attention layers compute:

```
XCross = Attention(QNoisy, KText, VText)
```

Where KText/VText derive from `encoder_hidden_states`. Concatenating action embeddings simply extends the KV sequence, which standard multi-head attention handles natively. The Guidance Attention design ensures action information reaches only the noisy context (current generation), not the historical context.

### Comparison with YUME 1.5

| Aspect | YUME 1.5 | Helios + Action |
|--------|----------|-----------------|
| Base model | Wan2.2-5B | Wan2.1-14B |
| Action method | Text baked into caption | Separate T5 embeddings concatenated |
| Anti-drifting | Self-Forcing (expensive) | Frame-Aware Corrupt + Relative RoPE (efficient) |
| History compression | TSCM (5-level adaptive) | Multi-Term Memory (3-level fixed) |
| Inference speed | 12 FPS @540P A100 | 19.5 FPS @H100 |
| Distillation | 50 to 4 steps | 50 to 3 steps |
| Per-chunk action switch | Limited (re-encode T5) | Instant (cached lookup) |
