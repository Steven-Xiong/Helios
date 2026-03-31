# Helios Interactive World Model — 数据处理全流程文档

## 目录

1. [总览](#1-总览)
2. [原始数据格式](#2-原始数据格式)
3. [Step 1: 预计算 Action Embedding Cache](#3-step-1-预计算-action-embedding-cache)
4. [Step 2: 视频 → VAE Latent + Text Embedding](#4-step-2-视频--vae-latent--text-embedding)
5. [Step 3: 生成 ODE 训练对（Stage 3 专用）](#5-step-3-生成-ode-训练对stage-3-专用)
6. [Step 4: 文本 Embedding（可选，Self-Forcing 用）](#6-step-4-文本-embedding可选self-forcing-用)
7. [预处理产物汇总](#7-预处理产物汇总)
8. [训练时数据加载](#8-训练时数据加载)
9. [端到端数据流图](#9-端到端数据流图)
10. [常见问题](#10-常见问题)

---

## 1. 总览

整个数据处理流程将原始视频 + 动作标注转换为训练可直接使用的预编码 tensor 文件（`.pt`），从而避免训练时实时编码的 GPU/IO 开销。

```
原始数据                    预处理                           训练数据
─────────               ──────────────                   ──────────
MP4 视频                                                  ┌─────────────┐
Action 标注 ──→  VAE Encode + T5 Encode  ──→              │ .pt 文件     │
文本 Caption           (离线批量)                          │  vae_latent  │
                                                          │  prompt_embed│
                                                          │  chunk_actions│
Action 词表   ──→  T5 Encode (81 组合)   ──→              │ action_embeds│
                                                          │  _cache.pt   │
                                                          └─────────────┘
Teacher 模型  ──→  Pipeline 推理采样     ──→              ode_pairs/*.pt
```

各步骤对应脚本与适用训练阶段：

| 步骤 | 脚本 | 产物 | 适用阶段 |
|------|------|------|----------|
| Action Embedding Cache | `precompute_action_embeds.py` | `action_embeds_cache.pt` | 所有阶段 |
| VAE Latent 编码（YUME 数据） | `convert_sekai_to_helios.py` | `{uttid}_{frames}_{h}_{w}.pt` | Stage 1, 2, 3(GAN) |
| VAE Latent 编码（通用短视频） | `get_short-latents.py` | `{uttid}_{frames}_{h}_{w}.pt` | Stage 1, 2, 3(GAN) |
| VAE Latent 编码（通用长视频） | `get_long-latents.py` | `{uttid}_{frames}_{h}_{w}.pt` | Stage 1, 2, 3(GAN) |
| ODE 训练对生成 | `get_ode-pairs.py` | `{uttid}.pt` | Stage 3 |
| 文本 Embedding | `get_text-embedding.py` | `{uttid}.pt` | Stage 3 (Self-Forcing) |

---

## 2. 原始数据格式

### 2.1 通用视频数据（JSON 格式）

适用于 `get_short-latents.py` / `get_long-latents.py`，需准备 JSON 元数据 + 视频文件：

```json
[
    {
        "cut": [0, 81],
        "crop": [0, 832, 0, 480],
        "fps": 24.0,
        "num_frames": 81,
        "resolution": {"height": 480, "width": 832},
        "cap": ["A stunning mid-afternoon scene..."],
        "path": "videos/2_240_ori81.mp4"
    }
]
```

目录结构：

```
data/
├── toy_data/
│   ├── videos/
│   │   ├── 2_240_ori81.mp4
│   │   └── ...
│   └── toy_filter.json
```

### 2.2 YUME / Sekai 数据（动作标注格式）

适用于 `convert_sekai_to_helios.py`，数据包含完整视频和切分好的 33 帧动作片段：

```
data/seadance2_yume/
├── video/                          ← 原始完整视频（如 361 帧）
│   ├── rotate_0096.mp4
│   └── ...
├── mp4_frame/                      ← 按动作类别组织的 33 帧片段
│   ├── Keys_W_Mouse_→/
│   │   ├── rotate_0096_..._frames_00000-00032.mp4
│   │   └── ...
│   ├── Keys_S_A_Mouse_Down_Right/
│   │   └── ...
│   └── Keys_None_Mouse_·/
│       └── ...
└── world_model_action12_train_1200_v2.tsv   ← action_class + caption
```

**目录名编码规则**：`Keys_{键盘}_Mouse_{鼠标方向}` → 解析为 `(keys, mouse)` 标签：

| 目录名中的键盘部分 | 解析结果 | 目录名中的鼠标部分 | 解析结果 |
|---|---|---|---|
| `W` | `"W"` | `Right` | `"→"` |
| `S_A` | `"S+A"` | `Down_Right` | `"↓→"` |
| `None` | `"None"` | `·` | `"·"` |

**TSV 格式**（可选，提供 per-video 的 caption）：

```
action_class	prompt	video_name
outdoor_walk	This video depicts a first-person view...	rotate_0096.mp4
```

---

## 3. Step 1: 预计算 Action Embedding Cache

### 3.1 Action 词表

我们定义了 9 种键盘动作 × 9 种鼠标动作 = **81 种组合**：

**键盘动作（VOCAB_KEYBOARD）**：

| Key | 自然语言描述 |
|-----|-------------|
| `W` | Person moves forward (W). |
| `A` | Person moves left (A). |
| `S` | Person moves backward (S). |
| `D` | Person moves right (D). |
| `W+A` | Person moves forward and left (W+A). |
| `W+D` | Person moves forward and right (W+D). |
| `S+D` | Person moves backward and right (S+D). |
| `S+A` | Person moves backward and left (S+A). |
| `None` | Person stands still (·). |

**鼠标动作（VOCAB_MOUSE）**：

| Mouse | 自然语言描述 |
|-------|-------------|
| `→` | Camera turns right (→). |
| `←` | Camera turns left (←). |
| `↑` | Camera tilts up (↑). |
| `↓` | Camera tilts down (↓). |
| `↑→` | Camera tilts up and turns right (↑→). |
| `↑←` | Camera tilts up and turns left (↑←). |
| `↓→` | Camera tilts down and turns right (↓→). |
| `↓←` | Camera tilts down and turns left (↓←). |
| `·` | Camera remains still (·). |

### 3.2 编码过程

对于每个 `(keys, mouse)` 组合，将两段自然语言拼接后通过 **UMT5 text encoder** 编码：

```
action_text = "Person moves forward (W). Camera turns right (→)."
                        ↓
              UMT5EncoderModel.encode_prompt()
                        ↓
              prompt_embed: tensor(seq_len, 4096)
```

### 3.3 运行命令

```bash
python tools/offload_data/precompute_action_embeds.py \
    --pretrained_model_name_or_path BestWishYSH/Helios-Base \
    --output_path data/helios/action_embeds_cache.pt
```

### 3.4 产物

```
action_embeds_cache.pt
├── ("W", "→")    → tensor(seq_len, 4096)    # bf16
├── ("W", "←")    → tensor(seq_len, 4096)
├── ("None", "·") → tensor(seq_len, 4096)    # 默认 fallback
└── ...  共 81 条
```

训练时，dataloader 返回每个样本的 `(action_keys, action_mouse)` → 查表获取对应 embedding → **拼接到 prompt_embeds 的 sequence 维度**：

```python
prompt_embeds = torch.cat([prompt_embeds, action_embeds_batch], dim=1)
# prompt_embeds: (B, seq_len_scene + seq_len_action, 4096)
```

---

## 4. Step 2: 视频 → VAE Latent + Text Embedding

这是最核心的预处理步骤，将原始视频编码为训练所需的 `.pt` 文件。有两条路径：

### 4.1 路径 A：YUME/Sekai 数据转换（推荐，含动作标注）

**脚本**：`tools/offload_data/convert_sekai_to_helios.py`

**处理流程**：

```
原始视频 (361 帧, 任意分辨率)
         │
         ↓ resize → (384, 640)
         │
         ↓ 归一化: pixel / 127.5 - 1.0
         │
         ↓ 按 33 帧一组切分 (frame_window_size = (9-1)*4+1 = 33)
         │
    ┌────┴────┐  ┌────┴────┐  ┌────┴────┐  ···
    │ Chunk 0 │  │ Chunk 1 │  │ Chunk 2 │
    │ 0~32    │  │ 33~65   │  │ 66~98   │
    └────┬────┘  └────┬────┘  └────┬────┘
         │            │            │
         ↓            ↓            ↓
    VAE.encode()  VAE.encode()  VAE.encode()    ← 逐 chunk 编码
         │            │            │
         ↓            ↓            ↓
    latent (C,9,H',W')  ×  num_chunks
         │
         ↓  (latent - mean) * (1/std)    ← 标准化
         │
         ↓  stack → vae_latent: (num_chunks, C, 9, H', W')
         │
         │  同时: 每个 chunk 取中间帧的 action 标签
         │        → chunk_actions[i] = {"keys": "W", "mouse": "→"}
         │
         │  同时: UMT5 编码 caption
         │        → prompt_embed: (seq_len, 4096)
         │
         ↓
    保存 {uttid}_{num_frames}_{384}_{640}.pt
```

**关键参数**：
- `latent_window_size = 9`：每个 chunk 在 latent 空间的时间维度
- `frame_window_size = 33`：每个 chunk 在像素空间的帧数（temporal 4× 压缩 + 1）
- `target_height = 384, target_width = 640`：统一分辨率
- `batch_size = 4`：VAE 编码时的 chunk batch 大小

**运行命令**：

```bash
torchrun --nproc_per_node 8 tools/offload_data/convert_sekai_to_helios.py \
    --video_dir data/seadance2_yume/video \
    --action_dir data/seadance2_yume/mp4_frame \
    --tsv_path data/seadance2_yume/world_model_action12_train_1200_v2.tsv \
    --output_dir data/helios/seadance2_v2_helios_latents \
    --pretrained_model_name_or_path BestWishYSH/Helios-Base \
    --target_height 384 \
    --target_width 640
```

也支持 Sekai-Project 格式的 CSV 标注：

```bash
torchrun --nproc_per_node 8 tools/offload_data/convert_sekai_to_helios.py \
    --video_dir data/sekai/video \
    --action_dir data/sekai/mp4_frame \
    --caption_csvs data/sekai/annotations.csv \
    --output_dir data/helios/sekai_helios_latents
```

### 4.2 路径 B：通用视频编码（无动作标注）

**短视频编码**（`get_short-latents.py`）— 与路径 A 相同的 33 帧 chunk 切分，但数据源为 JSON + MP4：

```bash
# 修改脚本内的 base_video_path / csv_paths / output_latent_paths 后运行
torchrun --nproc_per_node 8 tools/offload_data/get_short-latents.py \
    --pretrained_model_name_or_path BestWishYSH/Helios-Base
```

**长视频编码**（`get_long-latents.py`）— 整段视频一次 VAE encode（适合 81 帧以内的短视频）：

```bash
torchrun --nproc_per_node 8 tools/offload_data/get_long-latents.py \
    --pretrained_model_name_or_path BestWishYSH/Helios-Base
```

### 4.3 产物 `.pt` 文件结构

```python
{
    # VAE latent —— 核心训练数据
    "vae_latent": tensor(num_chunks, C=16, T=9, H'=48, W'=80),
    #   C=16: VAE latent channels
    #   T=9:  latent_window_size (对应 33 帧)
    #   H'=48, W'=80: 空间 8× 压缩后 (384/8, 640/8)

    # 文本 embedding
    "prompt_embed": tensor(seq_len, 4096),    # UMT5 编码的 caption

    # 原始文本
    "prompt_raw": "This video depicts a first-person view...",

    # 首帧图片（用于 I2V 验证）
    "first_frames_image": tensor(3, 384, 640),  # uint8 或 PIL

    # Action 标签（仅 convert_sekai_to_helios.py 生成）
    "chunk_actions": [
        {"keys": "W",    "mouse": "→"},    # chunk 0
        {"keys": "W+D",  "mouse": "·"},    # chunk 1
        {"keys": "None", "mouse": "←"},    # chunk 2
        ...
    ],

    # 动作类别（仅 convert_sekai_to_helios.py 生成）
    "action_class": "outdoor_walk",
}
```

### 4.4 文件命名规则

```
{video_name}_{total_frames}_{height}_{width}.pt
```

示例：`rotate_0096_361_384_640.pt`

该命名被 dataloader 用于分桶（bucketing）：解析末尾三段数字得到 `(num_frame, height, width)` → 同 bucket 的样本 batch 在一起。

---

## 5. Step 3: 生成 ODE 训练对（Stage 3 专用）

### 5.1 目的

Stage 3 的 ODE regression 需要 teacher 模型的去噪轨迹作为监督信号。通过冻结的 Helios Pipeline 对文本 prompt 做完整推理，保存中间步的 `(latent, timestep)` 对。

### 5.2 处理流程

```
文本 prompts (.txt)
        │
        ↓  UMT5 encode → prompt_embed
        │
        ↓  HeliosPipeline(..., output_type="latent")
        │     ├── 50 步推理（或 20+20+20 金字塔）
        │     ├── chunk-by-chunk 自回归
        │     └── 记录每步的 latents + timesteps
        │
        ↓  子采样关键 timestep
        │     第 1 个 section（更多采样点）:
        │       Stage 0: [998.5, 902.2, 834.0, 783.1]
        │       Stage 1: [742.8, 640.0, 547.2, 463.0]
        │       Stage 2: [385.4, 328.6, 254.0, 151.5]
        │     后续 section（更少采样点）:
        │       Stage 0: [998.5, 834.0]
        │       Stage 1: [742.8, 547.2]
        │       Stage 2: [385.4, 254.0]
        │     + 每段末尾的最终 latent (index=-1)
        │
        ↓  保存
    {uttid}.pt
```

### 5.3 产物 `.pt` 文件结构

```python
{
    "latent_window_size": 9,
    "prompt_raw": "A beautiful sunset over the ocean...",
    "prompt_embed": tensor(1, seq_len, 4096),
    "ode_latents": [
        # section 0 (第一个 chunk)
        [
            # stage 0 (低分辨率)
            {
                "latents":   tensor(5, 1, 16, 9, H', W'),  # 4 采样点 + 1 最终
                "timesteps": tensor(4),                      # 对应 4 个 timestep
            },
            # stage 1 (中分辨率)
            {...},
            # stage 2 (高分辨率)
            {...},
        ],
        # section 1, 2, ... (后续 chunk)
        [...],
    ],
}
```

### 5.4 运行命令

```bash
torchrun --nproc_per_node 8 tools/offload_data/get_ode-pairs.py \
    --base_model_path BestWishYSH/Helios-Base \
    --transformer_path BestWishYSH/Helios-Mid \
    --height 384 --width 640 \
    --num_frames 165 \
    --use_dynamic_shifting \
    --is_enable_stage2 \
    --stage2_num_inference_steps_list 20 20 20
```

如果已训练了带 action 的模型，也可加载 LoRA 生成 ODE 对：

```bash
torchrun --nproc_per_node 8 tools/offload_data/get_ode-pairs.py \
    --base_model_path BestWishYSH/Helios-Base \
    --transformer_path BestWishYSH/Helios-Mid \
    --lora_path /path/to/checkpoint-XXXX \
    --partial_path /path/to/checkpoint-XXXX/transformer_partial.pth \
    --height 384 --width 640 \
    --num_frames 165
```

---

## 6. Step 4: 文本 Embedding（可选，Self-Forcing 用）

为纯文本 prompt 预计算 UMT5 embedding，用于 Stage 3 的 Self-Forcing 分支。

```bash
torchrun --nproc_per_node 8 tools/offload_data/get_text-embedding.py \
    --pretrained_model_name_or_path BestWishYSH/Helios-Base
```

产物：每行文本一个 `.pt`，包含 `prompt_raw` + `prompt_embed`。

---

## 7. 预处理产物汇总

最终的数据目录结构：

```
data/helios/
├── action_embeds_cache.pt                    ← 81 种 action embedding
│
├── seadance2_v2_helios_latents/             ← Stage 1/2/3(GAN) 训练数据
│   ├── rotate_0096_361_384_640.pt
│   ├── close_0001_241_384_640.pt
│   ├── ...
│   └── dataset_cache.pkl                     ← 自动生成的元数据缓存
│
├── yume_training_helios_latents/            ← 另一组训练数据
│   ├── AJw3NeaFtXE_0088880_0090680_201_384_640.pt
│   └── ...
│
├── ode_pairs/                               ← Stage 3 ODE 训练数据
│   ├── vidprom_filtered_extended/
│   │   ├── vidprom_filtered_extended_00000.pt
│   │   ├── vidprom_filtered_extended_00001.pt
│   │   └── ...
│   └── ...
│
└── text_embeds/                             ← (可选) 纯文本 embedding
    └── ...
```

---

## 8. 训练时数据加载

### 8.1 Stage 1 & 2：`BucketedFeatureDataset`

**文件**：`helios/dataset/dataloader_history_latents_dist.py`

**加载流程**：

```
.pt 文件名 → 解析 (num_frame, height, width) → 分桶
         │
         ↓  torch.load(.pt)
         │
         ↓  prepare_stage1_latent():
         │    ├── 随机选择一个 chunk 作为 target
         │    ├── 其前面的内容作为 history
         │    ├── 第一帧 latent 作为 x0
         │    └── 用零填充不存在的历史
         │
         ↓  提取该 chunk 的 action:
         │    chunk_actions[choice_idx] → (action_keys, action_mouse)
         │
         ↓  输出 dict:
              {
                  x0_latents:       (C, 1, H', W'),
                  history_latents:  (C, 19, H', W'),    # 16+2+1 = 19
                  target_latents:   (C, 9, H', W'),     # 当前 chunk
                  prompt_embeds:    (seq_len, 4096),
                  action_keys:      "W",
                  action_mouse:     "→",
              }
```

**History 窗口构成**（`history_sizes = [16, 2, 1]`）：

| 名称 | 帧数 | 来源 | 用途 |
|------|------|------|------|
| `history_long` | 16 | 更早的 latent | 长程上下文 |
| `history_mid` | 2 | 前一个 chunk 的末尾 | 中程衔接 |
| `history_short` | 1 | 紧邻的最后一帧 | 短程连续性 |
| **总计** | **19** | | `patch_long/mid/short` 分别处理 |

**分桶机制**：

- 文件名中的 `(num_frame, height, width)` 用作 bucket key
- 同一 batch 的所有样本属于相同 bucket → 保证 tensor shape 一致
- `BucketedSampler` 支持跨数据集的比例采样（`dataset_sampling_ratios`）

**多分辨率支持**（Stage 2 金字塔训练）：

```
允许的分辨率:
  高: (384, 640)       → 48×80 latent
  中: (192, 320)       → 24×40 latent
  低: (96,  160)       → 12×20 latent
```

当加载中/低分辨率样本时，同时加载对应的高分辨率 `.pt` 作为 `base_vae_latent`，用于提供 x0 和 history conditioning。

### 8.2 Stage 3：`DMDDataset`

**文件**：`helios/dataset/dataloader_dmd.py`

Stage 3 混合三种数据源：

| 分支 | 数据源 | 配置键 | 用途 |
|------|--------|--------|------|
| **GAN** | 真实视频 latent | `gan_data_root` | 判别器真样本 / GT history |
| **ODE** | 教师模型轨迹 | `ode_data_root` | ODE regression 监督 |
| **Text** | 纯文本 embedding | `text_data_root` | Self-Forcing 文本条件 |

各分支独立加载后通过 `_align_sample_counts` 对齐长度，确保每个 index 同时有三种数据。

**GAN 分支支持 `is_use_gt_history`**：从真实数据中抽取 chunk 作为 GT history，此时会携带 per-chunk 的 action 标签。

### 8.3 Action Embedding 注入（训练循环）

在 `train_helios.py` 的训练循环中：

```python
# 1. 从 batch 获取 action 标签
action_keys_list = batch["action_keys"]     # ["W", "None", "S+A", ...]
action_mouse_list = batch["action_mouse"]   # ["→", "·", "↓←", ...]

# 2. 查表获取 embedding
for ak, am in zip(action_keys_list, action_mouse_list):
    ae = action_embeds_cache.get((ak, am))
    if ae is None:
        ae = action_embeds_cache[("None", "·")]  # fallback

# 3. 拼接到 prompt embedding
prompt_embeds = torch.cat([prompt_embeds, action_embeds_batch], dim=1)
# shape: (B, seq_scene + seq_action, 4096)
```

Stage 3 的 GAN 分支同理，通过 `gan_action_keys` / `gan_action_mouse` 拼接。

---

## 9. 端到端数据流图

```
                        ┌─────────────────────────────────────────────┐
                        │             原始数据层                        │
                        │                                              │
                        │  MP4 视频 + Action 目录 + Caption TSV/CSV    │
                        └────────────────────┬────────────────────────┘
                                             │
              ┌──────────────────────────────┼──────────────────────────────┐
              │                              │                              │
              ▼                              ▼                              ▼
    ┌─────────────────┐          ┌───────────────────┐          ┌──────────────────┐
    │ precompute       │          │ convert_sekai      │          │ get_ode-pairs    │
    │ _action_embeds   │          │ _to_helios         │          │                  │
    │                  │          │ / get_short-latents │          │ Teacher Pipeline │
    │ 9×9=81 组合      │          │ / get_long-latents  │          │ 推理 + 子采样    │
    │ UMT5 编码        │          │                     │          │                  │
    └────────┬────────┘          │ VAE 33帧 chunk 编码 │          └────────┬─────────┘
             │                   │ UMT5 caption 编码    │                   │
             │                   │ Action 标签收集       │                   │
             ▼                   └──────────┬──────────┘                   ▼
    action_embeds_cache.pt       latent .pt 文件群                  ode .pt 文件群
             │                   (含 vae_latent,                   (含 ode_latents,
             │                    prompt_embed,                     prompt_embed)
             │                    chunk_actions)                          │
             │                          │                                │
             └──────────┬───────────────┴────────────────────────────────┘
                        │
                        ▼
              ┌─────────────────────────────────────────┐
              │            训练 DataLoader                │
              │                                          │
              │  Stage 1/2: BucketedFeatureDataset       │
              │    → x0, history, target, prompt, action │
              │                                          │
              │  Stage 3: DMDDataset                     │
              │    → GAN data + ODE pairs + Text embeds  │
              │    → action embeds (GT history 时)        │
              │                                          │
              └─────────────────┬───────────────────────┘
                                │
                                ▼
              ┌─────────────────────────────────────────┐
              │       train_helios.py 训练循环            │
              │                                          │
              │  prompt_embeds = cat(scene, action)      │
              │  → Transformer forward                   │
              │  → Flow matching / ODE / DMD loss        │
              └─────────────────────────────────────────┘
```

---

## 10. 常见问题

### Q1: 为什么要预编码而不是在线编码？

VAE 和 T5 编码非常耗 GPU 和时间。预编码后，训练时只需 `torch.load` 读取 tensor，大幅提升训练吞吐。每个 `.pt` 文件约 50-200 MB。

### Q2: `chunk_actions` 是怎样对齐到 chunk 的？

`convert_sekai_to_helios.py` 中，对每个 33 帧 chunk 取其**中间帧**（`mid_frame = i * 33 + 16`），在 action label 列表中查找覆盖该帧的 action。训练时，dataloader 随机选择一个 chunk `choice_idx` 作为 target，相应的 `chunk_actions[choice_idx]` 就是该样本的 action 标签。

### Q3: 如果某个 action 组合不在 cache 中怎么办？

训练代码中有 fallback 逻辑：找不到 `(ak, am)` 时，使用 `("None", "·")` 即"站立不动 + 相机静止"的 embedding。

### Q4: 多分辨率数据怎么准备？

默认只预编码 384×640。如果要做 Stage 2 金字塔训练的多分辨率，需额外编码 192×320 和 96×160 的 latent，放在 `mid/` 和 `low/` 子目录中。Dataloader 会根据文件名中的分辨率自动分桶，并在加载低分辨率数据时自动查找对应的高分辨率文件。

### Q5: ODE 对只能用文本生成吗？

当前 `get_ode-pairs.py` 只支持 text-to-video 模式。如果需要 image-conditioned 的 ODE 对，需修改脚本增加 image 输入逻辑。

### Q6: 数据缓存 `dataset_cache.pkl` 是什么？

`BucketedFeatureDataset` 在首次扫描数据目录后，会将文件名 → `(uttid, num_frame, height, width, file_path)` 的映射保存为 pickle 缓存，后续启动时直接加载，避免重复遍历大量文件。如果数据有变动，设置 `force_rebuild: true` 强制重建。

### Q7: 各阶段的数据配置在哪设置？

在 `scripts/training/configs/` 下的 YAML 文件中：

```yaml
data_config:
  use_stage1_dataset: true                    # Stage 1/2 用 BucketedFeatureDataset
  use_stage3_dataset: false                   # Stage 3 用 DMDDataset
  instance_data_root:                         # Stage 1/2 的 latent 目录
    - "data/helios/seadance2_v2_helios_latents"
    - "data/helios/yume_training_helios_latents"
  action_embeds_cache_path: "data/helios/action_embeds_cache.pt"
  single_height: 384
  single_width: 640
  force_rebuild: true

  # Stage 3 专用
  gan_data_root: [...]                        # GAN 分支的真实 latent
  ode_data_root: [...]                        # ODE 分支的教师轨迹
  text_data_root: [...]                       # Text 分支的文本 embedding
```
