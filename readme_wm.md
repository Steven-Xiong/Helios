# Helios World Model — 完整技术文档

## 1. 项目概览

Helios 是一个基于 Wan2.1-14B 的长视频生成框架，通过三阶段渐进式训练实现高质量、可控的视频生成。在此基础上，我们扩展了 **Action-Conditioned World Model** 能力：通过注入键盘/鼠标动作 embedding，让模型学会根据 action 指令控制摄像头运动。

### 核心特性

- **Chunk-by-Chunk 自回归生成**：通过 history latent conditioning 实现任意长度视频
- **多分辨率金字塔细化**：从低分辨率到高分辨率逐级生成，提升效率和质量
- **DMD 对抗蒸馏**：将 20+20+20 步推理压缩至 2+2+2 步，大幅加速
- **Action Conditioning**：keyboard + mouse embedding 注入，实现动作到相机运动的映射

---

## 2. 代码结构

```
Helios/
├── train_helios.py                  # 训练入口（Stage 1/2/3 统一）
├── infer_helios.py                  # 推理入口（支持多种输入模式）
├── visualize_results.py             # 结果 HTML 可视化
├── run.sh                           # Action 模型推理脚本（带 LoRA）
├── run_org.sh                       # 原版三阶段推理对比
│
├── helios/
│   ├── diffusers_version/
│   │   ├── pipeline_helios_diffusers.py   # 核心推理 Pipeline
│   │   ├── transformer_helios_diffusers.py # Transformer（diffusers 版本）
│   │   └── scheduling_helios_diffusers.py  # 调度器（euler/unipc/dmd）
│   ├── modules/
│   │   ├── transformer_helios.py          # Transformer（训练版本）
│   │   └── helios_kernels/                # Triton kernels（RoPE、Norm 等）
│   ├── scheduler/
│   │   └── scheduling_helios.py           # 训练用调度器
│   ├── dataset/
│   │   ├── dataloader_history_latents_dist.py  # Stage 1/2 数据加载
│   │   ├── dataloader_dmd.py                   # Stage 3 数据加载
│   │   └── dataloader_mp4_dist.py              # 原始 MP4 加载（未启用）
│   ├── pipelines/
│   │   └── pipeline_helios.py             # Pipeline（非 diffusers 版本）
│   └── utils/
│       ├── utils_base.py                  # Checkpoint 保存/加载、工具函数
│       └── train_config.py                # OmegaConf 配置定义
│
├── scripts/
│   ├── training/
│   │   ├── configs/
│   │   │   ├── stage_1_action_init.yaml   # Action Stage 1 训练配置
│   │   │   ├── stage_2_action_init.yaml   # Action Stage 2 训练配置
│   │   │   ├── stage_3_action_post.yaml   # Action Stage 3 训练配置
│   │   │   ├── stage_1_init.yaml          # 原版 Stage 1
│   │   │   ├── stage_2_init.yaml          # 原版 Stage 2
│   │   │   └── stage_3_post.yaml          # 原版 Stage 3
│   │   ├── train_ddp.sh
│   │   └── train_deepspeed.sh
│   └── inference/
│       ├── helios-base_i2v.sh             # Base 模型推理参考
│       ├── helios-mid_i2v.sh              # Mid 模型推理参考
│       └── helios-distilled_i2v.sh        # Distilled 模型推理参考
│
├── tools/
│   ├── offload_data/
│   │   ├── convert_sekai_to_helios.py     # YUME 数据 → Helios 格式
│   │   ├── precompute_action_embeds.py    # 预计算 action embedding cache
│   │   ├── get_long-latents.py / .sh      # 编码长视频 latent
│   │   ├── get_short-latents.py / .sh     # 编码短视频 latent
│   │   ├── get_text-embedding.py / .sh    # 编码文本 embedding
│   │   └── get_ode-pairs.py / .sh         # 生成 ODE 训练对
│   ├── merge_lora.py                      # LoRA 权重合并
│   └── merge_lora_base.py
│
├── eval/                                  # 评估脚本（aesthetic、motion、semantic 等）
└── data/
    └── helios/
        ├── action_embeds_cache.pt         # Action embedding 缓存
        └── seadance2_v2_helios_latents/   # 预编码训练数据
```

---

## 3. 三阶段训练流程

### 3.1 总体架构

所有三个阶段都以 `Helios-Base`（Wan2.1-14B）为 transformer 基座，通过 **LoRA** 微调。各阶段通过 checkpoint 继承：

```
Stage 1 ─→ Stage 2 ─→ Stage 3
  (Base + LoRA)  加载 S1 ckpt   加载 S2 ckpt
```

### 3.2 Stage 1 — Autoregressive History Conditioning

**目标**：学习 chunk-by-chunk 长视频生成能力

| 配置项 | 值 |
|--------|-----|
| 基座 | `Helios-Base` |
| 数据集 | `dataloader_history_latents_dist`（预编码 .pt） |
| 调度器 | `UniPCMultistepScheduler`（50 步） |
| LoRA rank | 128 |
| LoRA 目标 | `all-linear`，排除 `down`/`up` |
| 额外训练模块 | `patch_short/mid/long`（full），`patch_embedding`（LoRA） |
| History sizes | `[16, 2, 1]`（long, mid, short） |
| Latent window | 9（→ 33 帧/chunk） |
| 学习率 | 5e-5 |
| 损失 | Flow matching loss，`weighting_scheme: logit_normal` |
| Action | 通过 `action_embeds_cache.pt` 注入 |

**关键机制**：
- `patch_short`：Conv3D(1,2,2) 处理短期历史（1 帧）
- `patch_mid`：Conv3D(2,4,4) 处理中期历史（2 帧）
- `patch_long`：Conv3D(4,8,8) 处理长期历史（16 帧）
- History latent 拼接到主序列前面，经 transformer 联合 attention

### 3.3 Stage 2 — Multi-Stage Pyramid Refinement

**目标**：学习从粗到细的多分辨率去噪

| 配置项 | 值 |
|--------|-----|
| 基座 | `Helios-Base` + Stage 1 ckpt |
| 数据集 | 同 Stage 1 |
| 调度器 | `HeliosScheduler`（stages=3） |
| Pyramid stages | 3（各 20 步推理） |
| Stage range | `[0, 1/3, 2/3, 1]` |
| LoRA rank | 128 |
| 额外训练模块 | patch 冻结 |
| 学习率 | 1e-4 |
| 损失 | Flow matching loss，`weighting_scheme: none` |

**关键机制**：
- Stage 0：1/4 分辨率生成初始 latent
- Stage 1：2× 上采样 + renoise + 去噪细化
- Stage 2：2× 上采样 + renoise + 去噪到最终分辨率
- `stage2_sample_ratios: [1, 2, 1]` 控制各阶段采样比例

### 3.4 Stage 3 — Adversarial Hierarchical Distillation (DMD)

**目标**：将 20+20+20 步蒸馏到 2+2+2 步

| 配置项 | 值 |
|--------|-----|
| 基座 | `Helios-Base` + Stage 2 ckpt |
| 数据集 | `dataloader_dmd`（GAN/ODE/TEXT .pt） |
| 训练模式 | DMD（generator + critic 交替训练） |
| Generator 学习率 | 2e-6 |
| Critic 学习率 | 4e-7 |
| Critic LoRA rank | 128 |
| Update ratio | `dfake_gen_update_ratio: 5`（每 5 次 critic 更新 1 次 generator） |
| 蒸馏步数 | `dmd_denoising_step_list: [1000, 750, 500, 250]` |
| EMA | `decay: 0.99`，`start_step: 750` |
| 额外训练模块 | `patch_short/mid/long`（LoRA） |
| `guidance_scale` | 1.0 |

**关键机制**：
- Generator 生成 fake sample，Critic 给分 → 对抗训练
- ODE regression loss 保证一致性
- `is_use_gt_history: true` → 用真实历史 latent 作为条件
- `is_amplify_first_chunk: true` → 第一个 chunk 用更多步数

---

## 4. 数据准备

### 4.1 预计算 Action Embedding

```bash
python tools/offload_data/precompute_action_embeds.py \
    --output_path data/helios/action_embeds_cache.pt
```

遍历 `VOCAB_KEYBOARD × VOCAB_MOUSE`（9×9=81 种组合），用 T5 编码文本 prompt，生成缓存字典 `{(keys, mouse): tensor(1, seq_len, 4096)}`。

### 4.2 转换 YUME 数据到 Helios 格式

```bash
torchrun --nproc_per_node 8 tools/offload_data/convert_sekai_to_helios.py \
    --video_dir data/seadance2_yume/video \
    --action_dir data/seadance2_yume/mp4_frame \
    --tsv_path data/seadance2_yume/world_model_action12_train_1200_v2.tsv \
    --output_dir data/helios/seadance2_v2_helios_latents
```

每个视频输出一个 `.pt` 文件，包含：
- `vae_latent`：VAE 编码后的 latent（按 33 帧 chunk 编码）
- `prompt_embed`：T5 文本 embedding
- `prompt_raw`：原始文本
- `chunk_actions`：每个 chunk 的 `{keys, mouse}` 动作标签
- `first_frames_image`：首帧图片

### 4.3 数据格式

**Stage 1/2**（`dataloader_history_latents_dist`）：
- 输入：`.pt` 文件，包含 `vae_latent`, `prompt_embed`, `chunk_actions`
- 按 `(num_frame, height, width)` 分桶 batch
- 输出：`x0_latents`, `history_latents`, `target_latents`, `prompt_embeds`, action keys/mouse

**Stage 3**（`dataloader_dmd`）：
- 输入：GAN/ODE/TEXT 三类 `.pt` 文件
- 输出：`gan_vae_latents`, `gan_prompt_embeds`, `gan_history_latents` 等
- 支持 GT history：`is_use_gt_history` 从真实数据提取历史 latent

---

## 5. 推理流程

### 5.1 模型加载

```python
transformer = HeliosTransformer3DModel.from_pretrained(transformer_path, ...)
vae = AutoencoderKLWan.from_pretrained(base_model_path, ...)
scheduler = HeliosScheduler.from_pretrained(base_model_path, ...)
pipe = HeliosPipeline.from_pretrained(base_model_path, transformer=transformer, vae=vae, scheduler=scheduler)
```

**关键点**：`base_model_path` 和 `transformer_path` 可以不同：
- `base_model_path` 提供 scheduler config、VAE、text encoder
- `transformer_path` 提供 transformer 权重

例如 Action Stage 3 推理时：
- `base_model_path = Helios-Distilled` → scheduler 有 stages=3, dmd
- `transformer_path = Helios-Base` → transformer 基座与训练一致

### 5.2 LoRA 加载

```python
# 1. 分离 LoRA 和 norm 权重
full_state = safetensors.torch.load_file(pytorch_lora_weights.safetensors)
lora_state = {k: v for k, v in full_state.items() if "lora" in k}
norm_state = {k: v for k, v in full_state.items() if "lora" not in k}

# 2. 添加 PEFT adapter 并加载 LoRA
transformer.add_adapter(LoraConfig(r=128, target_modules="all-linear", ...))
set_peft_model_state_dict(transformer, peft_state)

# 3. 加载 partial checkpoint（patch modules + extras）
load_extra_components(args, transformer, "transformer_partial.pth")
```

`transformer_partial.pth` 包含训练时 PEFT 包裹的 patch 权重（`base_layer.weight` 格式），加载时自动合并 LoRA 到 base weight。

### 5.3 推理 Pipeline 数据流

```
Input (prompt + image/video) → encode → prompt_embeds + image_latents
                                           ↓
For each chunk k (0 → num_chunks):
    ┌──────────────────────────────────────────────────┐
    │ 1. 构建 history latent：                           │
    │    latents_history_short ← last 1 frame           │
    │    latents_history_mid   ← last 2 frames          │
    │    latents_history_long  ← last 16 frames         │
    │                                                    │
    │ 2. Action embedding 注入：                         │
    │    prompt_embeds = cat(prompt_embeds, action_embed) │
    │                                                    │
    │ 3. 生成（stage1_sample 或 stage2_sample）：        │
    │    - Stage 2/3: 金字塔 coarse-to-fine              │
    │      Stage 0: 1/4 res → denoise                    │
    │      Stage 1: 2× upsample + renoise → denoise     │
    │      Stage 2: 2× upsample + renoise → denoise     │
    │    - DMD: predict x0 → renoise → predict x0       │
    │                                                    │
    │ 4. 更新 history + VAE decode                       │
    └──────────────────────────────────────────────────┘
                                           ↓
Concatenate chunks → output .mp4
```

### 5.4 输入模式

| 模式 | 触发参数 | 说明 |
|------|---------|------|
| 单 prompt | 默认 | 单条 prompt + 可选 image/video |
| Prompt 列表 | `--prompt_txt_path` | 文本文件每行一条 prompt |
| Image-Prompt CSV | `--image_prompt_csv_path` | CSV 含 `id`, `prompt`, `image_name`，支持 per-row action |
| Interactive CSV | `--interactive_prompt_csv_path` | CSV 含 `id`, `prompt_index`，支持 per-chunk 动态 prompt + action |

### 5.5 Action Embedding 构建

三种方式获取 per-chunk action embedding：

1. **从 prompt 解析**：自动解析 `"the camera moves forward, the camera pans left"` → `("W", "←")`
2. **从 CSV 列**：`action_keys` 和 `action_mouse` 列
3. **命令行参数**：`--action_keys W --action_mouse →`

Action embedding 从 `action_embeds_cache.pt` 查表，拼接到 `prompt_embeds` 的 sequence 维度。

---

## 6. 推理配置对照

### 6.1 原版三阶段（`run_org.sh`）

| 模型 | base_model_path | transformer_path | Stage 2 | 步数 | guidance | amplify |
|------|----------------|-----------------|---------|------|---------|---------|
| Base | Helios-Base | Helios-Base | ✗ | 50 | 5.0 | ✗ |
| Mid | Helios-Mid | Helios-Mid | ✓ | 20+20+20 | 5.0 | ✗ |
| Distilled | Helios-Distilled | Helios-Distilled | ✓ | 2+2+2 | 1.0 | ✓ |

### 6.2 Action 模型（`run.sh`）

| Stage | base_model_path | transformer_path | LoRA | partial | Stage 2 | 步数 | guidance | amplify |
|-------|----------------|-----------------|------|---------|---------|------|---------|---------|
| S1 推理 | Helios-Base | Helios-Base | S1 ckpt | ✓ | ✗ | 50 | 5.0 | ✗ |
| S2 推理 | Helios-Base | Helios-Base | S2 ckpt | ✓ | ✓ | 20+20+20 | 5.0 | ✗ |
| S3 推理 | Helios-Distilled | Helios-Base | S3 ckpt | ✓ | ✓ | 2+2+2 | 1.0 | ✓ |

**Stage 3 推理注意事项**：
- `base_model_path` 必须用 `Helios-Distilled`，因为其 scheduler 有 `stages=3` + `scheduler_type=dmd`
- `transformer_path` 必须用 `Helios-Base`，因为 LoRA 是在 Base 基座上训的
- 两者的 VAE 和 text encoder 是共享的（来自 Wan2.1），所以可以混用

---

## 7. Checkpoint 结构

每个 checkpoint 目录包含：

```
checkpoint-XXXX/
├── pytorch_lora_weights.safetensors    # PEFT LoRA 权重（所有 linear 层 + patch LoRA）
└── transformer_partial.pth             # 额外组件：
                                        #   - patch_short/mid/long（base_layer + LoRA）
                                        #   - q_loras/k_loras/v_loras（restrict_self_attn 时）
                                        #   - history_key_scale（is_amplify_history 时）
                                        #   - gan_heads/gan_final_head（is_use_gan 时）
```

**LoRA 权重 key 格式**（safetensors）：
```
transformer.blocks.{i}.attn1.to_k.lora_A.weight
transformer.blocks.{i}.attn1.to_k.lora_B.weight
transformer.blocks.{i}.ffn.net.0.proj.lora_A.weight
transformer.condition_embedder.time_proj.lora_A.weight
transformer.patch_short.lora_A.weight          # 这 6 个 key 在 partial 中处理
transformer.proj_out.lora_A.weight
...
```

**Partial 权重 key 格式**（pth，PEFT 包裹后保存）：
```
patch_short.base_layer.weight     → 推理时 remap 为 weight 并合并 LoRA
patch_short.base_layer.bias       → 推理时 remap 为 bias
patch_short.lora_A.default.weight → merged into base_layer.weight
patch_short.lora_B.default.weight → merged into base_layer.weight
```

---

## 8. 关键配置参数说明

### 训练配置

| 参数 | 说明 | Stage 1 | Stage 2 | Stage 3 |
|------|------|---------|---------|---------|
| `is_enable_stage1` | 启用 history conditioning | ✓ | ✓ | ✓ |
| `is_enable_stage2` | 启用金字塔多阶段 | ✗ | ✓ | ✓ |
| `is_train_dmd` | 启用 DMD 对抗训练 | ✗ | ✗ | ✓ |
| `is_train_lora_patch_embedding` | 对 patch_embedding 加 LoRA | ✓ | ✗ | ✗ |
| `is_train_full_multi_term_memory_patchg` | 全量训练 patch_short/mid/long | ✓ | ✗ | ✗ |
| `is_train_lora_multi_term_memory_patchg` | 对 patch_short/mid/long 加 LoRA | ✗ | ✗ | ✓ |
| `is_amplify_first_chunk` | 第一个 chunk 用更多步 | ✗ | ✗ | ✓ |
| `is_use_gt_history` | 用真实历史 latent 训练 | ✗ | ✗ | ✓ |

### 推理参数

| 参数 | 说明 |
|------|------|
| `--base_model_path` | 提供 scheduler/VAE/text_encoder |
| `--transformer_path` | 提供 transformer 基座 |
| `--lora_path` | LoRA 权重路径 |
| `--partial_path` | partial checkpoint 路径 |
| `--is_enable_stage2` | 启用金字塔推理 |
| `--pyramid_num_inference_steps_list` | 每个金字塔阶段的步数 |
| `--is_amplify_first_chunk` | 第一个 chunk 用更多步（DMD） |
| `--guidance_scale` | CFG 引导强度（Distilled 用 1.0） |
| `--action_embeds_cache` | Action embedding 缓存路径 |
| `--action_keys` / `--action_mouse` | 单次推理的动作指令 |

---

## 9. 运行指南

### 9.1 数据准备

```bash
# 1. 预计算 action embedding cache
python tools/offload_data/precompute_action_embeds.py \
    --output_path data/helios/action_embeds_cache.pt

# 2. 转换数据
torchrun --nproc_per_node 8 tools/offload_data/convert_sekai_to_helios.py \
    --video_dir /path/to/video \
    --action_dir /path/to/mp4_frame \
    --tsv_path /path/to/tsv \
    --output_dir data/helios/seadance2_v2_helios_latents
```

### 9.2 训练

```bash
# Stage 1
accelerate launch --config_file scripts/accelerate_configs/multi_node_example_zero2.yaml \
    train_helios.py --config scripts/training/configs/stage_1_action_init.yaml

# Stage 2（需要在 yaml 中设置 load_model_path 指向 Stage 1 checkpoint）
accelerate launch --config_file scripts/accelerate_configs/multi_node_example_zero2.yaml \
    train_helios.py --config scripts/training/configs/stage_2_action_init.yaml

# Stage 3（需要在 yaml 中设置 load_model_path 和 critic_lora_name_or_path 指向 Stage 2 checkpoint）
accelerate launch --config_file scripts/accelerate_configs/multi_node_example_zero2.yaml \
    train_helios.py --config scripts/training/configs/stage_3_action_post.yaml
```

### 9.3 推理

```bash
# Action 模型推理（Stage 3 checkpoint）
bash run.sh

# 原版模型对比
bash run_org.sh
```

### 9.4 单样本推理示例

```bash
CUDA_VISIBLE_DEVICES=0 python infer_helios.py \
    --base_model_path "BestWishYsh/Helios-Distilled" \
    --transformer_path "BestWishYsh/Helios-Base" \
    --lora_path "/path/to/checkpoint-1000" \
    --partial_path "/path/to/checkpoint-1000/transformer_partial.pth" \
    --sample_type i2v \
    --image_path "example/scene.jpg" \
    --prompt "This video depicts a first-person view. Person moves forward (W). Camera turns right (→)." \
    --action_embeds_cache data/helios/action_embeds_cache.pt \
    --action_keys W \
    --action_mouse "→" \
    --num_frames 321 \
    --guidance_scale 1.0 \
    --is_enable_stage2 \
    --pyramid_num_inference_steps_list 2 2 2 \
    --is_amplify_first_chunk \
    --output_folder "./output_helios/test"
```

---

## 10. 已知问题与注意事项

1. **PEFT exclude_modules warning**：PEFT v0.18.1 对 `exclude_modules=["down", "up"]` 报 warning（模型中无同名模块），多创建的 LoRA 位为零初始化，无影响。

2. **base_model_path vs transformer_path**：Stage 3 推理必须分开指定——scheduler 来自 Distilled（stages=3），transformer 来自 Base（与训练一致）。

3. **Patch 加载**：训练保存的 partial checkpoint 中 patch 权重为 PEFT 格式（`base_layer.*`），加载时自动 remap 并合并 LoRA（见 `load_extra_components`）。

4. **Action embedding 维度**：每个 action embedding 为 `(1, seq_len, 4096)`，拼接到 prompt_embeds 的 sequence 维度。

5. **Stage 3 训练基座问题**：当前 action 训练三阶段均使用 `Helios-Base` 做基座。原版 Stage 3 使用 `Helios-Distilled` 的 `transformer_ode` subfolder。如果需要更好的少步推理质量，可考虑以 Distilled 为基座重训 Stage 3。
