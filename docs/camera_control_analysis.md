# Helios Action World Model — Camera Control 方案分析报告

> 日期：2026-04-01
> 背景：Stage 1 camera control 训练结果约束偏弱，分析原因并调研业界方案

---

## 目录

1. [当前方案概述](#1-当前方案概述)
2. [Camera Control 偏弱的原因分析](#2-camera-control-偏弱的原因分析)
3. [业界方案调研](#3-业界方案调研)
   - [3.1 Lingbot-World](#31-lingbot-world)
   - [3.2 HY-WorldPlay](#32-hy-worldplay)
   - [3.3 Matrix-Game-3](#33-matrix-game-3)
   - [3.4 Infinite-World](#34-infinite-world)
4. [横向对比](#4-横向对比)
5. [LoRA 训练的影响](#5-lora-训练的影响)
6. [数据现状与差距](#6-数据现状与差距)
7. [改进方案与推荐路线](#7-改进方案与推荐路线)

---

## 1. 当前方案概述

### 架构

- **Base Model**: Wan2.1-14B (Helios-Base)
- **Action 表征**: 9 keyboard × 9 mouse = 81 类离散动作 → T5 自然语言 embedding
- **Action 注入**: 预缓存 T5 embedding → concat 到 `encoder_hidden_states` → 共享 text cross-attention
- **History/Memory**: 多尺度 VAE latent `[16, 2, 1]` + multi-term memory patch + Guidance Cross-Attention
- **Anti-drifting**: Frame-aware corruption (对 history 加噪)
- **训练**: LoRA rank 128 on all-linear (exclude down/up) + norm layers 全参 + multi-term memory patch 全参
- **训练步数**: Stage 1 init = 2000 steps, LR 5e-5

### 数据流

```
Action: "W" + "→"
  → T5 encode → action_embed (seq_len, 4096)
  → torch.cat([scene_prompt_embed, action_embed], dim=1)
  → encoder_hidden_states
  → Cross-Attention (Q=noisy latent, KV=scene+action)
```

### 来源

从 YUME 1.5 迁移而来。YUME 1.5 在 Wan2.2-5B 上将 WASD+mouse 编成自然语言拼进 caption，通过 T5 单次编码后走 cross-attention。Helios 版本改进为预缓存 T5 embedding 按 seq 维度拼接，但**本质注入方式不变**。

---

## 2. Camera Control 偏弱的原因分析

### 2.1 根本原因：Action 信号与 Text 共享 Cross-Attention

Action embedding 仅仅是 concat 在 scene prompt 后面，在 cross-attention 里变成额外的 KV tokens：

```python
# train_helios.py L1307
prompt_embeds = torch.cat([prompt_embeds, action_embeds_batch], dim=1)
```

问题：
- Scene prompt tokens 数量远多于 action tokens（几十个 vs 几个）
- 预训练模型的 cross-attention 已经学会强 attend 到 scene description，action tokens 被当作"噪声"忽略
- **没有任何机制让 action 信号获得优先级**

### 2.2 缺少独立的 Action 条件通路

当前 action 和 scene description 共享同一个 cross-attention，没有架构层面的区分。模型需要纯靠 2000 步 LoRA 训练从数据中自己"发现"哪些 tokens 是控制相机的——这对 LoRA 微调来说太难了。

### 2.3 离散动作缺乏几何约束

"Camera turns right (→)" 这种文本描述是**语义级别**的，没有几何上的精确含义。模型无法区分转 10° 还是转 90°，也无法学习到 smooth 的 camera trajectory。

### 2.4 训练不充分

- `max_train_steps: 2000` 对学习一个新的 conditioning modality 偏少
- `caption_dropout_p: 0.1` 会同时 drop 掉 scene 和 action（action 信号也被丢了）

### 2.5 LoRA 加重了问题（但不是根本原因）

详见 [第 5 节](#5-lora-训练的影响)。

---

## 3. 业界方案调研

### 3.1 Lingbot-World

**论文**: arXiv:2601.20540 | **Base Model**: Wan2.2 (14B 级)

#### Action/Camera 编码

| 模式 | 输入 | 维度 |
|------|------|------|
| Cam | 连续 c2w 4×4 + K 内参 | 6D ray bundle (origin + direction) per pixel |
| Act | 3D ray direction + action.npy | 7D |

#### 注入方式：Per-Block FiLM (Scale + Shift)

```
c2w → 6D ray per pixel → patchify → Linear(patch_embedding_wancamctrl) → 2-layer MLP

每个 WanAttentionBlock:
  1. Self-attention → x
  2. cam_feat = cam_injector_layer2(SiLU(cam_injector_layer1(plucker_emb))) + plucker_emb
  3. cam_scale = cam_scale_layer(cam_feat)
     cam_shift = cam_shift_layer(cam_feat)
  4. x = (1 + cam_scale) * x + cam_shift   ← FiLM 调制
  5. Text cross-attention → x
```

#### 关键特点

- Camera 信号是**像素级对齐**的（每个 patch token 对应独立的 ray 信息）
- FiLM 直接缩放和偏移 feature，控制力极强
- 与 text cross-attention **完全分离**
- **无 ControlNet**，新增层在 DiT 内部

### 3.2 HY-WorldPlay

**论文**: HY-World 1.5 Tech Report | **Base Model**: HunyuanVideo 1.5

#### 双通路设计

**通路 A — 离散动作 → AdaLN:**

```python
# 81 类离散 action (9 translation × 9 rotation，从相对 pose 量化)
vec = self.time_in(timestep)
vec = vec + self.action_in(action_id)  # TimestepEmbedder → 加到 AdaLN modulation
# vec 广播到所有 spatial tokens，通过 AdaLN 调制每一层
```

**通路 B — 连续 Pose → PRoPE:**

```python
# Per-frame w2c 4×4 + K 3×3 → PRoPE (Projective Positional Encoding)
img_q_prope, img_k_prope, img_v_prope = prope_qkv(img_q, img_k, img_v, viewmats, Ks)
img_attn_prope = self_attn(img_q_prope, img_k_prope, img_v_prope)

# 融合（prope_proj zero-init）
img = img + gate * (img_attn_proj(img_attn) + img_attn_prope_proj(img_attn_prope))
```

#### 关键特点

- 离散动作通过 **AdaLN** 全局调制（像 timestep 一样强），注入到每一层
- 连续 pose 通过 **PRoPE** 在 self-attention 级别注入精确几何信息
- `img_attn_prope_proj` **zero-init** 保护预训练质量
- **WorldCompass RL 后训练**: 用 camera pose estimator 做 reward，GRPO 优化 action-following

### 3.3 Matrix-Game-3

**论文**: Technical Report 2026 | **Base Model**: Wan2.2-5B (+28B MoE)

#### 三通路设计

**通路 A — Text → Cross-Attention (标准)**

**通路 B — Camera → Per-Block FiLM (同 Lingbot):**

```python
# c2w → 6D ray bundle per pixel → patchify → Linear + MLP
# 每个 block: x = (1 + cam_scale) * x + cam_shift
```

**通路 C — Action → GameFactory ActionModule:**

```python
class ActionModule(nn.Module):
    # mouse_dim_in=2, keyboard_dim_in=6 (连续值)
    # Mouse 通路: 时间窗口(window_size=3) + hidden concat → MLP → self-attn with 3D RoPE → proj → residual add
    # Keyboard 通路: Embedding → grouped → cross-attn (Q=hidden, KV=keyboard) → proj → residual add
    # 只在部分 block 激活 (通过 blocks 参数控制)
    # Memory 支持: memory 和 prediction 分别应用 RoPE
```

#### 关键特点

- **三条通路完全解耦**: 语义(text) / 几何(ray) / 控制(action)
- ActionModule 来自 GameFactory (arXiv:2501.08325)，有时间窗口感知
- **Camera-aware memory selection**: 根据视锥重叠度选取最相关的历史帧
- **DMD 蒸馏** + INT8 量化 + VAE decoder 蒸馏 → 40fps 实时

### 3.4 Infinite-World

**论文**: arXiv:2602.02393 | **Base Model**: Wan-style DiT

#### Action 注入：Additive Token Bias

```python
class ActionEncoder(nn.Module):
    # 10 类 move + 10 类 view (含 uncertain)
    # move_embedding = nn.Embedding(10, 256)
    # view_embedding = nn.Embedding(10, 256)
    # → concat → 1D Conv stack (temporal downsample) → Linear → 1536D (= dit_dim)

# 注入: patch_embedding 之后直接加到 token features
x = patch_embedding(noisy_latent)
x = x + action_embedding  # broadcast over H, W
```

#### 关键特点

- 简单直接的 additive bias
- 10 类 move + 10 类 view 包含 `uncertain` 类处理无标注数据
- **TemporalLatentEncoder** 做历史 latent 的时间压缩
- **无连续 camera pose**

---

## 4. 横向对比

### 4.1 Action 编码方式

| | Helios (当前) | Lingbot | WorldPlay | MG3 | Infinite-World |
|---|---|---|---|---|---|
| **表征** | 81 类 → T5 文本 | 连续 c2w → 6D ray | 81 类 → TimestepEmbedder | keyboard 6D + mouse 2D 连续 | 10+10 类 → Embedding |
| **编码器** | T5 (通用 text encoder) | Linear + MLP (专用) | TimestepEmbedder (专用) | GameFactory ActionModule (专用) | Embedding + Conv (专用) |
| **有几何信号** | ❌ | ✅ per-pixel ray | ✅ PRoPE (w2c+K) | ✅ per-pixel ray | ❌ |

### 4.2 Action 注入方式

| | Helios (当前) | Lingbot | WorldPlay | MG3 | Infinite-World |
|---|---|---|---|---|---|
| **注入位置** | concat → 共享 cross-attn | per-block FiLM | AdaLN vec + PRoPE attn | FiLM + ActionModule attn | additive on patch tokens |
| **独立于 text?** | ❌ 混在一起 | ✅ | ✅ | ✅ | ✅ |
| **空间对齐?** | ❌ 全局 seq 拼接 | ✅ per-token | ✅ per-token (PRoPE) | ✅ per-token + 时间窗口 | ⚠️ broadcast over H,W |

### 4.3 训练策略

| | Helios | Lingbot | WorldPlay | MG3 | Infinite-World |
|---|---|---|---|---|---|
| **基础 Loss** | Flow matching MSE | (未开源) | Masked latent MSE | Rectified flow MSE | Rectified flow MSE |
| **Camera Loss** | ❌ | (未知) | ❌ (base 训练无) | ❌ | ❌ |
| **RL 后训练** | ❌ | ❌ | ✅ WorldCompass | ❌ | ❌ |
| **微调方式** | LoRA rank 128 | 全参/未知 | 全参 action 层 | 全参/未知 | 未知 |
| **蒸馏** | Stage 2/3 planned | ❌ | Context Forcing | DMD multi-seg | ❌ |

### 4.4 Memory 机制

| | Helios | Lingbot | WorldPlay | MG3 | Infinite-World |
|---|---|---|---|---|---|
| **机制** | 多尺度 [16,2,1] + patch + RoPE | (未开源) | Reconstituted context | Latent concat + FOV-aware select | TemporalLatentEncoder + concat |
| **Anti-drift** | Frame-aware corruption | (未知) | Context Forcing | Prediction residual + re-injection | (不明确) |
| **Memory 选取** | 固定 [16,2,1] | (未知) | (未知) | **视锥重叠度** | 滑窗 |

### 4.5 综合对比

```
控制精度:    MG3 ≈ Lingbot > WorldPlay >> Helios(当前)
架构复杂度:  MG3 > WorldPlay > Lingbot >> Helios(当前)
实现难度:    Helios(当前) << Lingbot < WorldPlay < MG3
Base model:  Helios 14B ≈ Lingbot 14B > WorldPlay(HunYuan) > MG3 5B
```

**核心结论**: 四个参考方案**都没有**把 action 放进 text cross-attention。它们各自选择了不同的独立注入通路。

---

## 5. LoRA 训练的影响

### LoRA 加重问题的三个原因

**1. Cross-Attention K/V 投影被 LoRA 约束**

模型需要从 K/V 投影中学会区分 scene text tokens 和 action tokens。LoRA rank 128 的更新空间需要在 scene 理解和 action 理解之间分配，action 很容易被挤压。

**2. 预训练 Attention 偏好会"锁定"**

LoRA 本质是在预训练权重附近做小幅扰动 (`W + BA`)。预训练时 cross-attention 学到的是"关注描述性文本"，LoRA 很难让模型从相似的 T5 token 中提取出截然不同的控制信号。

**3. 2000 步 LoRA 训练量不足**

对学习一种新的 conditioning modality（识别 action tokens → 提取运动信息 → 反映到 latent），2000 步 LoRA 链路太长。

### 但 LoRA 不是根本原因

即使改成全参训练，问题也不会完全解决：
- Cross-attention 同时服务 scene 理解 + action 控制，两者会互相干扰
- 全参训练可能改坏 scene understanding 的 K/V 投影
- 这就是为什么四个参考项目**都选择了独立通路** —— 新模块处理 action，保持 text cross-attention 不变

---

## 6. 数据现状与差距

### 6.1 现有数据

**yume_training (Sekai/YouTube 真实视频, VIPE 估计 pose)**

```
文件: mp4_frame/Keys_{keys}_Mouse_{mouse}/*.{mp4, txt, npy}
.npy: shape=(48, 4, 4), dtype=float64, c2w 矩阵
逐帧运动: translation ~0.0001, rotation ~0.07°/frame
来源: VIPE 单目 pose 估计
```

**seadance2_yume_v3 (UE 合成视频, VIPE 估计 pose)**

```
文件: mp4_frame/Keys_{keys}_Mouse_{mouse}/*.{mp4, txt, npy}
.npy: shape=(33, 4, 4), dtype=float64, c2w 矩阵
逐帧运动: translation ~0.001-0.03, rotation ~0.01-0.12°/frame
c2w 接近单位矩阵 (运动非常小)
来源: VIPE 单目 pose 估计 (但 UE 数据应有 GT pose)
```

**共同特点**

| 有的 | 没有的 |
|------|--------|
| ✅ 逐帧 c2w 4×4 矩阵（`.npy`） | ❌ 内参 K（焦距、主点） |
| ✅ 离散 WASD+Mouse 标签（`.txt`） | ❌ `.pt` 训练文件中不含 pose |
| ✅ 有效旋转矩阵（det≈1.0） | ❌ Ground truth 内参 |

### 6.2 与 Lingbot/MG3 需求的差距

**Lingbot-World 需要:**

| 数据 | 状态 | 说明 |
|------|------|------|
| `intrinsics.npy` [T,4] | ❌ 缺失 | 需要真实 [fx,fy,cx,cy] |
| `poses.npy` [T,4,4] c2w | ✅ 有 | .npy 文件 |
| `action.npy` 连续向量 | ❌ 缺失 | 只有离散标签 |

**Matrix-Game-3 需要:**

| 数据 | 状态 | 说明 |
|------|------|------|
| c2w 4×4 per frame | ✅ 有 | .npy 文件 |
| K (内参) | ⚠️ 可假设 | MG3 硬编码 FOV=90° |
| keyboard 6D + mouse 2D 连续 | ⚠️ 需映射 | 现有离散标签可转换 |

### 6.3 关键质量问题

**seadance2 的 VIPE pose 质量不足:**
- UE 渲染数据**应该有完美的 GT pose**
- 但 VIPE 估计的 c2w 几乎是单位矩阵，运动极小
- 这意味着 VIPE 在合成数据上估计不准
- **需要获取 UE 原始 camera transform**

**yume_training 的 VIPE pose:**
- 真实视频只能靠估计，VIPE 是合理选择
- 但有**尺度歧义** (scale ambiguity)
- Lingbot/MG3 都做 `normalize translation` 来缓解

**结论: ray conditioning 的前置条件尚不满足**（特别是 seadance2 的 pose 质量问题）

---

## 7. 改进方案与推荐路线

### 7.1 方案对比

| 方案 | 实现时间 | 训练收敛 | 兼容现有 pipeline | Camera 精度 | 数据要求 |
|------|---------|---------|----------------|-----------|---------|
| **WorldPlay AdaLN** | ⭐⭐⭐⭐⭐ (半天) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ (语义级) | 无新数据 |
| **Infinite-World Additive** | ⭐⭐⭐⭐ (1天) | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ (语义级) | 无新数据 |
| **Lingbot FiLM** | ⭐⭐⭐ (2-3天) | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ (几何级) | 需 c2w+K |
| **MG3 ActionModule + FiLM** | ⭐⭐ (3-5天) | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ (几何级) | 需 c2w+K+连续 action |

### 7.2 推荐: WorldPlay AdaLN (Phase 1, 首选)

**为什么首选:**
1. 81 类离散 action 与现有体系完美对齐，数据 pipeline 零改动
2. 代码改动量 ~50 行，涉及 4 个文件
3. AdaLN 是条件化最强的方式之一（与 timestep 同级），action 信号不会被淹没
4. Zero-init 保护预训练质量
5. 不需要 pose 数据、不需要 K

**核心改动:**

```python
# 1. transformer_helios.py — HeliosTransformer3DModel.__init__
self.action_in = nn.Sequential(
    nn.Embedding(81, inner_dim),
    nn.SiLU(),
    nn.Linear(inner_dim, inner_dim),
)
nn.init.zeros_(self.action_in[-1].weight)
nn.init.zeros_(self.action_in[-1].bias)

# 2. transformer_helios.py — forward
# 在 temb = self.time_embedding(timestep_proj) 之后:
if action_id is not None:
    temb = temb + self.action_in(action_id)

# 3. train_helios.py — 替换 action_embeds concat 逻辑
# 原: prompt_embeds = torch.cat([prompt_embeds, action_embeds_batch], dim=1)
# 新: action_id = keyboard_idx * 9 + mouse_idx → 传进 transformer

# 4. pipeline_helios.py / infer_helios.py — 推理时传 action_id
```

**训练建议:**
- `action_in` 模块**全参训练**
- DiT 其他部分继续 LoRA rank 128
- 可以同步增加训练步数到 5000+

### 7.3 Phase 2: Lingbot/MG3 式 Ray FiLM

**前置条件:**
1. seadance2 获取 UE ground truth camera pose（非 VIPE 估计）
2. 确定 K 来源：UE 数据读 FOV 设置；YouTube 数据假设 FOV=90° (像 MG3)
3. 修改 `convert_sekai_to_helios.py` 将 c2w 打包到 `.pt`
4. 修改 dataloader 传递 pose tensor

**核心改动:**

```python
# 每个 HeliosTransformerBlock 新增:
self.cam_injector_layer1 = nn.Linear(dim, dim)
self.cam_injector_layer2 = nn.Linear(dim, dim)
self.cam_scale_layer = nn.Linear(dim, dim)
self.cam_shift_layer = nn.Linear(dim, dim)

# forward:
cam_feat = self.cam_injector_layer2(F.silu(self.cam_injector_layer1(plucker_emb))) + plucker_emb
cam_scale = self.cam_scale_layer(cam_feat)
cam_shift = self.cam_shift_layer(cam_feat)
x = (1 + cam_scale) * x + cam_shift
```

### 7.4 Phase 3 (可选): 进阶优化

- **MG3 式 ActionModule**: keyboard/mouse 分开处理，时间窗口 attention
- **Camera-aware memory selection**: 根据视锥重叠度选帧（替代固定 [16,2,1]）
- **WorldCompass RL**: pose estimator 做 reward，GRPO 后训练

### 7.5 推荐时间线

```
Week 1:   WorldPlay AdaLN (实现 + 训练验证)
          → 验证 action following 是否显著改善
Week 2-3: 准备 pose 数据 (获取 UE GT pose, 确定 K, 修改数据 pipeline)
Week 3-4: Lingbot/MG3 FiLM ray conditioning
          → 验证精确 camera 控制
Week 5+:  可选 — MG3 ActionModule / Camera-aware memory / RL
```

---

## 附录 A: 关键文件索引

### Helios

| 文件 | 说明 |
|------|------|
| `scripts/training/configs/stage_1_action_init.yaml` | Stage 1 训练配置 |
| `scripts/training/configs/stage_2_action_init.yaml` | Stage 2 训练配置 (LoRA 256, navit pyramid) |
| `helios/modules/transformer_helios.py` | DiT 架构 (HeliosTransformer3DModel) |
| `train_helios.py` | 训练入口 (action_embeds concat 逻辑) |
| `helios/pipelines/pipeline_helios.py` | 推理 pipeline (per-chunk action) |
| `infer_helios.py` | 推理脚本 |
| `tools/offload_data/precompute_action_embeds.py` | 81 类 T5 embedding 缓存生成 |
| `tools/offload_data/convert_sekai_to_helios.py` | 数据转换 (.pt 不含 pose) |
| `src/fastvideo/scripts/prepare_vipe_to_yume.py` | VIPE → YUME 数据格式 |

### 参考项目

| 项目 | 关键文件 |
|------|---------|
| **Lingbot** | `wan/modules/model.py` (FiLM), `wan/utils/cam_utils.py` (ray) |
| **WorldPlay** | `hyvideo/models/transformers/worldplay_1_5_transformer.py` (AdaLN + PRoPE) |
| **MG3** | `wan/modules/action_module.py` (ActionModule), `utils/cam_utils.py` (ray + FOV memory) |
| **Infinite-World** | `infworld/models/dit_model.py` (ActionEncoder + additive) |

## 附录 B: Stage 2 训练配置备注

Stage 2 (`stage_2_action_init.yaml`) 相比 Stage 1 的主要变化:
- LoRA rank 提升到 256 (Stage 1 为 128)
- `caption_dropout_p: 0` (Stage 1 为 0.1)
- `is_enable_stage2: true`, `is_navit_pyramid: true` — 3-stage pyramid distillation
- `learning_rate: 1e-4` (Stage 1 为 5e-5)
- `train_norm_layers: false` (Stage 1 为 true)
- 加入 `yume_training_helios_latents` 作为额外数据源
- 但 action 注入方式不变（仍然 concat 到 cross-attention）

**Stage 2 也会继承 Stage 1 camera control 弱的问题**，因为 action 注入架构没有改变。建议在 Stage 1 改进 action 注入方式后再推进 Stage 2。
