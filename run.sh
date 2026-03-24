# Must run from Helios root directory
cd /mnt/bn/voyager-sg-l3/zhexiao.xiong/Helios

MINICONDA_ROOT="/mnt/bn/voyager-sg-l3/zhexiao.xiong/miniconda3"
PYTHON_BIN="$MINICONDA_ROOT/bin/python3.13"
ACCELERATE_BIN="$MINICONDA_ROOT/bin/accelerate"
PY_MM="$("$PYTHON_BIN" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")"
PY_SITE_PKGS="$MINICONDA_ROOT/lib/python${PY_MM}/site-packages"

# Force all Python tooling to use the same environment.
export PATH="$MINICONDA_ROOT/bin:${PATH}"
export PYTHONNOUSERSITE=1
hash -r

# Fix CUDA library paths and ensure helios module is importable
#
# IMPORTANT:
# Bind runtime CUDA libs to the same Python env used by training.
# Mixing in /usr/local/python3.10 nvidia libs can load an older cuDNN
# (missing symbol cudnnGetLibConfig) and crash torch workers.
export LD_LIBRARY_PATH="$PY_SITE_PKGS/nvidia/cusparselt/lib:${LD_LIBRARY_PATH}"
export PYTHONPATH="/mnt/bn/voyager-sg-l3/zhexiao.xiong/Helios:${PYTHONPATH}"
export WANDB_API_KEY="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"
export WANDB_BASE_URL="https://api.wandb.ai" #没有就是默认bytedance的？

# ============================================================
# 全局配置 / Global config（按需修改）
# ============================================================
LOCAL_MODELS="/mnt/bn/voyager-sg-l3/zhexiao.xiong/Helios/BestWishYSH"

DATA_ROOTS=(
    # "data/helios/seadance2_v2_helios_latents"
    "data/helios/seadance2_v3_helios_latents"
    # "data/helios/yume_training_helios_latents"
)

# 将 DATA_ROOTS 注入指定 yaml 的 instance_data_root 字段
patch_data_roots() {
    local cfg_file="$1"
    "$PYTHON_BIN" -c "
import yaml
with open('${cfg_file}') as f:
    cfg = yaml.safe_load(f)
cfg['data_config']['instance_data_root'] = [$(printf "'%s'," "${DATA_ROOTS[@]}")]
with open('${cfg_file}', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
"
}

# 1. 预编码 action embedding cache（一次性）
# python tools/offload_data/precompute_action_embeds.py \
#     --output_path data/helios/action_embeds_cache.pt

# 2. 转换 YUME 数据到 Helios 格式（使用原始完整视频 + TSV caption + per-chunk action 标签）
# torchrun --nproc_per_node 8 --master_port 9601 \
#     tools/offload_data/convert_sekai_to_helios.py \
#     --video_dir /mnt/bn/voyager-sg-l3/zhexiao.xiong/Helios/data/seadance2_yume_v3/video \
#     --action_dir /mnt/bn/voyager-sg-l3/zhexiao.xiong/Helios/data/seadance2_yume_v3/mp4_frame \
#     --tsv_path /mnt/bn/voyager-sg-l3/zhexiao.xiong/Helios/data/seadance2_yume_v3/world_model_action12_train_3000_simple1cam_actionfirst.tsv \
#     --output_dir data/helios/seadance2_v3_helios_latents

# 3. 训练（生成带时间戳的 exp 目录，注入 yaml）
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXP_DIR="/mnt/bn/voyager-sg-l3/zhexiao.xiong/Helios/ckpts/helios/stage1_action_${TIMESTAMP}"
mkdir -p "$EXP_DIR"

# 复制 yaml 并替换 output_dir / logging_dir / val_output_dir
VAL_DIR="/mnt/bn/voyager-sg-l3/zhexiao.xiong/Helios/output_helios/stage1_action_${TIMESTAMP}"
TMP_CONFIG="$EXP_DIR/train_config.yaml"
"$PYTHON_BIN" - <<PY
import yaml

base_cfg_path = "scripts/training/configs/stage_1_action_init.yaml"
with open(base_cfg_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

if not isinstance(cfg, dict):
    raise RuntimeError(f"Failed to load valid yaml from {base_cfg_path}: got {type(cfg).__name__}")

cfg["output_dir"] = "$EXP_DIR"
cfg["logging_dir"] = "$EXP_DIR/logs"
cfg.setdefault("validation_config", {})["val_output_dir"] = "$VAL_DIR"
cfg.setdefault("data_config", {})["instance_data_root"] = [$(printf "'%s'," "${DATA_ROOTS[@]}")]
cfg.setdefault("model_config", {})["pretrained_model_name_or_path"] = "$LOCAL_MODELS/Helios-Base"
cfg.setdefault("model_config", {})["transformer_model_name_or_path"] = "$LOCAL_MODELS/Helios-Base"

with open("$TMP_CONFIG", "w", encoding="utf-8") as f:
    yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
PY

# accelerate launch --config_file scripts/accelerate_configs/multi_node_example_zero2.yaml --main_process_port 9500 \ # 如果用deepspeed

accelerate launch --main_process_port 9500 \
    train_helios.py \
    --config "$TMP_CONFIG"

# ============================================================
# Stage 2: Multi-Stage Pyramid Refinement (在 Stage 1 收敛后执行)
# ============================================================
# STAGE1_CKPT 填 Stage 1 最终 checkpoint 路径
# STAGE1_CKPT="/mnt/bn/voyager-sg-l3/zhexiao.xiong/Helios/ckpts/helios/stage1_action_20260316_222852/checkpoint-2000"
# TIMESTAMP_S2=$(date +%Y%m%d_%H%M%S)
# EXP_DIR_S2="/mnt/bn/voyager-sg-l3/zhexiao.xiong/Helios/ckpts/helios/stage2_action_${TIMESTAMP_S2}"
# mkdir -p "$EXP_DIR_S2"

# TMP_CONFIG_S2="$EXP_DIR_S2/train_config.yaml"
# sed "s|output_dir:.*|output_dir: $EXP_DIR_S2|; s|logging_dir:.*|logging_dir: $EXP_DIR_S2/logs|; s|load_checkpoints_custom:.*|load_checkpoints_custom: true|" \
#     scripts/training/configs/stage_2_action_init.yaml > "$TMP_CONFIG_S2"
# sed -i "/load_checkpoints_custom:/a\\  load_model_path: \"$STAGE1_CKPT\"" "$TMP_CONFIG_S2"
# sed -i "/validation_config:/a\\  val_output_dir: /mnt/bn/voyager-sg-l3/zhexiao.xiong/Helios/output_helios/stage2_action_${TIMESTAMP_S2}" "$TMP_CONFIG_S2"
# sed -i "s|pretrained_model_name_or_path:.*|pretrained_model_name_or_path: \"$LOCAL_MODELS/Helios-Base\"|" "$TMP_CONFIG_S2"
# sed -i "s|transformer_model_name_or_path:.*|transformer_model_name_or_path: \"$LOCAL_MODELS/Helios-Base\"|" "$TMP_CONFIG_S2"
# patch_data_roots "$TMP_CONFIG_S2"

# # accelerate launch --config_file scripts/accelerate_configs/multi_node_example_zero2.yaml --main_process_port 9601 \ # 如果用deepspeed
# accelerate launch --main_process_port 9601 \
#     train_helios.py \
#     --config "$TMP_CONFIG_S2"

# ============================================================
# Stage 3: Adversarial Hierarchical Distillation (在 Stage 2 收敛后执行)
# ============================================================
# STAGE2_CKPT 填 Stage 2 最终 checkpoint 路径
# STAGE2_CKPT="/mnt/bn/voyager-sg-l3/zhexiao.xiong/Helios/ckpts/helios/stage2_action_20260318_004721/checkpoint-2000"
# TIMESTAMP_S3=$(date +%Y%m%d_%H%M%S)
# EXP_DIR_S3="/mnt/bn/voyager-sg-l3/zhexiao.xiong/Helios/ckpts/helios/stage3_action_${TIMESTAMP_S3}"
# mkdir -p "$EXP_DIR_S3"

# TMP_CONFIG_S3="$EXP_DIR_S3/train_config.yaml"
# sed "s|output_dir:.*|output_dir: $EXP_DIR_S3|; s|logging_dir:.*|logging_dir: $EXP_DIR_S3/logs|; s|load_checkpoints_custom:.*|load_checkpoints_custom: true|" \
#     scripts/training/configs/stage_3_action_post.yaml > "$TMP_CONFIG_S3"
# # generator 和 critic 都从 Stage 2 checkpoint 初始化
# sed -i "/load_checkpoints_custom:/a\\  load_model_path: \"$STAGE2_CKPT\"" "$TMP_CONFIG_S3"
# sed -i "/load_model_path:/a\\  critic_lora_name_or_path: \"$STAGE2_CKPT\"" "$TMP_CONFIG_S3"
# sed -i "/validation_config:/a\\  val_output_dir: /mnt/bn/voyager-sg-l3/zhexiao.xiong/Helios/output_helios/stage3_action_${TIMESTAMP_S3}" "$TMP_CONFIG_S3"
# sed -i "s|pretrained_model_name_or_path:.*|pretrained_model_name_or_path: \"$LOCAL_MODELS/Helios-Base\"|" "$TMP_CONFIG_S3"
# sed -i "s|transformer_model_name_or_path:.*|transformer_model_name_or_path: \"$LOCAL_MODELS/Helios-Base\"|" "$TMP_CONFIG_S3"
# sed -i "s|real_score_model_name_or_path:.*|real_score_model_name_or_path: \"$LOCAL_MODELS/Helios-Base\"|" "$TMP_CONFIG_S3"
# patch_data_roots "$TMP_CONFIG_S3"

# accelerate launch --config_file scripts/accelerate_configs/multi_node_example_zero2.yaml --main_process_port 9602 \
#     train_helios.py \
#     --config "$TMP_CONFIG_S3"

# ============================================================
# 4. 推理 — 见 infer.sh
# ============================================================


cd /mnt/bn/voyager-sg-l3/zhexiao.xiong
python runner.py

