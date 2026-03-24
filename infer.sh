#!/usr/bin/env bash
# Must run from Helios root directory
cd /mnt/bn/voyager-sg-l3/zhexiao.xiong/Helios

MINICONDA_ROOT="/mnt/bn/voyager-sg-l3/zhexiao.xiong/miniconda3"
PYTHON_BIN="$MINICONDA_ROOT/bin/python3.13"
PY_MM="$("$PYTHON_BIN" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")"
PY_SITE_PKGS="$MINICONDA_ROOT/lib/python${PY_MM}/site-packages"

export PATH="$MINICONDA_ROOT/bin:${PATH}"
export PYTHONNOUSERSITE=1
hash -r

export LD_LIBRARY_PATH="$PY_SITE_PKGS/nvidia/cudnn/lib:$PY_SITE_PKGS/nvidia/cusparselt/lib:${LD_LIBRARY_PATH}"
export PYTHONPATH="/mnt/bn/voyager-sg-l3/zhexiao.xiong/Helios:${PYTHONPATH}"

LOCAL_MODELS="/mnt/bn/voyager-sg-l3/zhexiao.xiong/Helios/BestWishYSH"

# ============================================================
# 4a. 推理 — Stage 1 (Helios-Base, 50 steps)
# ============================================================
CKPT="ckpts/helios/stage1_action_20260322_020120/checkpoint-2500"
TIMESTAMP_INFER=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="./output_helios/eval_seadance2_stage1_${TIMESTAMP_INFER}"

torchrun --nproc_per_node 8 --master_port 9605 infer_helios.py \
    --base_model_path "$LOCAL_MODELS/Helios-Base" \
    --transformer_path "$LOCAL_MODELS/Helios-Base" \
    --sample_type i2v \
    --image_prompt_csv_path data/seadance2_yume_v3/world_model_action12_train_3000_simple1cam_actionfirst.tsv \
    --base_image_prompt_path data/seadance2_yume_v3/first_frame\
    --action_embeds_cache data/helios/action_embeds_cache.pt \
    --lora_path "$CKPT" \
    --partial_path "$CKPT/transformer_partial.pth" \
    --output_folder "$OUTPUT_DIR" \
    --num_frames 321 \
    --height 384 \
    --width 640 \
    --num_inference_steps 50 \
    --guidance_scale 5.0 \
    --fps 24

# ============================================================
# 4b. 推理 — Stage 2 (pyramid 20+20+20)
# ============================================================
# CKPT_S2="ckpts/helios/stage2_action_20260318_004721/checkpoint-2000"
# TIMESTAMP_INFER_S2=$(date +%Y%m%d_%H%M%S)
# OUTPUT_DIR_S2="./output_helios/eval_seadance2_stage2_${TIMESTAMP_INFER_S2}"

# torchrun --nproc_per_node 8 --master_port 9605 infer_helios.py \
#     --base_model_path "$LOCAL_MODELS/Helios-Mid" \
#     --transformer_path "$LOCAL_MODELS/Helios-Base" \
#     --sample_type i2v \
#     --image_prompt_csv_path data/seadance2_yume_test_12classv2/test_eval.csv \
#     --base_image_prompt_path data/seadance2_yume_test_12classv2/first_frame \
#     --action_embeds_cache data/helios/action_embeds_cache.pt \
#     --lora_path "$CKPT_S2" \
#     --partial_path "$CKPT_S2/transformer_partial.pth" \
#     --output_folder "$OUTPUT_DIR_S2" \
#     --num_frames 321 \
#     --height 384 \
#     --width 640 \
#     --guidance_scale 5.0 \
#     --is_enable_stage2 \
#     --pyramid_num_inference_steps_list 20 20 20 \
#     --use_zero_init \
#     --zero_steps 1 \
#     --fps 24

# ============================================================
# 4c. 推理 — Stage 3 (pyramid 2+2+2)
# ============================================================
# CKPT_S3="/mnt/bn/voyager-sg-l3/zhexiao.xiong/Helios/ckpts/helios/stage3_action_20260318_211239/checkpoint-1000"
# TIMESTAMP_INFER_S3=$(date +%Y%m%d_%H%M%S)
# OUTPUT_DIR_S3="./output_helios/eval_seadance2_stage3_${TIMESTAMP_INFER_S3}"

# torchrun --nproc_per_node 8 --master_port 9605 infer_helios.py \
#     --base_model_path "$LOCAL_MODELS/Helios-Distilled" \
#     --transformer_path "$LOCAL_MODELS/Helios-Base" \
#     --sample_type i2v \
#     --image_prompt_csv_path data/seadance2_yume_test_12classv2/test_eval.csv \
#     --base_image_prompt_path data/seadance2_yume_test_12classv2/first_frame \
#     --action_embeds_cache data/helios/action_embeds_cache.pt \
#     --lora_path "$CKPT_S3" \
#     --partial_path "$CKPT_S3/transformer_partial.pth" \
#     --output_folder "$OUTPUT_DIR_S3" \
#     --num_frames 321 \
#     --height 384 \
#     --width 640 \
#     --guidance_scale 1.0 \
#     --is_enable_stage2 \
#     --pyramid_num_inference_steps_list 2 2 2 \
#     --is_amplify_first_chunk \
#     --fps 24



cd /mnt/bn/voyager-sg-l3/zhexiao.xiong
python runner.py
