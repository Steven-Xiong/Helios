# Must run from Helios root directory
cd /mnt/bn/voyager-sg-l3/zhexiao.xiong/Helios

export LD_LIBRARY_PATH="/usr/local/lib/python3.10/site-packages/cusparselt/lib:${LD_LIBRARY_PATH}"
export PYTHONPATH="/mnt/bn/voyager-sg-l3/zhexiao.xiong/Helios:${PYTHONPATH}"

# ============================================================
# 原版推理（无 LoRA）—— 用于对比基线，确认各阶段原始生成质量
# ============================================================

# ============================================================
# Org-1. Helios-Base（Stage 1 对应基座，50 steps）
# ============================================================
TIMESTAMP_BASE=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR_BASE="./output_helios/eval_seadance2_org_base_onseadancev3train${TIMESTAMP_BASE}"

torchrun --nproc_per_node 8 --master_port 9401 infer_helios.py \
    --base_model_path "/mnt/bn/voyager-sg-l3/zhexiao.xiong/Helios/BestWishYSH/Helios-Base" \
    --transformer_path "/mnt/bn/voyager-sg-l3/zhexiao.xiong/Helios/BestWishYSH/Helios-Base" \
    --sample_type i2v \
    --image_prompt_csv_path /mnt/bn/voyager-sg-l3/zhexiao.xiong/Helios/data/seadance2_yume_v3/world_model_action12_train_3000_simple1cam_actionfirst.tsv \
    --base_image_prompt_path /mnt/bn/voyager-sg-l3/zhexiao.xiong/Helios/data/seadance2_yume_v3/first_frame \
    --output_folder "$OUTPUT_DIR_BASE" \
    --num_frames 321 \
    --height 384 \
    --width 640 \
    --num_inference_steps 50 \
    --guidance_scale 5.0 \
    --use_zero_init \
    --zero_steps 1 \
    --fps 24

# ============================================================
# Org-2. Helios-Mid（Stage 2 对应，pyramid 20+20+20）
# ============================================================
# TIMESTAMP_MID=$(date +%Y%m%d_%H%M%S)
# OUTPUT_DIR_MID="./output_helios/eval_seadance2_org_mid_${TIMESTAMP_MID}"

# torchrun --nproc_per_node 8 --master_port 9602 infer_helios.py \
#     --base_model_path "BestWishYsh/Helios-Mid" \
#     --transformer_path "BestWishYsh/Helios-Mid" \
#     --sample_type i2v \
#     --image_prompt_csv_path data/seadance2_yume_test_12classv2/test_eval.csv \
#     --base_image_prompt_path data/seadance2_yume_test_12classv2/first_frame \
#     --output_folder "$OUTPUT_DIR_MID" \
#     --num_frames 321 \
#     --height 384 \
#     --width 640 \
#     --guidance_scale 5.0 \
#     --is_enable_stage2 \
#     --pyramid_num_inference_steps_list 20 20 20 \
#     --use_zero_init \
#     --zero_steps 1 \
#     --fps 24

# # ============================================================
# # Org-3. Helios-Distilled（Stage 3 对应，pyramid 2+2+2）
# # ============================================================
# TIMESTAMP_DIST=$(date +%Y%m%d_%H%M%S)
# OUTPUT_DIR_DIST="./output_helios/eval_seadance2_org_distilled_${TIMESTAMP_DIST}"

# torchrun --nproc_per_node 8 --master_port 9603 infer_helios.py \
#     --base_model_path "BestWishYsh/Helios-Distilled" \
#     --transformer_path "BestWishYsh/Helios-Distilled" \
#     --sample_type i2v \
#     --image_prompt_csv_path data/seadance2_yume_test_12classv2/test_eval.csv \
#     --base_image_prompt_path data/seadance2_yume_test_12classv2/first_frame \
#     --output_folder "$OUTPUT_DIR_DIST" \
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