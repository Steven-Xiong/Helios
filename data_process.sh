# python tools/offload_data/precompute_action_embeds.py \
#     --output_path data/helios/action_embeds_cache.pt


# ── seadance2_yume_v3 (existing) ──
# torchrun --nproc_per_node 8 --master_port 9601 tools/offload_data/convert_sekai_to_helios.py \
#     --video_dir /mnt/bn/voyager-sg-l3/zhexiao.xiong/zhexiao.xiong/data/seadance2_yume_v3/video \
#     --action_dir /mnt/bn/voyager-sg-l3/zhexiao.xiong/zhexiao.xiong/data/seadance2_yume_v3/mp4_frame \
#     --tsv_path /mnt/bn/voyager-sg-l3/zhexiao.xiong/zhexiao.xiong/data/seadance2_yume_v3/world_model_action12_train_3000_simple1cam_actionfirst.tsv \
#     --output_dir data/helios/seadance2_v3_helios_latents

# ── yume_training (real-walking + game-walking + game-drone) ──
# Requires: prepare_data.sh already ran to populate video/ and mp4_frame/
# SEKAI_CSV=/mnt/bn/voyager-sg-l3/zhexiao.xiong/zhexiao.xiong/data/Sekai-Project/train
# torchrun --nproc_per_node 8 --master_port 9502 tools/offload_data/convert_sekai_to_helios.py \
#     --video_dir /mnt/bn/voyager-sg-l3/zhexiao.xiong/zhexiao.xiong/data/yume_training/video \
#     --action_dir /mnt/bn/voyager-sg-l3/zhexiao.xiong/zhexiao.xiong/data/yume_training/mp4_frame \
#     --caption_csvs \
#         "$SEKAI_CSV/sekai-real-walking-hq.csv" \
#         "$SEKAI_CSV/sekai-game-walking.csv" \
#         "$SEKAI_CSV/sekai-game-drone.csv" \
#     --output_dir data/helios/yume_training_helios_latents \
#     --batch_size 8


# ── Re-encode: strip camera motion from captions ──
# Reads .pt from yume_training_helios_latents, strips motion text,
# re-encodes with T5, writes to yume_training_helios_latents_global.
# Original files are NOT modified.
torchrun --nproc_per_node 8 --master_port 9503 tools/offload_data/re_encode_prompt.py \
    --input_dir  data/helios/yume_training_helios_latents \
    --output_dir data/helios/yume_training_helios_latents_global \
    --batch_size 32

# ── Extract prompts from latent data for ODE pair generation ──
# Scans all .pt files, grabs prompt_raw field, one per line.
# python -c "
# import os, glob, torch
# data_roots = [
#     'data/helios/seadance2_v3_helios_latents',
#     'data/helios/yume_training_helios_latents',
# ]
# prompts = []
# for root in data_roots:
#     for pt_path in sorted(glob.glob(os.path.join(root, '*.pt'))):
#         try:
#             d = torch.load(pt_path, map_location='cpu', weights_only=False)
#             p = d.get('prompt_raw', '')
#             if isinstance(p, str) and p.strip():
#                 prompts.append(p.strip())
#         except Exception:
#             pass
# os.makedirs('data/helios', exist_ok=True)
# with open('data/helios/ode_prompts.txt', 'w') as f:
#     for p in prompts:
#         f.write(p + '\n')
# print(f'Extracted {len(prompts)} prompts → data/helios/ode_prompts.txt')
# "


# ── Generate ODE pairs using teacher model (Stage 2 merged) ──
# Teacher = merged S2-post model.  Generates denoising trajectories.
# Requires: merged S2-post transformer + prompt txt file from above.
# TEACHER_PATH=/path/to/stage2_post_YYYYMMDD/merged_transformer
# torchrun --nproc_per_node 8 --master_port 9503 \
#     tools/offload_data/get_ode-pairs.py \
#     --transformer_path "$TEACHER_PATH" \
#     --base_model_path BestWishYSH/Helios-Base \
#     --subfolder "" \
#     --prompt_txt_files data/helios/ode_prompts.txt \
#     --output_dirs  data/helios/ode_pairs \
#     --use_dynamic_shifting \
#     --time_shift_type "linear" \
#     --use_default_loader \
#     --is_enable_stage2 \
#     --num_frames 165
# NOTE: This step is automated in run.sh between S2-post and S3-ode.

cd /mnt/bn/voyager-sg-l3/zhexiao.xiong
python runner.py
