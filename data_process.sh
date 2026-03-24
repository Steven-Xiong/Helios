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
SEKAI_CSV=/mnt/bn/voyager-sg-l3/zhexiao.xiong/zhexiao.xiong/data/Sekai-Project/train
torchrun --nproc_per_node 8 --master_port 9502 tools/offload_data/convert_sekai_to_helios.py \
    --video_dir /mnt/bn/voyager-sg-l3/zhexiao.xiong/zhexiao.xiong/data/yume_training/video \
    --action_dir /mnt/bn/voyager-sg-l3/zhexiao.xiong/zhexiao.xiong/data/yume_training/mp4_frame \
    --caption_csvs \
        "$SEKAI_CSV/sekai-real-walking-hq.csv" \
        "$SEKAI_CSV/sekai-game-walking.csv" \
        "$SEKAI_CSV/sekai-game-drone.csv" \
    --output_dir data/helios/yume_training_helios_latents \
    --num_workers 8


cd /mnt/bn/voyager-sg-l3/zhexiao.xiong
python runner.py
