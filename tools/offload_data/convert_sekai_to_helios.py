"""
Convert YUME Sekai dataset to Helios .pt format using ORIGINAL full-length videos.

Data layout:
  data/seadance2_yume/
    video/              ← original full videos (e.g. rotate_0096.mp4, 361 frames)
    mp4_frame/          ← per-action 33-frame clips with .txt action labels
      Keys_W_Mouse_→/
        rotate_0096_..._frames_00000-00032.txt  → Keys: W, Mouse: →

This script:
  1. Loads original full videos from video/ (361 frames, well above the 121-frame minimum)
  2. Collects per-segment action labels from mp4_frame/ .txt files
  3. VAE-encodes the full video in 33-frame chunks
  4. Builds caption from the dominant action of the video
  5. Saves .pt with per-chunk action labels for training

Usage:
    torchrun --nproc_per_node 8 convert_sekai_to_helios.py \
        --video_dir data/seadance2_yume/video \
        --action_dir data/seadance2_yume/mp4_frame \
        --output_dir data/helios/sekai_helios_latents \
        --pretrained_model_name_or_path BestWishYSH/Helios-Base
"""

import argparse
import glob
import os
import re
from collections import defaultdict

import torch
import torch.distributed as dist
import torchvision.io
from helios.utils.utils_base import encode_prompt
from tqdm import tqdm
from transformers import AutoTokenizer, UMT5EncoderModel

from diffusers import AutoencoderKLWan
from diffusers.training_utils import free_memory


DEFAULT_SCENE_CAPTION = "This video depicts a first-person view (FPV) egocentric scene."



MOUSE_DIR_TO_SYMBOL = {
    "Up": "↑", "Down": "↓", "Left": "←", "Right": "→",
    "Up_Left": "↑←", "Up_Right": "↑→", "Down_Left": "↓←", "Down_Right": "↓→",
    "·": "·",
}


def parse_action_from_dirname(dirname):
    """Parse keys and mouse from directory name like 'Keys_S_A_Mouse_Down_Right'."""
    m = re.match(r"Keys_(.+)_Mouse_(.+)", dirname)
    if not m:
        return "None", "·"
    keys_raw = m.group(1)   # e.g. "S_A" or "W" or "None"
    mouse_raw = m.group(2)  # e.g. "Down_Right" or "·" or "Up"

    keys = keys_raw.replace("_", "+") if keys_raw != "None" else "None"
    mouse = MOUSE_DIR_TO_SYMBOL.get(mouse_raw, "·")
    return keys, mouse


def collect_action_labels(action_dir, video_names=None):
    """
    Build mapping: video_name -> [(start_frame, end_frame, keys, mouse), ...]
    Parses action from directory names (fast, no file reads) and frame ranges from filenames.

    If video_names (set of stems from video_dir) is given, resolve video_prefix
    by matching against known names; otherwise fall back to the legacy 2-part split.
    """
    actions_by_video = defaultdict(list)

    for category_dir in sorted(os.listdir(action_dir)):
        cat_path = os.path.join(action_dir, category_dir)
        if not os.path.isdir(cat_path):
            continue

        keys, mouse = parse_action_from_dirname(category_dir)

        for filename in os.listdir(cat_path):
            if not filename.endswith(".mp4"):
                continue
            basename = filename[:-4]
            m = re.match(r"(.+)_frames_(\d+)-(\d+)", basename)
            if not m:
                continue

            video_prefix = m.group(1)
            start_frame = int(m.group(2))
            end_frame = int(m.group(3))

            video_name = _resolve_video_name(video_prefix, video_names)
            actions_by_video[video_name].append((start_frame, end_frame, keys, mouse))

    for video_name in actions_by_video:
        actions_by_video[video_name].sort(key=lambda x: x[0])

    return actions_by_video


def _resolve_video_name(video_prefix, video_names):
    """Map a clip's video_prefix to an actual video name.

    Tries exact match first (covers yume_training 3-part names like
    'AJw3NeaFtXE_0088880_0090680'), then 2-part prefix (covers seadance2
    names like 'close_0001'), then falls back to the full prefix.
    """
    if video_names is not None:
        if video_prefix in video_names:
            return video_prefix
        two_part = "_".join(video_prefix.split("_")[:2])
        if two_part in video_names:
            return two_part
    else:
        return "_".join(video_prefix.split("_")[:2])
    return video_prefix


def get_action_for_frame(actions, frame_idx):
    """Find the action label covering a given frame index."""
    for start, end, keys, mouse in actions:
        if start <= frame_idx <= end:
            return keys, mouse
    return "None", "·"


def load_video_captions_tsv(tsv_path):
    """
    Load the TSV file mapping video_name -> (action_class, prompt).
    TSV columns: action_class, prompt, video_name
    """
    captions = {}
    if tsv_path is None or not os.path.exists(tsv_path):
        return captions
    with open(tsv_path, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                action_class, prompt, video_name = parts[0], parts[1], parts[2]
                video_id = video_name.replace(".mp4", "")
                captions[video_id] = {"action_class": action_class, "prompt": prompt}
    return captions


def load_sekai_captions_csv(csv_paths):
    """Load Sekai-Project CSV annotations into the same dict format.

    CSV columns: videoFile, cameraFile, caption, location, scene,
                 crowdDensity, weather, timeOfDay
    """
    import csv

    captions = {}
    if csv_paths is None:
        return captions
    for csv_path in csv_paths:
        if not os.path.exists(csv_path):
            print(f"[WARN] caption CSV not found: {csv_path}")
            continue
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                video_id = row["videoFile"].replace(".mp4", "")
                captions[video_id] = {
                    "action_class": row.get("scene", "outdoor"),
                    "prompt": row["caption"],
                }
    return captions


def main():
    parser = argparse.ArgumentParser(description="Convert YUME Sekai data to Helios .pt format")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory with original full videos")
    parser.add_argument("--action_dir", type=str, required=True, help="Directory with mp4_frame action clips")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--tsv_path", type=str, default=None, help="TSV with per-video action_class and prompt")
    parser.add_argument("--caption_csvs", type=str, nargs="*", default=None,
                        help="Sekai-Project CSV annotation files (videoFile,caption,...)")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="BestWishYSH/Helios-Base")
    parser.add_argument("--target_height", type=int, default=384)
    parser.add_argument("--target_width", type=int, default=640)
    parser.add_argument("--max_frames", type=int, default=0,
                        help="Max frames to use per video (0 = no limit)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Number of 33-frame chunks to VAE-encode at once")
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()
    world_size = dist.get_world_size()
    global_rank = dist.get_rank()

    weight_dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = UMT5EncoderModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", torch_dtype=weight_dtype
    )
    vae = AutoencoderKLWan.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", torch_dtype=torch.float32
    )

    latents_mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(device, weight_dtype)
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(device, weight_dtype)

    vae.eval().requires_grad_(False).to(device)
    text_encoder.eval().requires_grad_(False).to(device)

    os.makedirs(args.output_dir, exist_ok=True)

    # Load per-video captions from TSV and/or Sekai CSV
    video_captions = load_video_captions_tsv(args.tsv_path)
    csv_captions = load_sekai_captions_csv(args.caption_csvs)
    csv_captions.update(video_captions)  # TSV takes priority on overlap
    video_captions = csv_captions
    if global_rank == 0:
        print(f"Loaded captions for {len(video_captions)} videos (TSV + CSV)")

    # Build set of known video stems from video_dir for robust name matching
    video_names = {
        os.path.splitext(f)[0]
        for f in os.listdir(args.video_dir)
        if f.endswith(".mp4")
    }

    # Collect per-segment keyboard/mouse action labels from mp4_frame/
    if global_rank == 0:
        print("Collecting action labels from mp4_frame/...")
    actions_by_video = collect_action_labels(args.action_dir, video_names=video_names)
    if global_rank == 0:
        print(f"Found action labels for {len(actions_by_video)} videos")

    # List all original videos
    all_videos = sorted([
        os.path.join(args.video_dir, f) for f in os.listdir(args.video_dir) if f.endswith(".mp4")
    ])
    if global_rank == 0:
        print(f"Found {len(all_videos)} original videos in {args.video_dir}")

    my_videos = all_videos[global_rank::world_size]

    latent_window_size = 9
    frame_window_size = (latent_window_size - 1) * 4 + 1  # 33
    height, width = args.target_height, args.target_width

    # Pre-filter already processed videos to avoid wasting worker time
    _suffix_re = re.compile(r"_\d+_\d+_\d+\.pt$")
    existing_video_names = set()
    if os.path.exists(args.output_dir):
        for f in os.listdir(args.output_dir):
            m = _suffix_re.search(f)
            if m:
                existing_video_names.add(f[:m.start()])
    pending_videos = [
        v for v in my_videos
        if os.path.splitext(os.path.basename(v))[0] not in existing_video_names
    ]
    if global_rank == 0:
        print(f"Skipped {len(my_videos) - len(pending_videos)} already-processed, "
              f"{len(pending_videos)} remaining")

    if len(pending_videos) == 0:
        dist.barrier()
        dist.destroy_process_group()
        if global_rank == 0:
            print("Nothing to do — all videos already processed.")
        return

    pbar = tqdm(pending_videos, desc=f"[Rank {global_rank}]", disable=global_rank != 0)
    for video_path in pbar:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        free_memory()

        try:
            video, _, info = torchvision.io.read_video(video_path, pts_unit="sec")
        except Exception as e:
            print(f"[Rank {global_rank}] Error loading {video_path}: {e}")
            continue

        num_frames = video.shape[0]
        if num_frames < frame_window_size:
            del video
            continue

        if args.max_frames > 0 and num_frames > args.max_frames:
            video = video[:args.max_frames]
            num_frames = args.max_frames

        output_path = os.path.join(args.output_dir, f"{video_name}_{num_frames}_{height}_{width}.pt")
        if os.path.exists(output_path):
            del video
            continue

        pixel_values = video.permute(0, 3, 1, 2).float() / 127.5 - 1.0
        del video
        if pixel_values.shape[2] != height or pixel_values.shape[3] != width:
            pixel_values = torch.nn.functional.interpolate(
                pixel_values, size=(height, width), mode="bilinear", align_corners=False
            )
        pixel_values = pixel_values.permute(1, 0, 2, 3)  # (C, T, H, W)

        actions = actions_by_video.get(video_name, [])
        tsv_info = video_captions.get(video_name, None)
        if tsv_info is not None:
            caption = tsv_info["prompt"]
            action_class = tsv_info["action_class"]
        else:
            caption = DEFAULT_SCENE_CAPTION
            action_class = video_name.split("_")[0]

        with torch.no_grad():
            pixel_values = pixel_values.unsqueeze(0).to(dtype=vae.dtype, device=device)
            num_chunks = num_frames // frame_window_size
            history_latent_list = []
            chunk_actions = []

            for i in range(num_chunks):
                mid_frame = i * frame_window_size + frame_window_size // 2
                ck, cm = get_action_for_frame(actions, mid_frame)
                chunk_actions.append({"keys": ck, "mouse": cm})

            for batch_start in range(0, num_chunks, args.batch_size):
                batch_end = min(batch_start + args.batch_size, num_chunks)
                chunks = []
                for i in range(batch_start, batch_end):
                    start = i * frame_window_size
                    end = start + frame_window_size
                    chunks.append(pixel_values[0, :, start:end, :, :])
                chunk_batch = torch.stack(chunks, dim=0)  # (B, C, 33, H, W)
                batch_latents = vae.encode(chunk_batch).latent_dist.sample()
                batch_latents = (batch_latents - latents_mean) * latents_std
                for j in range(batch_latents.shape[0]):
                    history_latent_list.append(batch_latents[j:j+1])
                del chunk_batch, batch_latents

            vae_latent = torch.stack(history_latent_list, dim=1)

            prompt_embed, _ = encode_prompt(
                tokenizer=tokenizer, text_encoder=text_encoder, prompt=[caption], device=device
            )

            first_frame = (pixel_values[0, :, 0, :, :] * 127.5 + 127.5).clamp(0, 255).to(torch.uint8).cpu()

        to_save = {
            "vae_latent": vae_latent[0].cpu().detach(),
            "prompt_embed": prompt_embed[0].cpu().detach(),
            "prompt_raw": caption,
            "first_frames_image": first_frame,
            "action_class": action_class,
            "chunk_actions": chunk_actions,
        }

        try:
            torch.save(to_save, output_path)
            if global_rank == 0:
                pbar.set_postfix({"video": video_name, "frames": num_frames, "chunks": num_chunks})
        except Exception as e:
            print(f"[Rank {global_rank}] Error saving {output_path}: {e}")

        del pixel_values, vae_latent, prompt_embed, to_save
        free_memory()

    dist.barrier()
    dist.destroy_process_group()
    if global_rank == 0:
        print("Done.")


if __name__ == "__main__":
    main()
