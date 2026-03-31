"""
Strip camera-movement descriptions from pre-computed Helios .pt files and
re-encode the cleaned caption with T5.

Reads every .pt from --input_dir, writes a new .pt (same name) to --output_dir
with updated prompt_embed / prompt_raw.  VAE latents and everything else are
copied unchanged — this is pure text processing + T5, so it finishes in minutes.

Usage:
    torchrun --nproc_per_node 8 tools/offload_data/re_encode_prompt.py \
        --input_dir  data/helios/yume_training_helios_latents \
        --output_dir data/helios/yume_training_helios_latents_global
"""

import argparse
import os
import re
from typing import List, Optional, Union

import torch
import torch.distributed as dist
from helios.utils.utils_base import encode_prompt
from tqdm import tqdm
from transformers import AutoTokenizer, UMT5EncoderModel


# ---------------------------------------------------------------------------
# strip_camera_motion: regex-based removal of camera/viewer movement phrases
# ---------------------------------------------------------------------------

_MOTION_VERBS = (
    r"(?:advances?|moves?|pans?|tilts?|glides?|shifts?|adjusts?"
    r"|continues?|transitions?|progresses|follows?|maintains?"
    r"|approaches|turns?|swings?|tracks?|sweeps?|zooms?"
    r"|pulls?|pushes?|ascends?|descends?|retreats?|emerges?"
    r"|navigates?|crosses?|provides?|offers?|reveals?"
    r"|captures?|remains?)"
)
_MOTION_ADVERBS = r"(?:smoothly |steadily |gently |briefly |subtly |occasionally |then |slowly |further )?"

_OPENING_LOCATION_RE = re.compile(
    r"^(?:"
    r"(?:The (?:video|footage|clip) (?:begins|starts|opens) with )"
    r"|(?:The first-person perspective (?:begins|starts) with )"
    r"|(?:The camera " + _MOTION_VERBS + r"\s+)"
    r")?"
    r"(?:a |the )?(?:first-person |FPV |steady |smooth )?"
    r"(?:perspective |view |forward movement |movement |motion )?"
    r"(?:(?:of someone\s+)?(?:moving|walking|advancing|progressing|traveling|gliding)\s+)?"
    r"(?:steadily |smoothly |slowly |forward )?"
    r"(?:along|through|across|over|down|into|past|forward along|ahead along)?\s+"
    r"(?P<location>[^.]*?)"
    r"\.\s*",
    re.IGNORECASE,
)

_CAMERA_SENTENCE_RE = re.compile(
    r"(?<=[.!?])\s*"
    r"(?:"
    r"(?:The (?:camera|viewer|movement|journey|walk|ascent|descent|advance|progression))"
    r"|(?:As the (?:viewer|person|camera|walk|journey|movement)\s+"
    r"(?:progresses|continues|advances|moves|proceeds))"
    r"|(?:The (?:entire )?(?:sequence|journey|video|walk)\s+"
    r"(?:maintains|unfolds|continues|captures))"
    r"|(?:Throughout the (?:entire )?(?:sequence|journey|video|walk|clip))"
    r"|(?:(?:The )?[Cc]ontinuing (?:onward|forward|straight))"
    r")"
    r"[^.!?]*[.!?]",
    re.IGNORECASE,
)

_CAMERA_CLAUSE_RE = re.compile(
    r",?\s*(?:the camera|the viewer(?:'s gaze)?|the perspective)\s+"
    + _MOTION_ADVERBS + _MOTION_VERBS
    + r"[^,.\n]*",
    re.IGNORECASE,
)

_LEADING_MOTION_RE = re.compile(
    r"^(?:The (?:video|footage|clip) (?:begins|starts|opens) with )?"
    r"(?:a |the )?(?:first-person |FPV |steady |smooth )?"
    r"(?:perspective|view|forward movement|movement|motion)"
    r"(?:\s+(?:of someone\s+)?(?:moving|walking|advancing|progressing|traveling|gliding))"
    r"[^.]*\.\s*",
    re.IGNORECASE,
)

_MOTION_PHRASE_RE = re.compile(
    r",\s*(?:moving|advancing|walking|progressing|traveling|gliding|proceeding)"
    r"(?:\s+(?:steadily |smoothly |slowly |forward |ahead )?"
    r"(?:forward|ahead|along|through|across|onward|deeper|further|into))"
    r"[^,.\n]*",
    re.IGNORECASE,
)

_FPV_MOTION_RE = re.compile(
    r"\s*with a (?:first-person|FPV|steady)\s+(?:perspective|view)\s+"
    r"(?:of someone\s+)?(?:moving|walking|advancing|progressing)[^,.]*",
    re.IGNORECASE,
)


def strip_camera_motion(caption: str) -> str:
    """Remove camera/viewer movement descriptions from a Sekai-style caption."""

    location_prefix = ""
    m = _OPENING_LOCATION_RE.match(caption)
    if m:
        loc = m.group("location").strip()
        if len(loc) > 15:
            location_prefix = loc.rstrip(",. ") + ". "

    text = caption
    text = _LEADING_MOTION_RE.sub("", text)
    text = _CAMERA_SENTENCE_RE.sub("", text)
    text = _CAMERA_CLAUSE_RE.sub("", text)
    text = _MOTION_PHRASE_RE.sub("", text)
    text = _FPV_MOTION_RE.sub("", text)

    text = re.sub(r"\s*,\s*,", ",", text)
    text = re.sub(r"\.\s*\.", ".", text)
    text = re.sub(r"(?<=\.)\s*(?:As|And|But|While)\s*,", ".", text, flags=re.IGNORECASE)
    text = re.sub(r"\bAs\s*,\s*", "", text)
    text = re.sub(r"(?<=\.)\s*,\s*", ". ", text)
    text = re.sub(r"\.\s+(?:it|he|she|they)\s+" + _MOTION_VERBS + r"[^.]*\.", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*(?:it|he|she|they)\s+" + _MOTION_VERBS + r"[^.]*\.\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*[,;.]\s*", "", text)
    text = re.sub(r"^\s*capturing\b", "Capturing", text)
    text = re.sub(r"\.\s*\.", ".", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = text.strip()

    if location_prefix:
        loc_parts = [p.strip() for p in location_prefix.rstrip(". ").split(",") if len(p.strip()) > 8]
        already_present = any(p.lower() in text[:150].lower() for p in loc_parts)
        if not already_present:
            text = location_prefix + text

    if text and text[0].islower():
        text = text[0].upper() + text[1:]

    if not text or len(text) < 20:
        return caption

    return text


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="BestWishYSH/Helios-Base")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="T5 encoding batch size (text only, very lightweight)")
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()
    world_size = dist.get_world_size()
    global_rank = dist.get_rank()

    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer"
    )
    text_encoder = UMT5EncoderModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder",
        torch_dtype=torch.bfloat16,
    ).eval().requires_grad_(False).to(device)

    os.makedirs(args.output_dir, exist_ok=True)

    all_files = sorted([f for f in os.listdir(args.input_dir) if f.endswith(".pt")])
    my_files = all_files[global_rank::world_size]

    already_done = set(os.listdir(args.output_dir)) if os.path.exists(args.output_dir) else set()
    my_files = [f for f in my_files if f not in already_done]

    if global_rank == 0:
        print(f"Total .pt files: {len(all_files)}, this rank: {len(my_files)} remaining")

    # Two-phase batching:
    #   Phase 1 – scan batch_size files, collect (path, stripped_text) only (no heavy tensors in RAM)
    #   Phase 2 – T5-encode the batch of texts, then re-load each file, update embed, write out
    # This keeps peak memory at O(batch_size * text_length) instead of O(batch_size * 120MB).
    batch_in_paths = []
    batch_out_paths = []
    batch_texts = []
    n_stripped = 0

    pbar = tqdm(my_files, desc=f"[Rank {global_rank}]", disable=global_rank != 0)
    for filename in pbar:
        input_path = os.path.join(args.input_dir, filename)
        output_path = os.path.join(args.output_dir, filename)

        # Load only to extract prompt_raw, then release immediately
        try:
            data = torch.load(input_path, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"[Rank {global_rank}] WARNING: skipping corrupted file {filename}: {e}")
            continue
        raw = data.get("prompt_raw", "")
        original_raw = raw
        stripped = strip_camera_motion(raw) if raw else raw
        del data  # free the 120MB tensor now

        if stripped != original_raw:
            n_stripped += 1

        batch_in_paths.append(input_path)
        batch_out_paths.append(output_path)
        batch_texts.append((stripped, original_raw))

        if len(batch_texts) >= args.batch_size:
            _flush_batch(batch_in_paths, batch_out_paths, batch_texts,
                         tokenizer, text_encoder, device)
            batch_in_paths.clear()
            batch_out_paths.clear()
            batch_texts.clear()

    if batch_texts:
        _flush_batch(batch_in_paths, batch_out_paths, batch_texts,
                     tokenizer, text_encoder, device)

    dist.barrier()
    if global_rank == 0:
        total_out = len([f for f in os.listdir(args.output_dir) if f.endswith(".pt")])
        print(f"Done. {total_out} files in {args.output_dir}")

    dist.destroy_process_group()


def _flush_batch(in_paths, out_paths, texts_and_originals, tokenizer, text_encoder, device):
    """T5-encode a batch of stripped captions, then re-load each .pt, update, and write."""
    stripped_texts = [t for t, _ in texts_and_originals]

    with torch.no_grad():
        new_embeds, _ = encode_prompt(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            prompt=stripped_texts,
            device=device,
        )

    for i, (in_path, out_path) in enumerate(zip(in_paths, out_paths)):
        stripped, original_raw = texts_and_originals[i]
        try:
            data = torch.load(in_path, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"WARNING: skipping corrupted file during write phase {in_path}: {e}")
            continue
        data["prompt_embed"] = new_embeds[i].cpu().detach()
        data["prompt_raw_original"] = original_raw
        data["prompt_raw"] = stripped
        torch.save(data, out_path)
        del data


if __name__ == "__main__":
    main()
