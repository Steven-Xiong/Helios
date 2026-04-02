import functools
import importlib
import os
import re


os.environ["HF_ENABLE_PARALLEL_LOADING"] = "yes"
os.environ["HF_PARALLEL_LOADING_WORKERS"] = "8"

import argparse
import json
import time

import pandas as pd
import torch
import torch.distributed as dist
from tqdm import tqdm

if importlib.util.find_spec("torch_npu") is not None:
    import torch_npu
else:
    torch_npu = None

from helios.diffusers_version.pipeline_helios_diffusers import HeliosPipeline
from helios.diffusers_version.scheduling_helios_diffusers import HeliosScheduler
from helios.diffusers_version.transformer_helios_diffusers import HeliosTransformer3DModel
from helios.modules.helios_kernels import (
    replace_all_norms_with_flash_norms,
    replace_rmsnorm_with_fp32,
    replace_rope_with_flash_rope,
)
from helios.utils.utils_base import load_extra_components

from diffusers import ContextParallelConfig
from diffusers.models import AutoencoderKLWan
from diffusers.utils import export_to_video, load_image, load_video


def parse_args():
    parser = argparse.ArgumentParser(description="Generate video with model")

    # === Model paths ===
    parser.add_argument("--base_model_path", type=str, default="BestWishYsh/Helios-Base")
    parser.add_argument(
        "--transformer_path",
        type=str,
        default="BestWishYsh/Helios-Base",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--partial_path",
        type=str,
        default=None,
    )
    parser.add_argument("--output_folder", type=str, default="./output_helios")
    parser.add_argument("--enable_compile", action="store_true")
    parser.add_argument(
        "--attention_backend",
        type=str,
        default="auto",
        choices=["auto", "_flash_3", "flash", "_native_flash", "native"],
        help="Attention backend. 'auto' tries _flash_3 → flash → _native_flash → native.",
    )

    # === Generation parameters ===
    # environment
    parser.add_argument(
        "--sample_type",
        type=str,
        default="t2v",
        choices=["t2v", "i2v", "v2v"],
    )
    parser.add_argument(
        "--weight_dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="Data type for model weights.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for random number generator.")
    # base
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--num_frames", type=int, default=99)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    # cfg zero
    parser.add_argument("--use_zero_init", action="store_true")
    parser.add_argument("--zero_steps", type=int, default=1)
    # stage 1
    parser.add_argument("--num_latent_frames_per_chunk", type=int, default=9)
    # stage 2
    parser.add_argument("--is_enable_stage2", action="store_true")
    parser.add_argument("--pyramid_num_inference_steps_list", type=int, nargs="+", default=[20, 20, 20])
    # stage 3
    parser.add_argument("--is_skip_first_chunk", action="store_true")
    parser.add_argument("--is_amplify_first_chunk", action="store_true")

    # === Prompts ===
    parser.add_argument("--use_interpolate_prompt", action="store_true")
    parser.add_argument("--interpolation_steps", type=int, default=3)
    parser.add_argument("--interpolate_time", type=int, default=7)
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--image_noise_sigma_min", type=float, default=0.111, help="Balance motion amplitude and visual consistency"
    )
    parser.add_argument(
        "--image_noise_sigma_max", type=float, default=0.135, help="Balance motion amplitude and visual consistency"
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--video_noise_sigma_min", type=float, default=0.111, help="Balance motion amplitude and visual consistency"
    )
    parser.add_argument(
        "--video_noise_sigma_max", type=float, default=0.135, help="Balance motion amplitude and visual consistency"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A dynamic time-lapse video showing the rapidly moving scenery from the window of a speeding train. The camera captures various elements such as lush green fields, towering trees, quaint countryside houses, and distant mountain ranges passing by quickly. The train window frames the view, adding a sense of speed and motion as the landscape rushes past. The camera remains static but emphasizes the fast-paced movement outside. The overall atmosphere is serene yet exhilarating, capturing the essence of travel and exploration. Medium shot focusing on the train window and the rushing scenery beyond.",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
    )
    parser.add_argument(
        "--prompt_txt_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--base_image_prompt_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--image_prompt_csv_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--interactive_prompt_csv_path",
        type=str,
        default=None,
    )

    # === Action Conditioning ===
    parser.add_argument("--max_samples", type=int, default=0, help="Max samples to run (0 = all)")
    parser.add_argument("--action_embeds_cache", type=str, default=None, help="Path to action_embeds_cache.pt")
    parser.add_argument("--action_keys", type=str, default=None, help="Keyboard action (e.g. W, W+A, None)")
    parser.add_argument("--action_mouse", type=str, default=None, help="Mouse action (e.g. →, ↑←, ·)")
    parser.add_argument("--action_txt_path", type=str, default=None,
                        help="Per-chunk action txt. Each line = one chunk, with (keys) and (mouse) in parentheses.")

    # === Context parallelism ===
    # Please refer to https://huggingface.co/docs/diffusers/v0.37.0/en/training/distributed_inference#context-parallelism
    parser.add_argument("--enable_parallelism", action="store_true")
    parser.add_argument(
        "--cp_backend",
        type=str,
        choices=["ring", "ulysses", "unified", "ulysses_anything"],
        default="ulysses",
        help="Context parallel backend to use.",
    )

    # === Group-Offloading ===
    # Please refer to https://huggingface.co/docs/diffusers/v0.37.0/en/optimization/memory#group-offloading
    parser.add_argument("--enable_low_vram_mode", action="store_true")
    parser.add_argument(
        "--group_offloading_type",
        type=str,
        choices=["leaf_level", "block_level"],
        default="leaf_level",
        help="Specifies the granularity for group CPU offloading. Choose between 'leaf_level' (individual modules) or 'block_level' (entire blocks).",
    )
    parser.add_argument(
        "--num_blocks_per_group",
        type=str,
        default="4",
        help="The number of blocks to bundle together in each offloading group. Only relevant when using block-level offloading.",
    )

    return parser.parse_args()


@functools.lru_cache(maxsize=1)
def _resolve_attention_backend() -> str:
    """Probe once per process and cache the best available attention backend."""
    import torch as _torch
    cuda_major = _torch.cuda.get_device_capability()[0]
    candidates = (
        ["_flash_3", "flash", "_native_flash", "native"] if cuda_major >= 9
        else ["flash", "_native_flash", "native"]
    )
    for backend in candidates:
        try:
            # Import diffusers check function directly to avoid building a full model
            from diffusers.models.attention_dispatch import _check_attention_backend_requirements
            from diffusers.models.attention_dispatch import AttentionBackendName
            _check_attention_backend_requirements(AttentionBackendName(backend))
            return backend
        except Exception:
            continue
    return "native"


def read_prompt_table(path: str) -> pd.DataFrame:
    """Read a prompt table (CSV or TSV) and normalize to a common schema.

    Output always has at least:
      - id         : unique sample identifier (str)
      - prompt     : generation prompt (str)
      - image_name : first-frame filename relative to base_image_prompt_path (str, may be empty)

    Supported input formats:
      CSV  columns: id, prompt, image_name, [action_class, action_keys, action_mouse, ...]
      TSV  columns: action_class, prompt, video_name, [action_keys, action_mouse, ...]
      Sekai CSV   : videoFile, caption, [cameraFile, location, scene, ...]

    Multiple paths can be separated by commas to merge several CSV/TSV files.
    """
    paths = [p.strip() for p in path.split(",")]
    frames = []
    for p in paths:
        suffix = os.path.splitext(p)[1].lower()
        if suffix in {".tsv", ".tab"}:
            frames.append(pd.read_csv(p, sep="\t"))
        elif suffix == ".csv":
            frames.append(pd.read_csv(p, sep=","))
        else:
            frames.append(pd.read_csv(p, sep=None, engine="python"))
    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]

    # Sekai CSV format: videoFile → video_name, caption → prompt
    if "videoFile" in df.columns:
        rename_map = {"videoFile": "video_name", "caption": "prompt"}
        if "caption_original" in df.columns:
            rename_map["caption_original"] = "prompt_original"
        df = df.rename(columns=rename_map)

    # TSV / Sekai format uses video_name instead of id / image_name — normalize here
    if "video_name" in df.columns and "id" not in df.columns:
        stems = df["video_name"].astype(str).map(
            lambda x: os.path.splitext(os.path.basename(x))[0]
        )
        df = df.copy()
        df["id"] = stems
        if "image_name" not in df.columns:
            df["image_name"] = stems + ".jpg"

    # Ensure image_name column always exists
    if "image_name" not in df.columns:
        df["image_name"] = ""

    return df


def main():
    args = parse_args()

    assert not (args.enable_low_vram_mode and args.enable_compile), (
        "enable_low_vram_mode and enable_compile cannot be used together."
    )

    if args.weight_dtype == "fp32":
        args.weight_dtype = torch.float32
    elif args.weight_dtype == "fp16":
        args.weight_dtype = torch.float16
    else:
        args.weight_dtype = torch.bfloat16

    os.makedirs(args.output_folder, exist_ok=True)

    if dist.is_available() and "RANK" in os.environ:
        if args.cp_backend == "ulysses_anything":
            dist.init_process_group(backend="cpu:gloo,cuda:nccl")
        else:
            dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        device = torch.device("cuda", rank % torch.cuda.device_count())
        world_size = dist.get_world_size()
        torch.cuda.set_device(device)
        assert world_size == 1 or not args.enable_low_vram_mode, "enable_low_vram_mode is only for single GPU."
    else:
        rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        world_size = 1

    prompt = None
    image_path = None
    video_path = None
    interpolate_time_list = None
    if args.sample_type == "t2v" and args.prompt is None:
        prompt = "An extreme close-up of an gray-haired man with a beard in his 60s, he is deep in thought pondering the history of the universe as he sits at a cafe in Paris, his eyes focus on people offscreen as they walk as he sits mostly motionless, he is dressed in a wool coat suit coat with a button-down shirt , he wears a brown beret and glasses and has a very professorial appearance, and the end he offers a subtle closed-mouth smile as if he found the answer to the mystery of life, the lighting is very cinematic with the golden light and the Parisian streets and city in the background, depth of field, cinematic 35mm film."
    elif args.sample_type == "i2v" and (args.image_path is None and args.prompt is None):
        image_path = (
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg"
        )
        prompt = "An astronaut hatching from an egg, on the surface of the moon, the darkness and depth of space realised in the background. High quality, ultrarealistic detail and breath-taking movie-like camera shot."
    elif args.sample_type == "v2v" and (args.video_path is None and args.prompt is None):
        video_path = (
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/hiker.mp4"
        )
        prompt = "A robot standing on a mountain top. The sun is setting in the background."
    else:
        image_path = args.image_path
        video_path = args.video_path
        prompt = args.prompt

    # Auto-detect whether transformer weights are in a "transformer/" subfolder or
    # directly at transformer_path (e.g. merged_transformer dirs that are already flat).
    _has_subfolder = os.path.isdir(os.path.join(args.transformer_path, "transformer"))
    transformer = HeliosTransformer3DModel.from_pretrained(
        args.transformer_path,
        subfolder="transformer" if _has_subfolder else None,
        torch_dtype=args.weight_dtype,
    )
    if not args.enable_compile:
        transformer = replace_rmsnorm_with_fp32(transformer)
        transformer = replace_all_norms_with_flash_norms(transformer)
        replace_rope_with_flash_rope()
    if args.attention_backend != "auto":
        transformer.set_attention_backend(args.attention_backend)
        print(f"[Attention] Using user-specified backend: {args.attention_backend}")
    else:
        transformer.set_attention_backend(_resolve_attention_backend())
        print(f"[Attention] Using backend: {_resolve_attention_backend()}")

    vae = AutoencoderKLWan.from_pretrained(
        args.base_model_path,
        subfolder="vae",
        torch_dtype=torch.float32,
    )
    scheduler = HeliosScheduler.from_pretrained(
        args.base_model_path,
        subfolder="scheduler",
    )
    pipe = HeliosPipeline.from_pretrained(
        args.base_model_path,
        transformer=transformer,
        vae=vae,
        scheduler=scheduler,
        torch_dtype=args.weight_dtype,
    )

    if args.lora_path is not None:
        import safetensors.torch
        lora_file = os.path.join(args.lora_path, "pytorch_lora_weights.safetensors")
        if os.path.isfile(args.lora_path):
            lora_file = args.lora_path
        full_state = safetensors.torch.load_file(lora_file)

        lora_state = {k: v for k, v in full_state.items() if "lora" in k}
        norm_state = {k: v for k, v in full_state.items() if "lora" not in k}
        print(f"Loaded checkpoint: {len(lora_state)} LoRA keys, {len(norm_state)} norm keys")

        if lora_state:
            from peft import LoraConfig, set_peft_model_state_dict
            lora_rank = lora_state[next(k for k in lora_state if "lora_A" in k)].shape[0]
            lora_config = LoraConfig(
                r=lora_rank, lora_alpha=lora_rank,
                init_lora_weights="gaussian",
                target_modules="all-linear",
                exclude_modules=["down", "up"],
            )
            transformer.add_adapter(lora_config, adapter_name="default")
            transformer_key_prefix = "transformer."
            peft_state = {k[len(transformer_key_prefix):]: v for k, v in lora_state.items()
                          if k.startswith(transformer_key_prefix)}
            incompatible = set_peft_model_state_dict(transformer, peft_state, adapter_name="default")
            if incompatible.missing_keys:
                print(f"  Warning: {len(incompatible.missing_keys)} missing LoRA keys")
            if incompatible.unexpected_keys:
                print(f"  Warning: {len(incompatible.unexpected_keys)} unexpected LoRA keys")

        if norm_state:
            strip_prefix = "transformer.transformer."
            norm_renamed = {}
            for k, v in norm_state.items():
                nk = k[len(strip_prefix):] if k.startswith(strip_prefix) else k
                norm_renamed[nk] = v
            info = transformer.load_state_dict(norm_renamed, strict=False)
            loaded_norms = len(norm_renamed) - len(info.unexpected_keys)
            print(f"  Loaded {loaded_norms} norm weights into transformer")

        if args.partial_path is not None:
            if not hasattr(args, "training_config"):
                from argparse import Namespace

                args.training_config = Namespace()
            args.training_config.is_enable_stage1 = True
            args.training_config.restrict_self_attn = True
            args.training_config.is_amplify_history = True
            args.training_config.is_use_gan = True
            load_extra_components(args, transformer, args.partial_path)

    if args.enable_compile:
        torch.backends.cudnn.benchmark = True
        pipe.text_encoder.compile(mode="max-autotune-no-cudagraphs", dynamic=False)
        pipe.vae.compile(mode="max-autotune-no-cudagraphs", dynamic=False)
        pipe.transformer.compile(mode="max-autotune-no-cudagraphs", dynamic=False)

    if args.enable_low_vram_mode:
        pipe.enable_group_offload(
            onload_device=torch.device("cuda"),
            offload_device=torch.device("cpu"),
            offload_type=args.group_offloading_type,
            num_blocks_per_group=args.num_blocks_per_group if args.group_offloading_type == "block_level" else None,
            use_stream=True,
            record_stream=True,
        )
    else:
        pipe = pipe.to(device)

    if world_size > 1 and args.enable_parallelism:
        if args.cp_backend == "ring":
            cp_config = ContextParallelConfig(ring_degree=world_size)
        elif args.cp_backend == "unified":
            cp_config = ContextParallelConfig(ring_degree=world_size // 2, ulysses_degree=world_size // 2)
        elif args.cp_backend == "ulysses":
            cp_config = ContextParallelConfig(ulysses_degree=world_size)
        elif args.cp_backend == "ulysses_anything":
            cp_config = ContextParallelConfig(ulysses_degree=world_size, ulysses_anything=True)
        else:
            raise ValueError(f"Unsupported cp_backend: {args.cp_backend}")

        pipe.transformer.enable_parallelism(config=cp_config)

    # Load action embedding cache if provided
    action_embeds_cache = None
    if args.action_embeds_cache is not None and os.path.exists(args.action_embeds_cache):
        action_embeds_cache = torch.load(args.action_embeds_cache, map_location="cpu", weights_only=False)
        print(f"Loaded action embedding cache with {len(action_embeds_cache)} entries")

    # ── Prompt → (keys, mouse) parsing (mirrors YUME generate_per_sample_captions) ──
    #
    # Sekai captions use diverse phrasing ("advances steadily forward",
    # "pans slightly to the right", "the drone glides forward",
    # "continues to move forward", "pans from left to right"), so we use
    # multiple complementary patterns to maximise coverage.
    _MOVE_VERBS = r"moves|advances|progresses|glides|proceeds|pushes|follows|tracks|navigates"
    _PAN_VERBS = r"pans|shifts|turns|sweeps|rotates"
    _TILT_VERBS = r"tilts|angles"
    _ALL_VERBS = f"{_MOVE_VERBS}|{_PAN_VERBS}|{_TILT_VERBS}"
    _ADVERBS = r"(?:\s+(?:steadily|smoothly|slightly|gently|gradually|subtly|then|occasionally|slowly|briefly|further|continuously|rapidly|quickly))*"
    _PREPS = r"(?:\s+(?:to the|toward the|towards the))?"
    _DIRS = r"forward|backward|upward|downward|left|right|up|down"
    _SUBJECT = r"the (?:camera|viewer(?:'s (?:perspective|gaze|view))?|drone)(?:\s+angle)?"

    _CLAUSE_PAT = re.compile(
        rf',?\s*{_SUBJECT}{_ADVERBS}\s+'
        rf'({_ALL_VERBS}){_ADVERBS}{_PREPS}\s+'
        rf'({_DIRS})',
        re.IGNORECASE,
    )

    # "continues to move/pan/tilt forward" or "continues moving forward"
    _CONTINUES_PAT = re.compile(
        rf',?\s*{_SUBJECT}{_ADVERBS}\s+'
        rf'continues{_ADVERBS}\s+(?:'
        rf'(?:to\s+)?(?:move|pan|tilt|advance|progress|glide|shift|push|turn|track|navigate)'
        rf'|moving|panning|tilting|advancing|progressing|gliding|shifting|pushing|turning'
        rf'){_ADVERBS}{_PREPS}\s+({_DIRS})',
        re.IGNORECASE,
    )

    # "continues its [steady] forward journey" / "continues its upward climb"
    _CONTINUES_ITS_PAT = re.compile(
        rf',?\s*{_SUBJECT}{_ADVERBS}\s+'
        rf'continues\s+(?:its|the)\s+(?:steady\s+|smooth\s+)?({_DIRS})',
        re.IGNORECASE,
    )

    # "pans smoothly from left to right"
    _FROM_TO_PAT = re.compile(
        rf',?\s*{_SUBJECT}{_ADVERBS}\s+'
        rf'(?:{_ALL_VERBS}){_ADVERBS}\s+'
        rf'from\s+(?:the\s+)?({_DIRS})\s+to\s+(?:the\s+)?({_DIRS})',
        re.IGNORECASE,
    )

    # "the camera/viewer moving forward" (gerund without finite verb)
    _GERUND_PAT = re.compile(
        rf',?\s*{_SUBJECT}{_ADVERBS}\s+'
        rf'(moving|panning|tilting|advancing|progressing|gliding|shifting|pushing|turning)'
        rf'{_ADVERBS}{_PREPS}\s+({_DIRS})',
        re.IGNORECASE,
    )

    _DIR_NORMALIZE = {"upward": "up", "downward": "down"}
    _MOVE_TO_KEY = {"forward": "W", "backward": "S", "left": "A", "right": "D"}
    _PAN_VERBS_SET = {"pans", "shifts", "turns", "sweeps", "rotates",
                      "panning", "shifting", "turning"}
    _TILT_VERBS_SET = {"tilts", "angles", "tilting"}
    _ROT_TO_MOUSE = {
        "left": "←", "right": "→",
        "up": "↑", "down": "↓",
    }

    def _classify_verb(verb):
        """Classify a camera verb into 'move' (translation) or 'rotate' (pan/tilt)."""
        v = verb.lower()
        if v in _PAN_VERBS_SET or v in _TILT_VERBS_SET:
            return "rotate"
        return "move"

    def _verb_direction_to_action(verb, direction):
        """Map (verb, direction) → (keys, mouse)."""
        direction = _DIR_NORMALIZE.get(direction, direction)
        if _classify_verb(verb) == "move":
            return _MOVE_TO_KEY.get(direction, "None"), "·"
        else:
            return "None", _ROT_TO_MOUSE.get(direction, "·")

    def _collect_all_clause_matches(prompt_text):
        """Gather directional clause matches from all patterns, deduplicated.

        Each result carries .start, .end, .verb, .direction, and .text (the
        matched substring from the original prompt).
        """
        Match = type("Match", (), {})

        results = []
        for m in _CLAUSE_PAT.finditer(prompt_text):
            o = Match()
            o.start, o.end = m.start(), m.end()
            o.verb, o.direction = m.group(1).lower(), m.group(2).lower()
            o.text = prompt_text[m.start():m.end()].strip(", ")
            results.append(o)

        def _overlaps(new_start, new_end):
            return any(r.start <= new_start < r.end for r in results)

        for m in _CONTINUES_PAT.finditer(prompt_text):
            if not _overlaps(m.start(), m.end()):
                o = Match()
                o.start, o.end = m.start(), m.end()
                o.verb, o.direction = "moves", m.group(1).lower()
                o.text = prompt_text[m.start():m.end()].strip(", ")
                results.append(o)

        for m in _CONTINUES_ITS_PAT.finditer(prompt_text):
            if not _overlaps(m.start(), m.end()):
                o = Match()
                o.start, o.end = m.start(), m.end()
                o.verb, o.direction = "moves", m.group(1).lower()
                o.text = prompt_text[m.start():m.end()].strip(", ")
                results.append(o)

        for m in _FROM_TO_PAT.finditer(prompt_text):
            if not _overlaps(m.start(), m.end()):
                o = Match()
                o.start, o.end = m.start(), m.end()
                o.verb, o.direction = "pans", m.group(2).lower()
                o.text = prompt_text[m.start():m.end()].strip(", ")
                results.append(o)

        for m in _GERUND_PAT.finditer(prompt_text):
            if not _overlaps(m.start(), m.end()):
                o = Match()
                o.start, o.end = m.start(), m.end()
                gerund = m.group(1).lower()
                o.verb = gerund.rstrip("ing") + "s" if gerund not in _PAN_VERBS_SET and gerund not in _TILT_VERBS_SET else gerund
                o.direction = m.group(2).lower()
                o.text = prompt_text[m.start():m.end()].strip(", ")
                results.append(o)

        results.sort(key=lambda r: r.start)
        return results

    def parse_prompt_to_chunk_actions(prompt_text, num_chunks):
        """Parse camera clauses from prompt into per-chunk (keys, mouse, clause) triples.

        Adjacent move+rotation clauses (no significant text between them) are
        merged into a single combined action, e.g. "the camera moves forward,
        the camera pans left" → ("W", "←", "...") instead of two separate chunks.
        Non-camera text gaps (>=20 chars) become ("None", "·", "") segments.
        Segments are distributed evenly across num_chunks.
        """
        matches = _collect_all_clause_matches(prompt_text)
        if not matches:
            return [("None", "·", "")] * num_chunks

        MIN_ACTION_LEN = 20

        raw_clauses = []
        prev_end = 0
        for m in matches:
            gap = prompt_text[prev_end:m.start].strip(", ")
            raw_clauses.append((gap, m.verb, m.direction, m.text))
            prev_end = m.end
        trailing = prompt_text[prev_end:].strip(", .")

        segments = []
        i = 0
        while i < len(raw_clauses):
            gap, verb, direction, clause_text = raw_clauses[i]

            if len(gap) >= MIN_ACTION_LEN:
                segments.append(("None", "·", ""))

            keys, mouse = _verb_direction_to_action(verb, direction)

            if i + 1 < len(raw_clauses):
                next_gap, next_verb, next_dir, next_clause = raw_clauses[i + 1]
                can_merge = len(next_gap) < MIN_ACTION_LEN
                if can_merge:
                    is_move = _classify_verb(verb) == "move"
                    next_is_move = _classify_verb(next_verb) == "move"
                    if is_move and not next_is_move:
                        _, mouse = _verb_direction_to_action(next_verb, next_dir)
                        merged_text = f"{clause_text} + {next_clause}"
                        i += 2
                        segments.append((keys, mouse, merged_text))
                        continue
                    elif not is_move and next_is_move:
                        keys, _ = _verb_direction_to_action(next_verb, next_dir)
                        merged_text = f"{clause_text} + {next_clause}"
                        i += 2
                        segments.append((keys, mouse, merged_text))
                        continue

            segments.append((keys, mouse, clause_text))
            i += 1

        if len(trailing) >= MIN_ACTION_LEN:
            segments.append(("None", "·", ""))

        if not segments:
            return [("None", "·", "")] * num_chunks

        n = len(segments)
        base = num_chunks // n
        remainder = num_chunks % n
        actions = []
        for i, seg in enumerate(segments):
            steps = base + (1 if i < remainder else 0)
            actions.extend([seg] * steps)
        return actions

    # ── Per-chunk action override from txt file ──
    _PARENS_PAT = re.compile(r"\(([^)]*)\)")

    def parse_action_txt(path: str):
        """Parse a per-chunk action txt file.

        Each line has two parenthesised tokens:
          "... (W). ... (·). ..."  → ("W", "·")
          "... (D). ... (→). ..."  → ("D", "→")
        Returns list of (keys, mouse) tuples, one per line.
        """
        actions = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                tokens = _PARENS_PAT.findall(line)
                if len(tokens) >= 2:
                    actions.append((tokens[0], tokens[1]))
                elif len(tokens) == 1:
                    actions.append((tokens[0], "·"))
                else:
                    actions.append(("None", "·"))
        return actions

    # ── Unified action resolution ──

    def _pad_or_truncate(actions, num_chunks):
        """Pad (repeat last) or truncate an action list to exactly num_chunks."""
        if len(actions) < num_chunks:
            return actions + [actions[-1]] * (num_chunks - len(actions))
        return actions[:num_chunks]

    def _parse_chunk_actions_json(raw):
        """Parse a chunk_actions_json string → list of (keys, mouse) tuples."""
        import json as _json
        parsed = _json.loads(raw)
        return [(str(c[0] if isinstance(c, (list, tuple)) else c.get("keys", "None")),
                 str(c[1] if isinstance(c, (list, tuple)) else c.get("mouse", "·")))
                for c in parsed]

    def resolve_row_actions(row, df, num_chunks, action_txt_actions):
        """Resolve per-chunk (keys, mouse) actions for a single CSV row.

        Returns (actions, source) where:
          - actions: list of exactly num_chunks (keys, mouse) tuples
          - source: str label indicating which priority was used

        Resolution order:
          1. chunk_actions_json column  — GT per-chunk actions exported from PT files
          2. --action_txt_path          — fixed sequence shared across all videos
          3. action_keys/action_mouse   — single action repeated to all chunks
          4. prompt_original column     — regex-parsed from original caption (movement intact)
          5. prompt column              — fallback regex parse
        """
        # 1. chunk_actions_json — ground truth from PT files
        ca_json = row.get("chunk_actions_json", "")
        if isinstance(ca_json, str) and ca_json.strip():
            return _pad_or_truncate(_parse_chunk_actions_json(ca_json), num_chunks), "chunk_actions_json"

        # 2. --action_txt_path — manually specified, shared across all videos
        if action_txt_actions is not None:
            return _pad_or_truncate(list(action_txt_actions), num_chunks), "action_txt_path"

        # 3. CSV action_keys/action_mouse — single action repeated
        if "action_keys" in df.columns and "action_mouse" in df.columns:
            return [(str(row["action_keys"]), str(row["action_mouse"]))] * num_chunks, "csv_action_columns"

        # 4. prompt_original — regex parse from original (movement-containing) caption
        orig = row.get("prompt_original")
        if isinstance(orig, str) and orig.strip():
            return parse_prompt_to_chunk_actions(orig, num_chunks), "prompt_original_regex"

        # 5. prompt — fallback
        p = row.get("refined_prompt") or row.get("prompt", "")
        return parse_prompt_to_chunk_actions(p, num_chunks), "prompt_regex"

    def actions_to_embeds(actions, cache):
        """Convert list of (keys, mouse) tuples → list of action embeddings."""
        if cache is None:
            return None
        default = cache.get(("None", "·"))
        return [cache.get((a[0], a[1]), default) for a in actions]

    # Pre-parse action txt if provided
    action_txt_actions = None
    if args.action_txt_path is not None:
        action_txt_actions = parse_action_txt(args.action_txt_path)
        if rank == 0:
            print(f"Loaded {len(action_txt_actions)} per-chunk actions from {args.action_txt_path}")
            for i, (k, m) in enumerate(action_txt_actions):
                print(f"  chunk {i}: keys={k}, mouse={m}")

    # Per-sample metadata log, saved to inference_meta.json for HTML visualization
    chunk_actions_log = {}
    inference_meta = {"params": {}, "samples": {}}

    if args.prompt_txt_path is not None:
        with open(args.prompt_txt_path, "r") as f:
            prompt_list = [line.strip() for line in f.readlines() if line.strip()]
        if not args.enable_parallelism:
            prompt_list_with_idx = [(i, prompt) for i, prompt in enumerate(prompt_list)]
            prompt_list_with_idx = prompt_list_with_idx[rank::world_size]
        else:
            prompt_list_with_idx = [(i, prompt) for i, prompt in enumerate(prompt_list)]

        for idx, prompt in tqdm(prompt_list_with_idx, desc="Processing prompts"):
            output_path = os.path.join(args.output_folder, f"{idx}.mp4")
            if os.path.exists(output_path):
                print("skipping!")
                continue

            with torch.no_grad():
                try:
                    output = pipe(
                        prompt=prompt,
                        negative_prompt=args.negative_prompt,
                        height=args.height,
                        width=args.width,
                        num_frames=args.num_frames,
                        num_inference_steps=args.num_inference_steps,
                        guidance_scale=args.guidance_scale,
                        generator=torch.Generator(device="cuda").manual_seed(args.seed),
                        # stage 1
                        history_sizes=[16, 2, 1],
                        num_latent_frames_per_chunk=args.num_latent_frames_per_chunk,
                        keep_first_frame=True,
                        # stage 2
                        is_enable_stage2=args.is_enable_stage2,
                        pyramid_num_inference_steps_list=args.pyramid_num_inference_steps_list,
                        # stage 3
                        is_skip_first_chunk=args.is_skip_first_chunk,
                        is_amplify_first_chunk=args.is_amplify_first_chunk,
                        # cfg zero
                        use_zero_init=args.use_zero_init,
                        zero_steps=args.zero_steps,
                        # i2v
                        image=load_image(image_path).resize((args.width, args.height))
                        if image_path is not None
                        else None,
                        image_noise_sigma_min=args.image_noise_sigma_min,
                        image_noise_sigma_max=args.image_noise_sigma_max,
                        # v2v
                        video=load_video(video_path) if video_path is not None else None,
                        video_noise_sigma_min=args.video_noise_sigma_min,
                        video_noise_sigma_max=args.video_noise_sigma_max,
                        # interpolate_prompt
                        use_interpolate_prompt=args.use_interpolate_prompt,
                        interpolation_steps=args.interpolation_steps,
                        interpolate_time_list=interpolate_time_list,
                    ).frames[0]
                except Exception:
                    continue
            if not args.enable_parallelism or rank == 0:
                export_to_video(output, output_path, fps=24)
    elif args.image_prompt_csv_path is not None:
        df = read_prompt_table(args.image_prompt_csv_path)
        if args.max_samples > 0:
            df = df.head(args.max_samples)

        num_chunks = max(1, args.num_frames // 33)
        for _, r in df.iterrows():
            ra, src = resolve_row_actions(r, df, num_chunks, action_txt_actions)
            sid = str(r["id"])
            chunk_actions_log[sid] = [list(a) for a in ra]
            actual_prompt = str(r.get("refined_prompt") or r["prompt"])
            inference_meta["samples"][sid] = {
                "prompt": actual_prompt,
                "prompt_original": str(r["prompt_original"]) if "prompt_original" in df.columns and pd.notna(r.get("prompt_original")) else "",
                "action_source": src,
                "chunk_actions": [list(a) for a in ra],
            }

        if not args.enable_parallelism:
            df = df.iloc[rank::world_size]

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing prompts"):
            output_path = os.path.join(args.output_folder, f"{row['id']}.mp4")
            if os.path.exists(output_path):
                print("skipping!")
                continue

            prompt = row.get("refined_prompt") or row["prompt"]
            image_path = (
                os.path.join(args.base_image_prompt_path, str(row["image_name"]))
                if args.base_image_prompt_path and row["image_name"]
                else None
            )

            row_actions, _ = resolve_row_actions(row, df, num_chunks, action_txt_actions)
            row_action_embeds = actions_to_embeds(row_actions, action_embeds_cache)

            if rank == 0:
                print(f"  [{row['id']}] chunks={num_chunks}, actions={chunk_actions_log.get(str(row['id']))}")

            with torch.no_grad():
                try:
                    output = pipe(
                        prompt=prompt,
                        negative_prompt=args.negative_prompt,
                        height=args.height,
                        width=args.width,
                        num_frames=args.num_frames,
                        num_inference_steps=args.num_inference_steps,
                        guidance_scale=args.guidance_scale,
                        generator=torch.Generator(device="cuda").manual_seed(args.seed),
                        # stage 1
                        history_sizes=[16, 2, 1],
                        num_latent_frames_per_chunk=args.num_latent_frames_per_chunk,
                        keep_first_frame=True,
                        # stage 2
                        is_enable_stage2=args.is_enable_stage2,
                        pyramid_num_inference_steps_list=args.pyramid_num_inference_steps_list,
                        # stage 3
                        is_skip_first_chunk=args.is_skip_first_chunk,
                        is_amplify_first_chunk=args.is_amplify_first_chunk,
                        # cfg zero
                        use_zero_init=args.use_zero_init,
                        zero_steps=args.zero_steps,
                        # i2v
                        image=load_image(image_path).resize((args.width, args.height))
                        if image_path is not None
                        else None,
                        image_noise_sigma_min=args.image_noise_sigma_min,
                        image_noise_sigma_max=args.image_noise_sigma_max,
                        # v2v
                        video=load_video(video_path) if video_path is not None else None,
                        video_noise_sigma_min=args.video_noise_sigma_min,
                        video_noise_sigma_max=args.video_noise_sigma_max,
                        # interpolate_prompt
                        use_interpolate_prompt=args.use_interpolate_prompt,
                        interpolation_steps=args.interpolation_steps,
                        interpolate_time_list=interpolate_time_list,
                        # action conditioning
                        action_embeds_list=row_action_embeds,
                    ).frames[0]
                except Exception as e:
                    print(f"Error processing {row['id']}: {e}")
                    continue
            if not args.enable_parallelism or rank == 0:
                export_to_video(output, output_path, fps=24)
    elif args.interactive_prompt_csv_path is not None:
        df = read_prompt_table(args.interactive_prompt_csv_path)

        df = df.sort_values(by=["id", "prompt_index"])
        all_video_ids = df["id"].unique()
        if args.max_samples > 0:
            all_video_ids = all_video_ids[:args.max_samples]

        num_chunks = max(1, args.num_frames // 33)
        for vid in all_video_ids:
            group = df[df["id"] == vid]
            first_row = group.iloc[0]
            ra, src = resolve_row_actions(first_row, df, num_chunks, action_txt_actions)
            sid = str(vid)
            chunk_actions_log[sid] = [list(a) for a in ra]
            if "refined_prompt" in df.columns:
                prompts = group["refined_prompt"].fillna(group["prompt"]).tolist()
            else:
                prompts = group["prompt"].tolist()
            inference_meta["samples"][sid] = {
                "prompt": prompts[0] if len(prompts) == 1 else prompts,
                "prompt_original": str(first_row["prompt_original"]) if "prompt_original" in df.columns and pd.notna(first_row.get("prompt_original")) else "",
                "action_source": src,
                "chunk_actions": [list(a) for a in ra],
            }

        if not args.enable_parallelism:
            my_video_ids = all_video_ids[rank::world_size]
        else:
            my_video_ids = all_video_ids

        for video_id in tqdm(my_video_ids, desc="Processing prompts"):
            output_path = os.path.join(args.output_folder, f"{video_id}.mp4")

            if os.path.exists(output_path):
                print(f"skipping {output_path}!")
                continue

            group_df = df[df["id"] == video_id]

            if "refined_prompt" in df.columns:
                prompt_list = group_df["refined_prompt"].fillna(group_df["prompt"]).tolist()
            else:
                prompt_list = group_df["prompt"].tolist()
            interpolate_time_list = [args.interpolate_time] * len(prompt_list)

            row_actions, _ = resolve_row_actions(group_df.iloc[0], df, num_chunks, action_txt_actions)
            csv_action_embeds = actions_to_embeds(row_actions, action_embeds_cache)

            with torch.no_grad():
                try:
                    output = pipe(
                        prompt=prompt_list,
                        negative_prompt=args.negative_prompt,
                        height=args.height,
                        width=args.width,
                        num_frames=args.num_frames,
                        num_inference_steps=args.num_inference_steps,
                        guidance_scale=args.guidance_scale,
                        generator=torch.Generator(device="cuda").manual_seed(args.seed),
                        # stage 1
                        history_sizes=[16, 2, 1],
                        num_latent_frames_per_chunk=args.num_latent_frames_per_chunk,
                        keep_first_frame=True,
                        # stage 2
                        is_enable_stage2=args.is_enable_stage2,
                        pyramid_num_inference_steps_list=args.pyramid_num_inference_steps_list,
                        # stage 3
                        is_skip_first_chunk=args.is_skip_first_chunk,
                        is_amplify_first_chunk=args.is_amplify_first_chunk,
                        # cfg zero
                        use_zero_init=args.use_zero_init,
                        zero_steps=args.zero_steps,
                        # i2v
                        image=load_image(image_path).resize((args.width, args.height))
                        if image_path is not None
                        else None,
                        image_noise_sigma_min=args.image_noise_sigma_min,
                        image_noise_sigma_max=args.image_noise_sigma_max,
                        # v2v
                        video=load_video(video_path) if video_path is not None else None,
                        video_noise_sigma_min=args.video_noise_sigma_min,
                        video_noise_sigma_max=args.video_noise_sigma_max,
                        # interpolate_prompt
                        use_interpolate_prompt=args.use_interpolate_prompt,
                        interpolation_steps=args.interpolation_steps,
                        interpolate_time_list=interpolate_time_list,
                        # action conditioning
                        action_embeds_list=csv_action_embeds,
                    ).frames[0]
                except Exception:
                    continue
            if not args.enable_parallelism or rank == 0:
                export_to_video(output, output_path, fps=24)
    else:
        num_chunks = max(1, args.num_frames // 33)
        if action_txt_actions is not None:
            single_actions = _pad_or_truncate(list(action_txt_actions), num_chunks)
            _single_src = "action_txt_path"
        elif args.action_keys is not None and args.action_mouse is not None:
            single_actions = [(args.action_keys, args.action_mouse)] * num_chunks
            _single_src = "cli_action_keys"
        else:
            single_actions = parse_prompt_to_chunk_actions(prompt, num_chunks)
            _single_src = "prompt_regex"
        single_action_embeds = actions_to_embeds(single_actions, action_embeds_cache)
        inference_meta["samples"]["single"] = {
            "prompt": prompt,
            "prompt_original": "",
            "action_source": _single_src,
            "chunk_actions": [list(a) for a in single_actions],
        }

        with torch.no_grad():
            output = pipe(
                prompt=prompt,
                negative_prompt=args.negative_prompt,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=torch.Generator(device="cuda").manual_seed(args.seed),
                # stage 1
                history_sizes=[16, 2, 1],
                num_latent_frames_per_chunk=args.num_latent_frames_per_chunk,
                keep_first_frame=True,
                # stage 2
                is_enable_stage2=args.is_enable_stage2,
                pyramid_num_inference_steps_list=args.pyramid_num_inference_steps_list,
                # stage 3
                is_skip_first_chunk=args.is_skip_first_chunk,
                is_amplify_first_chunk=args.is_amplify_first_chunk,
                # cfg zero
                use_zero_init=args.use_zero_init,
                zero_steps=args.zero_steps,
                # i2v
                image=load_image(image_path).resize((args.width, args.height)) if image_path is not None else None,
                image_noise_sigma_min=args.image_noise_sigma_min,
                image_noise_sigma_max=args.image_noise_sigma_max,
                # v2v
                video=load_video(video_path) if video_path is not None else None,
                video_noise_sigma_min=args.video_noise_sigma_min,
                video_noise_sigma_max=args.video_noise_sigma_max,
                # interpolate_prompt
                use_interpolate_prompt=args.use_interpolate_prompt,
                interpolation_steps=args.interpolation_steps,
                interpolate_time_list=interpolate_time_list,
                # action conditioning
                action_embeds_list=single_action_embeds,
            ).frames[0]
            # elapsed_time = time.time() - start_time
            # print(f"Inference time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")

        if not args.enable_parallelism or rank == 0:
            file_count = len(
                [f for f in os.listdir(args.output_folder) if os.path.isfile(os.path.join(args.output_folder, f))]
            )
            output_path = os.path.join(
                args.output_folder, f"{file_count:04d}_{args.sample_type}_{int(time.time())}.mp4"
            )
            export_to_video(output, output_path, fps=24)

    print(f"Max memory: {torch.cuda.max_memory_allocated() / 1024**3:.3f} GB")

    # ── Save chunk actions log & inference metadata (rank 0 only) ──
    chunk_actions_path = None
    if rank == 0 and chunk_actions_log:
        chunk_actions_path = os.path.join(args.output_folder, "chunk_actions.json")
        with open(chunk_actions_path, "w") as f:
            json.dump(chunk_actions_log, f, ensure_ascii=False, indent=1)
        print(f"Saved chunk actions for {len(chunk_actions_log)} samples → {chunk_actions_path}")

    if rank == 0 and inference_meta["samples"]:
        inference_meta["params"] = {
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "seed": args.seed,
            "negative_prompt": args.negative_prompt or "",
            "num_latent_frames_per_chunk": args.num_latent_frames_per_chunk,
            "is_enable_stage2": args.is_enable_stage2,
            "pyramid_num_inference_steps_list": args.pyramid_num_inference_steps_list,
            "use_zero_init": args.use_zero_init,
            "zero_steps": args.zero_steps,
            "image_noise_sigma_min": args.image_noise_sigma_min,
            "image_noise_sigma_max": args.image_noise_sigma_max,
            "use_interpolate_prompt": args.use_interpolate_prompt,
        }
        meta_path = os.path.join(args.output_folder, "inference_meta.json")
        with open(meta_path, "w") as f:
            json.dump(inference_meta, f, ensure_ascii=False, indent=1)
        print(f"Saved inference metadata → {meta_path}")

    if rank == 0 and args.image_prompt_csv_path is not None:
        from visualize_results import build_eval_html

        html_dir = os.path.join(args.output_folder, "html")
        html_path = os.path.join(html_dir, "results.html")
        print(f"Generating HTML visualization → {html_path}")
        build_eval_html(
            video_dir=args.output_folder,
            label_path=args.image_prompt_csv_path,
            first_frame_dir=args.base_image_prompt_path,
            output_path=html_path,
            chunk_actions_path=chunk_actions_path,
        )
    elif rank == 0 and args.interactive_prompt_csv_path is not None:
        from visualize_results import build_eval_html

        html_dir = os.path.join(args.output_folder, "html")
        html_path = os.path.join(html_dir, "results.html")
        print(f"Generating HTML visualization → {html_path}")
        build_eval_html(
            video_dir=args.output_folder,
            label_path=args.interactive_prompt_csv_path,
            first_frame_dir=None,
            output_path=html_path,
            chunk_actions_path=chunk_actions_path,
        )


if __name__ == "__main__":
    main()
