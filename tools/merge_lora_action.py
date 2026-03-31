"""
Merge LoRA + norm + partial weights into transformer base and save a standalone checkpoint.

The training code saves both LoRA weights and norm layer weights (norm_q, norm_k, etc.)
into the same pytorch_lora_weights.safetensors.  diffusers' load_lora_weights() rejects
files containing non-LoRA keys, so we split them manually before loading.

Flow:
  1. Load base transformer
  2. Build pipeline
  3. Load safetensors → split into LoRA keys + norm keys
  4. Load LoRA via pipe (temp file with LoRA-only keys)
  5. Fuse LoRA into base → unload adapter
  6. Apply norm weights directly into transformer
  7. Load transformer_partial.pth (multi_term_memory_patch etc.)
  8. Save full merged transformer

Usage (first merge — base has subfolder "transformer"):
    python tools/merge_lora_action.py \
        --base_transformer  BestWishYSH/Helios-Base \
        --subfolder         transformer \
        --pipeline           BestWishYSH/Helios-Base \
        --checkpoint         ckpts/.../checkpoint-5000 \
        --output             ckpts/.../merged_transformer

Usage (subsequent merge — merged dir is flat, no subfolder):
    python tools/merge_lora_action.py \
        --base_transformer  ckpts/.../merged_transformer \
        --pipeline           BestWishYSH/Helios-Base \
        --checkpoint         ckpts/.../checkpoint-5000 \
        --output             ckpts/.../merged_transformer_v2
"""
import argparse, os, sys, tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from argparse import Namespace
from safetensors.torch import load_file, save_file
from helios.modules.transformer_helios import HeliosTransformer3DModel
from helios.pipelines.pipeline_helios import HeliosPipeline
from helios.utils.utils_base import load_extra_components, NORM_LAYER_PREFIXES


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_transformer", required=True,
                    help="Dir containing the base transformer weights")
    p.add_argument("--subfolder", default=None,
                    help="Subfolder inside base_transformer (e.g. 'transformer' for HF models). "
                         "Omit for flat merged dirs.")
    p.add_argument("--pipeline", required=True,
                    help="Pipeline model path (provides VAE/tokenizer/scheduler)")
    p.add_argument("--checkpoint", required=True,
                    help="Training checkpoint dir (contains pytorch_lora_weights.safetensors)")
    p.add_argument("--output", required=True, help="Where to save merged transformer")
    p.add_argument("--use_ema", action="store_true",
                    help="Look for weights in checkpoint/model_ema/ instead")
    args = p.parse_args()

    ckpt = os.path.join(args.checkpoint, "model_ema") if args.use_ema and \
           os.path.isdir(os.path.join(args.checkpoint, "model_ema")) else args.checkpoint

    lora_path = os.path.join(ckpt, "pytorch_lora_weights.safetensors")
    partial_path = os.path.join(ckpt, "transformer_partial.pth")
    if not os.path.exists(lora_path):
        raise FileNotFoundError(f"LoRA weights not found: {lora_path}")

    # ── 0. Split safetensors into LoRA keys vs norm keys ──
    all_weights = load_file(lora_path)
    lora_weights = {}
    norm_weights = {}
    for k, v in all_weights.items():
        if any(prefix in k for prefix in NORM_LAYER_PREFIXES):
            norm_weights[k] = v
        else:
            lora_weights[k] = v
    print(f"[0/6] Loaded {lora_path}: {len(lora_weights)} LoRA keys, {len(norm_weights)} norm keys")

    # ── 1. Load base transformer ──
    kwargs = {
        "has_multi_term_memory_patch": True,
        "zero_history_timestep": True,
        "guidance_cross_attn": True,
        "restrict_self_attn": False,
        "is_train_restrict_lora": False,
        "restrict_lora": False,
        "restrict_lora_rank": 128,
    }
    load_args = dict(transformer_additional_kwargs=kwargs)
    if args.subfolder:
        load_args["subfolder"] = args.subfolder

    print(f"[1/6] Loading transformer from {args.base_transformer}"
          f" (subfolder={args.subfolder or 'none'})")
    transformer = HeliosTransformer3DModel.from_pretrained(
        args.base_transformer, **load_args)

    # ── 2. Build pipeline ──
    print(f"[2/6] Building pipeline from {args.pipeline}")
    pipe = HeliosPipeline.from_pretrained(args.pipeline, transformer=transformer)

    # ── 3. Load LoRA (via temp file containing only LoRA keys) ──
    print(f"[3/6] Loading {len(lora_weights)} LoRA weights")
    with tempfile.TemporaryDirectory() as td:
        tmp_lora = os.path.join(td, "pytorch_lora_weights.safetensors")
        save_file(lora_weights, tmp_lora)
        pipe.load_lora_weights(td, adapter_name="default")
    pipe.set_adapters(["default"], adapter_weights=[1.0])

    # ── 4. Load partial BEFORE fuse (matches original merge script order) ──
    if os.path.exists(partial_path):
        print(f"[4/6] Loading partial from {partial_path}")
        mock = Namespace(training_config=Namespace(
            is_enable_stage1=True,
            is_train_full_multi_term_memory_patchg=True,
            is_train_lora_multi_term_memory_patchg=True,
            restrict_self_attn=False,
            is_train_restrict_lora=False,
            is_amplify_history=False,
            is_use_gan=False,
        ))
        load_extra_components(mock, transformer, partial_path)
    else:
        print(f"[4/6] No partial checkpoint, skipping")

    # ── 5. Fuse LoRA into base weights and remove adapter ──
    print(f"[5/6] Fusing LoRA into base weights")
    pipe.fuse_lora()
    pipe.unload_lora_weights()

    # ── 5b. Apply norm weights (after unload so parameter names are clean) ──
    if norm_weights:
        print(f"[5/6] Applying {len(norm_weights)} norm layer weights")
        strip_prefix = "transformer."
        clean_norm = {}
        for k, v in norm_weights.items():
            clean_k = k
            while clean_k.startswith(strip_prefix):
                clean_k = clean_k[len(strip_prefix):]
            clean_norm[clean_k] = v
        missing, unexpected = pipe.transformer.load_state_dict(clean_norm, strict=False)
        if unexpected:
            print(f"  WARNING: unexpected norm keys: {unexpected[:5]}")
    else:
        print(f"[5/6] No norm weights, skipping")

    # ── 6. Save merged transformer ──
    print(f"[6/6] Saving to {args.output}")
    os.makedirs(args.output, exist_ok=True)
    pipe.transformer.save_pretrained(args.output)
    print("Done!")


if __name__ == "__main__":
    main()
