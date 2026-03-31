#!/bin/bash
set -euo pipefail
cd /mnt/bn/voyager-sg-l3/zhexiao.xiong/Helios

# ╔══════════════════════════════════════════════════════════════╗
# ║  Helios Action World Model — 6-Stage Training Pipeline      ║
# ║                                                              ║
# ║  Paper flow (GitHub issue #43 confirmed):                    ║
# ║    S1-init → merge → S1-post → merge  = Our-Helios-Base     ║
# ║    S2-init → merge → S2-post → merge  = Our-Helios-Mid      ║
# ║    S3-ode  → merge → S3-post → merge  = Our-Helios-Distilled║
# ║                                                              ║
# ║  Each "merge" = fuse_lora() into base, save full transformer ║
# ╚══════════════════════════════════════════════════════════════╝

# ============================================================
#  Environment
# ============================================================
MINICONDA="/mnt/bn/voyager-sg-l3/zhexiao.xiong/miniconda3"
PY="$MINICONDA/bin/python3.13"
export PATH="$MINICONDA/bin:$PATH"
export PYTHONNOUSERSITE=1
hash -r

PY_VER="$("$PY" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")"
export LD_LIBRARY_PATH="$MINICONDA/lib/python${PY_VER}/site-packages/nvidia/cusparselt/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/mnt/bn/voyager-sg-l3/zhexiao.xiong/Helios:${PYTHONPATH:-}"
export WANDB_API_KEY="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"
export WANDB_BASE_URL="https://api.wandb.ai"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ============================================================
#  User config — EDIT THESE
# ============================================================
MODELS="/mnt/bn/voyager-sg-l3/zhexiao.xiong/Helios/BestWishYSH"
CKPTS="/mnt/bn/voyager-sg-l3/zhexiao.xiong/Helios/ckpts/helios"
OUTPUTS="/mnt/bn/voyager-sg-l3/zhexiao.xiong/Helios/output_helios"
DATA_ROOTS=(
    # "data/helios/seadance2_v3_helios_latents"
    "data/helios/yume_training_helios_latents"
)
CONFIGS="scripts/training/configs"

# ============================================================
#  Resume config — set these when resuming after interruption
# ============================================================
#   START_STAGE: which stage to begin from (default=1, run everything)
#     1 = S1-init    2 = S1-post    3 = S2-init
#     4 = S2-post    5 = S3-ode     6 = S3-post
START_STAGE=${START_STAGE:-1}

#   RESUME_EXP: if interrupted MID-STAGE (during training), set this to the
#   existing ckpts/helios/stageX_xxx_YYYYMMDD_HHMMSS dir so that
#   resume_from_checkpoint="latest" picks up where it left off.
#   Leave empty to create a fresh experiment dir for the starting stage.
RESUME_EXP="${RESUME_EXP:-}"

#   Merged transformer paths from COMPLETED stages.
#   Fill in the ones before your START_STAGE.
#   Example: resuming from S1-post needs S1I_MERGED.
#   Example: resuming from S2-init needs OUR_BASE.
S1I_MERGED="${S1I_MERGED:-}"
OUR_BASE="${OUR_BASE:-}"
S2I_MERGED="${S2I_MERGED:-}"
OUR_MID="${OUR_MID:-}"
OUR_ODE="${OUR_ODE:-}"

#   ODE_DATA_DIR: path to pre-generated ODE pair data.
#   If empty, ODE data is auto-generated after S2-post using OUR_MID as
#   the teacher model.  Set this to skip regeneration when resuming.
ODE_DATA_DIR="${ODE_DATA_DIR:-}"
#   ODE_MAX_PROMPTS: max number of prompts for ODE data gen (default 1000).
#   Original repo uses ~1000 (VidProM).  More = better diversity but slower.
ODE_MAX_PROMPTS="${ODE_MAX_PROMPTS:-100}"

#   ODE_SAMPLE_TYPE: "t2v" (text-only, original behavior) or "i2v" (text+image
#   from training data .pt files). Use "i2v" when your downstream task is I2V,
#   so that the ODE teacher trajectories match the I2V conditioning format.
ODE_SAMPLE_TYPE="${ODE_SAMPLE_TYPE:-i2v}"

# ============================================================
#  Validate resume config — fail early with clear errors
# ============================================================
require_var() {
    local name="$1" val="$2" needed_from="$3"
    if [ -z "$val" ]; then
        echo "ERROR: $name is required when START_STAGE=$START_STAGE ($needed_from)"
        echo "  Set it via: $name=/path/to/merged_transformer bash run.sh"
        exit 1
    fi
    if [ ! -d "$val" ]; then
        echo "ERROR: $name=$val does not exist"
        exit 1
    fi
}

if [ "$START_STAGE" -eq 2 ]; then require_var "S1I_MERGED" "$S1I_MERGED" "needed by S1-post"; fi
if [ "$START_STAGE" -ge 3 ]; then require_var "OUR_BASE"   "$OUR_BASE"   "needed by S2+ and S3-post teacher"; fi
if [ "$START_STAGE" -eq 4 ]; then require_var "S2I_MERGED" "$S2I_MERGED" "needed by S2-post"; fi
if [ "$START_STAGE" -ge 5 ]; then require_var "OUR_MID"    "$OUR_MID"    "needed by S3-ode"; fi
if [ "$START_STAGE" -ge 6 ]; then require_var "OUR_ODE"    "$OUR_ODE"    "needed by S3-post"; fi

echo ""
echo ">>> START_STAGE=$START_STAGE  RESUME_EXP=${RESUME_EXP:-<new>}"
echo ""

# ============================================================
#  Helper functions
# ============================================================

# Find latest checkpoint dir (prefers -final over same step number)
latest_ckpt() {
    local dir="$1"
    local best=""
    best=$(ls -d "$dir"/checkpoint-*-final 2>/dev/null | sort -t- -k2 -n | tail -1 || true)
    if [ -z "$best" ]; then
        best=$(ls -d "$dir"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 || true)
    fi
    if [ -z "$best" ]; then
        echo "ERROR: no checkpoint-* found in $dir" >&2
        exit 1
    fi
    echo "$best"
}

# Generate a training config from template yaml.
# Auto-detects flat merged dirs and sets subfolder accordingly.
#   $1 = template yaml   $2 = output yaml
#   $3 = exp_dir          $4 = val_dir
#   $5 = transformer_path (optional, overrides yaml)
#   $6 = pipeline_path    (optional, default=Helios-Base)
#   $7 = extra python to exec on cfg dict (optional)
make_config() {
    local tpl="$1" out="$2" exp="$3" val="$4"
    local tf="${5:-}" pipe="${6:-$MODELS/Helios-Base}" extra="${7:-}"

    "$PY" -c "
import yaml, sys, os

def is_flat_local_dir(p):
    \"\"\"True if p is a local dir without a 'transformer/' subfolder (i.e. merged).\"\"\"
    return os.path.isdir(p) and not os.path.isdir(os.path.join(p, 'transformer'))

def validate_local_path(p, name):
    if not p:
        return
    if '...' in p:
        raise SystemExit(f'ERROR: placeholder in {name}: {p}')
    if (p.startswith('/') or p.startswith('./') or p.startswith('../')) and not os.path.isdir(p):
        raise SystemExit(f'ERROR: {name} path does not exist: {p}')

with open('$tpl') as f:
    c = yaml.safe_load(f)

c['output_dir'] = '$exp'
c['logging_dir'] = '$exp/logs'
c.setdefault('validation_config',{})['val_output_dir'] = '$val'
c.setdefault('data_config',{})['instance_data_root'] = [$(printf "'%s'," "${DATA_ROOTS[@]}")]
c.setdefault('model_config',{})['pretrained_model_name_or_path'] = '$pipe'

tf = '$tf'
if tf:
    validate_local_path(tf, 'transformer_model_name_or_path')
    c['model_config']['transformer_model_name_or_path'] = tf
    if is_flat_local_dir(tf):
        c['model_config']['subfolder'] = ''

# Apply extra config (may set real_score_model_name_or_path etc.)
$extra

# Auto-detect critic/teacher subfolder for stage 3
rsm = c.get('model_config', {}).get('real_score_model_name_or_path', '')
if rsm:
    validate_local_path(rsm, 'real_score_model_name_or_path')
    if is_flat_local_dir(rsm):
        c['model_config']['critic_subfolder'] = ''

with open('$out', 'w') as f:
    yaml.dump(c, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
print(f'Config saved: $out')
"
}

# Merge LoRA into base transformer, then verify output
#   $1 = base_transformer  $2 = subfolder (or "" for flat)
#   $3 = checkpoint_dir    $4 = output_dir
do_merge() {
    local base="$1" sub="$2" ckpt="$3" out="$4"
    echo ""
    echo ">>> MERGE: $ckpt → $out"

    if [ -d "$out" ]; then
        echo ">>> Removing stale merge dir: $out"
        rm -rf "$out"
    fi

    local sub_arg=""
    [ -n "$sub" ] && sub_arg="--subfolder $sub"
    "$PY" -u tools/merge_lora_action.py \
        --base_transformer "$base" $sub_arg \
        --pipeline "$MODELS/Helios-Base" \
        --checkpoint "$ckpt" \
        --output "$out"

    if [ ! -f "$out/config.json" ]; then
        echo "ERROR: merge failed — $out/config.json not found"
        exit 1
    fi

    local n_shards
    n_shards=$(ls "$out"/diffusion_pytorch_model*.safetensors 2>/dev/null | wc -l || true)
    if [ "$n_shards" -eq 0 ]; then
        echo "ERROR: merge produced no safetensors files in $out"
        exit 1
    fi

    local expected_shards="$n_shards"
    local idx_file="$out/diffusion_pytorch_model.safetensors.index.json"
    if [ -f "$idx_file" ]; then
        expected_shards=$("$PY" -c "import json; d=json.load(open('$idx_file')); print(len(set(d.get('weight_map',{}).values())))")
        if [ "$n_shards" -ne "$expected_shards" ]; then
            echo "ERROR: merge incomplete — expected $expected_shards shards but found $n_shards"
            exit 1
        fi
    fi

    sync 2>/dev/null || true
    echo ">>> MERGE OK: $n_shards safetensors files in $out"
}

# Run one training stage
#   $1 = stage name (for logging)  $2 = config yaml  $3 = port
#   $4 = accelerate config yaml (optional; omit for DDP, set for DeepSpeed)
do_train() {
    local stage_name="$1" cfg="$2" port="$3" accel_cfg="${4:-}"
    echo ""
    echo "════════════════════════════════════════════"
    echo "  TRAINING:  $stage_name"
    echo "  Config:    $cfg"
    echo "  Accel cfg: ${accel_cfg:-<default DDP>}"
    echo "════════════════════════════════════════════"
    if [ -n "$accel_cfg" ]; then
        accelerate launch --config_file "$accel_cfg" --main_process_port "$port" \
            train_helios.py --config "$cfg"
    else
        accelerate launch --main_process_port "$port" \
            train_helios.py --config "$cfg"
    fi
}

# Extract prompts from training data .pt files into a .txt file.
#   $1 = output .txt path
#   Reads from DATA_ROOTS array.  Deduplicates and samples up to
#   ODE_MAX_PROMPTS (default 1000) to keep ODE data gen tractable.
extract_prompts() {
    local out="$1"
    echo ">>> Extracting prompts from training data → $out  (max=${ODE_MAX_PROMPTS})"
    "$PY" -u -c "
import os, glob, torch, random, time

max_prompts = int('${ODE_MAX_PROMPTS}')
# Scan at most max_prompts*5 files (shuffle first for diversity)
max_files_to_scan = max_prompts * 10
data_roots = [$(printf "'%s'," "${DATA_ROOTS[@]}")]
seen = set()
prompts = []
t_start = time.time()
for root in data_roots:
    files = sorted(glob.glob(os.path.join(root, '*.pt')))
    random.seed(42)
    random.shuffle(files)
    files = files[:max_files_to_scan]
    print(f'Scanning {root}: {len(files)} files (shuffled, capped at {max_files_to_scan})', flush=True)
    for i, pt_path in enumerate(files):
        try:
            d = torch.load(pt_path, map_location='cpu', weights_only=False)
            p = d.get('prompt_raw', '')
            if isinstance(p, str) and p.strip() and p.strip() not in seen:
                seen.add(p.strip())
                prompts.append(p.strip())
        except Exception as e:
            print(f'  skip {os.path.basename(pt_path)}: {e}', flush=True)
        if (i + 1) % 500 == 0:
            elapsed = time.time() - t_start
            print(f'  ... {i+1}/{len(files)} scanned, {len(prompts)} unique prompts, {elapsed:.0f}s elapsed', flush=True)
        if len(prompts) >= max_prompts:
            print(f'  Reached {max_prompts} unique prompts at file {i+1}, stopping early.', flush=True)
            break
print(f'Total unique prompts: {len(prompts)}', flush=True)
os.makedirs(os.path.dirname('$out') or '.', exist_ok=True)
with open('$out', 'w') as f:
    for p in prompts:
        f.write(p + '\n')
print(f'Saved {len(prompts)} prompts to: $out  (elapsed {time.time()-t_start:.0f}s)', flush=True)
"
    if [ ! -f "$out" ]; then
        echo "ERROR: failed to extract prompts to $out"
        exit 1
    fi
}

# Generate ODE pair data using a teacher model.
#   $1 = teacher transformer path (e.g. OUR_MID, a flat merged dir)
#   $2 = prompt .txt file  (T2V mode) OR ignored (I2V mode)
#   $3 = output dir for ODE pairs
#   $4 = master port
gen_ode_data() {
    local teacher="$1" prompt_txt="$2" ode_out="$3" port="${4:-9503}"
    echo ""
    echo "════════════════════════════════════════════"
    echo "  ODE DATA GENERATION  (mode=${ODE_SAMPLE_TYPE:-t2v})"
    echo "  Teacher:   $teacher"
    echo "  Prompts:   $prompt_txt"
    echo "  Output:    $ode_out"
    echo "════════════════════════════════════════════"

    local sub_args=()
    if [ -d "$teacher" ] && [ ! -d "$teacher/transformer" ]; then
        sub_args=(--subfolder "")
    fi

    local mode_args=()
    if [ "${ODE_SAMPLE_TYPE:-t2v}" = "i2v" ]; then
        mode_args=(
            --sample_type i2v
            --pt_data_dirs $(printf "%s " "${DATA_ROOTS[@]}")
            --i2v_max_samples "${ODE_MAX_PROMPTS}"
        )
    else
        mode_args=(
            --sample_type t2v
            --prompt_txt_files "$prompt_txt"
        )
    fi

    torchrun --nproc_per_node 8 --master_port "$port" \
        tools/offload_data/get_ode-pairs.py \
        --transformer_path "$teacher" \
        --base_model_path "$MODELS/Helios-Base" \
        "${sub_args[@]}" \
        "${mode_args[@]}" \
        --output_dirs "$ode_out" \
        --use_dynamic_shifting \
        --time_shift_type "linear" \
        --use_default_loader \
        --is_enable_stage2 \
        --num_frames 165

    if [ -z "$(ls -A "$ode_out" 2>/dev/null)" ]; then
        echo "ERROR: ODE data dir is empty: $ode_out"
        exit 1
    fi
    echo ">>> ODE data generated: $(ls "$ode_out"/*.pt 2>/dev/null | wc -l || true) files"
}

# Create or reuse exp dir for a stage
#   $1 = stage name prefix (e.g. "stage1_init")
#   $2 = stage number (1-6)
#   Sets: _EXP, _VAL, _CFG
setup_exp_dir() {
    local prefix="$1" stage_num="$2"
    if [ "$stage_num" -eq "$START_STAGE" ] && [ -n "$RESUME_EXP" ]; then
        _EXP="$RESUME_EXP"
        _VAL="$OUTPUTS/$(basename "$RESUME_EXP")"
    else
        local tag="${prefix}_$(date +%Y%m%d_%H%M%S)"
        _EXP="$CKPTS/$tag"
        _VAL="$OUTPUTS/$tag"
    fi
    mkdir -p "$_EXP"
    _CFG="$_EXP/train_config.yaml"
}


# ############################################################
#  STAGE 1-INIT : Learn AR generation (LR=5e-5, rank=128)
# ############################################################
if [ "$START_STAGE" -le 1 ]; then
    setup_exp_dir "stage1_init" 1

    make_config "$CONFIGS/stage_1_action_init.yaml" "$_CFG" \
        "$_EXP" "$_VAL" "$MODELS/Helios-Base"
    do_train "Stage 1-init" "$_CFG" 9500

    S1I_CKPT=$(latest_ckpt "$_EXP")
    S1I_MERGED="$_EXP/merged_transformer"
    do_merge "$MODELS/Helios-Base" "transformer" "$S1I_CKPT" "$S1I_MERGED"

    RESUME_EXP=""
fi
echo ">>> S1I_MERGED = $S1I_MERGED"


# ############################################################
#  STAGE 1-POST : Refine AR (LR=3e-5, rank=128)
#   base = merged S1-init (flat dir, no subfolder)
# ############################################################
if [ "$START_STAGE" -le 2 ]; then
    setup_exp_dir "stage1_post" 2

    make_config "$CONFIGS/stage_1_action_post.yaml" "$_CFG" \
        "$_EXP" "$_VAL" "$S1I_MERGED"
    do_train "Stage 1-post" "$_CFG" 9500

    S1P_CKPT=$(latest_ckpt "$_EXP")
    S1P_MERGED="$_EXP/merged_transformer"
    do_merge "$S1I_MERGED" "" "$S1P_CKPT" "$S1P_MERGED"

    OUR_BASE="$S1P_MERGED"
    RESUME_EXP=""
fi
echo ">>> Our-Helios-Base = $OUR_BASE"


# ############################################################
#  STAGE 2-INIT : Learn pyramid (LR=1e-4, rank=256)
#   base = Our-Helios-Base
# ############################################################
if [ "$START_STAGE" -le 3 ]; then
    setup_exp_dir "stage2_init" 3

    make_config "$CONFIGS/stage_2_action_init.yaml" "$_CFG" \
        "$_EXP" "$_VAL" "$OUR_BASE"
    do_train "Stage 2-init" "$_CFG" 9501

    S2I_CKPT=$(latest_ckpt "$_EXP")
    S2I_MERGED="$_EXP/merged_transformer"
    do_merge "$OUR_BASE" "" "$S2I_CKPT" "$S2I_MERGED"

    RESUME_EXP=""
fi
echo ">>> S2I_MERGED = $S2I_MERGED"


# ############################################################
#  STAGE 2-POST : Refine pyramid (LR=3e-5, rank=256, +patch LoRA)
#   base = merged S2-init
# ############################################################
if [ "$START_STAGE" -le 4 ]; then
    setup_exp_dir "stage2_post" 4

    make_config "$CONFIGS/stage_2_action_post.yaml" "$_CFG" \
        "$_EXP" "$_VAL" "$S2I_MERGED"
    do_train "Stage 2-post" "$_CFG" 9501

    S2P_CKPT=$(latest_ckpt "$_EXP")
    S2P_MERGED="$_EXP/merged_transformer"
    do_merge "$S2I_MERGED" "" "$S2P_CKPT" "$S2P_MERGED"

    OUR_MID="$S2P_MERGED"
    RESUME_EXP=""
fi
echo ">>> Our-Helios-Mid = $OUR_MID"


# ############################################################
#  ODE DATA GENERATION (between S2 and S3)
#   Teacher = OUR_MID (S2-post merged model)
#   Skipped if ODE_DATA_DIR is already set (pre-generated)
# ############################################################
if [ "$START_STAGE" -le 5 ] && [ -z "$ODE_DATA_DIR" ]; then
    ODE_PROMPT_TXT="data/helios/ode_prompts.txt"
    ODE_DATA_DIR="data/helios/ode_pairs"

    if [ -d "$ODE_DATA_DIR" ] && [ -n "$(ls -A "$ODE_DATA_DIR" 2>/dev/null)" ]; then
        echo ">>> ODE data already exists at $ODE_DATA_DIR, skipping generation."
    else
        extract_prompts "$ODE_PROMPT_TXT"
        gen_ode_data "$OUR_MID" "$ODE_PROMPT_TXT" "$ODE_DATA_DIR" 9503
    fi
fi
echo ">>> ODE_DATA_DIR = ${ODE_DATA_DIR:-N/A}"


# ############################################################
#  STAGE 3-ODE : ODE regression only (LR=2e-6, rank=256)
#   base = Our-Helios-Mid
# ############################################################
if [ "$START_STAGE" -le 5 ]; then
    setup_exp_dir "stage3_ode" 5

    make_config "$CONFIGS/stage_3_action_ode.yaml" "$_CFG" \
        "$_EXP" "$_VAL" "$OUR_MID" "" \
        "c['data_config']['ode_data_root'] = ['$ODE_DATA_DIR']"
    do_train "Stage 3-ode" "$_CFG" 9502 "scripts/accelerate_configs/multi_node_example_zero2.yaml"

    S3O_CKPT=$(latest_ckpt "$_EXP")
    S3O_MERGED="$_EXP/merged_transformer"
    do_merge "$OUR_MID" "" "$S3O_CKPT" "$S3O_MERGED"

    OUR_ODE="$S3O_MERGED"
    RESUME_EXP=""
fi
echo ">>> Our-Helios-ODE = $OUR_ODE"


# ############################################################
#  STAGE 3-POST : DMD adversarial distillation (LR=2e-6, rank=256)
#   base = merged S3-ode, teacher = Our-Helios-Base (flat dir)
# ############################################################
if [ "$START_STAGE" -le 6 ]; then
    setup_exp_dir "stage3_post" 6

    make_config "$CONFIGS/stage_3_action_post.yaml" "$_CFG" \
        "$_EXP" "$_VAL" "$OUR_ODE" "$MODELS/Helios-Base" \
        "c['model_config']['real_score_model_name_or_path'] = '$OUR_BASE'"
    do_train "Stage 3-post" "$_CFG" 9502 "scripts/accelerate_configs/multi_node_example_zero2.yaml"

    S3P_CKPT=$(latest_ckpt "$_EXP")
    S3P_MERGED="$_EXP/merged_transformer"
    do_merge "$OUR_ODE" "" "$S3P_CKPT" "$S3P_MERGED"

    OUR_DISTILLED="$S3P_MERGED"
    RESUME_EXP=""
fi

cd /mnt/bn/voyager-sg-l3/zhexiao.xiong
python runner.py
# ============================================================
echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║         PIPELINE COMPLETE                     ║"
echo "╠══════════════════════════════════════════════╣"
echo "║  Our-Helios-Base:      ${OUR_BASE:-N/A}"
echo "║  Our-Helios-Mid:       ${OUR_MID:-N/A}"
echo "║  Our-Helios-Distilled: ${OUR_DISTILLED:-N/A}"
echo "╚══════════════════════════════════════════════╝"


