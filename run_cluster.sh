#!/bin/bash
set -euo pipefail
trap 'echo ">>> FATAL: run_cluster.sh failed at line $LINENO, command: $BASH_COMMAND, exit code: $?" >&2' ERR
cd /mnt/bn/voyager-sg-l3/zhexiao.xiong/Helios

# ╔══════════════════════════════════════════════════════════════╗
# ║  Helios Action World Model — Multi-Node Cluster Training    ║
# ║                                                              ║
# ║  4 nodes × 8 GPUs = 32 GPUs (configurable)                 ║
# ║  Run this script on EVERY node with appropriate NODE_RANK.  ║
# ║                                                              ║
# ║  Required env vars:                                          ║
# ║    MASTER_ADDR   — IP/hostname of node 0                    ║
# ║    NODE_RANK     — this node's rank (0, 1, 2, 3)           ║
# ║                                                              ║
# ║  Optional env vars:                                          ║
# ║    MASTER_PORT   (default: 9505)                            ║
# ║    NUM_NODES     (default: 4)                               ║
# ║    GPUS_PER_NODE (default: 8)                               ║
# ║                                                              ║
# ║  Usage examples:                                             ║
# ║    # Manual launch on each node:                            ║
# ║    MASTER_ADDR=10.0.0.1 NODE_RANK=0 bash run_cluster.sh    ║
# ║    MASTER_ADDR=10.0.0.1 NODE_RANK=1 bash run_cluster.sh    ║
# ║    ...                                                       ║
# ║                                                              ║
# ║    # SLURM:                                                  ║
# ║    srun --nodes=4 --ntasks-per-node=1 bash run_cluster.sh  ║
# ║                                                              ║
# ║  Effective batch sizes (batch × grad_accum × 32 GPUs):     ║
# ║    S1 init/post : 2 × 2 × 32 = 128  (was 32 on 8 GPUs)   ║
# ║    S2 init      : 1 × 1 × 32 = 32   (was 8)               ║
# ║    S2 post      : 1 × 2 × 32 = 64   (was 16)              ║
# ║    S3 ode/post  : 1 × 1 × 32 = 32   (was 8)               ║
# ║  Adjust gradient_accumulation_steps in YAML configs or set  ║
# ║  GRAD_ACCUM_DIVISOR=4 to auto-compensate.                  ║
# ╚══════════════════════════════════════════════════════════════╝

# ============================================================
#  Multi-node cluster settings (auto-detect Arnold / METIS / SLURM)
# ============================================================
GPUS_PER_NODE=${GPUS_PER_NODE:-8}

if [ -n "${ARNOLD_WORKER_HOSTS:-}" ]; then
    # --- ByteDance Arnold ---
    # Format: [ipv6]:port,[ipv6]:port,...  or  host:port,host:port,...
    IFS=',' read -ra _RAW_HOSTS <<< "$ARNOLD_WORKER_HOSTS"
    NUM_NODES=${#_RAW_HOSTS[@]}
    NODE_RANK=${ARNOLD_ID:-${INDEX:-0}}
    _FIRST="${_RAW_HOSTS[0]}"
    if [[ "$_FIRST" == \[* ]]; then
        # IPv6: [addr]:port — extract addr (strip brackets) and port
        MASTER_ADDR=${MASTER_ADDR:-$(echo "$_FIRST" | sed 's/^\[//;s/\]:.*//')}
        MASTER_PORT=${MASTER_PORT:-$(echo "$_FIRST" | sed 's/.*\]://')}
    else
        # IPv4: addr:port
        MASTER_ADDR=${MASTER_ADDR:-$(echo "$_FIRST" | cut -d: -f1)}
        MASTER_PORT=${MASTER_PORT:-$(echo "$_FIRST" | cut -d: -f2)}
    fi
    echo ">>> Auto-detected Arnold: ${NUM_NODES} nodes, this=rank${NODE_RANK}"

elif [ -n "${METIS_WORKER_0_HOST:-}" ]; then
    # --- ByteDance METIS ---
    _N=0; while [ -n "$(eval echo \${METIS_WORKER_${_N}_HOST:-})" ]; do _N=$((_N+1)); done
    NUM_NODES=$_N
    NODE_RANK=${METIS_TASK_INDEX:-${INDEX:-0}}
    MASTER_ADDR=${MASTER_ADDR:-$METIS_WORKER_0_HOST}
    MASTER_PORT=${MASTER_PORT:-${METIS_WORKER_0_PORT:-9505}}
    echo ">>> Auto-detected METIS: ${NUM_NODES} nodes, this=rank${NODE_RANK}"

elif [ -n "${SLURM_JOB_ID:-}" ]; then
    # --- SLURM ---
    NUM_NODES=${SLURM_NNODES:-4}
    NODE_RANK=${SLURM_NODEID:-0}
    if [ -z "${MASTER_ADDR:-}" ]; then
        MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
    fi
    MASTER_PORT=${MASTER_PORT:-9505}
    echo ">>> Auto-detected SLURM: ${NUM_NODES} nodes, this=rank${NODE_RANK}"

else
    # --- Manual / fallback ---
    NUM_NODES=${NUM_NODES:-4}
    NODE_RANK=${NODE_RANK:-${RANK:-${INDEX:-0}}}
    MASTER_PORT=${MASTER_PORT:-9505}
fi

MASTER_ADDR=${MASTER_ADDR:?"ERROR: MASTER_ADDR must be set (IP/hostname of node 0)"}
MASTER_PORT=${MASTER_PORT:-9505}
NUM_PROCESSES=$((NUM_NODES * GPUS_PER_NODE))

# Divide gradient_accumulation_steps by this factor (default 1 = no change)
GRAD_ACCUM_DIVISOR=${GRAD_ACCUM_DIVISOR:-1}


echo ""
echo ">>> [Node $NODE_RANK/$NUM_NODES] MASTER=$MASTER_ADDR:$MASTER_PORT  GPUS_PER_NODE=$GPUS_PER_NODE  TOTAL_GPUS=$NUM_PROCESSES"
echo ""

# ============================================================
#  Environment
# ============================================================
MINICONDA="/mnt/bn/voyager-sg-l3/zhexiao.xiong/miniconda3"
PY="$MINICONDA/bin/python3.13"
export PATH="$MINICONDA/bin:$PATH"
export PYTHONNOUSERSITE=1
hash -r

PY_VER="$("$PY" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")"
_NV_PKGS="$MINICONDA/lib/python${PY_VER}/site-packages/nvidia"
export LD_LIBRARY_PATH="$_NV_PKGS/cudnn/lib:$_NV_PKGS/cusparselt/lib:$_NV_PKGS/cublas/lib:$_NV_PKGS/cuda_runtime/lib:$_NV_PKGS/nvjitlink/lib:$_NV_PKGS/cuda_cupti/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/mnt/bn/voyager-sg-l3/zhexiao.xiong/Helios:${PYTHONPATH:-}"
export WANDB_API_KEY="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"
export WANDB_BASE_URL="https://api.wandb.ai"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

# Only node 0 reports to wandb to avoid duplicate logs
if [ "$NODE_RANK" -ne 0 ]; then
    export WANDB_MODE=disabled
fi

# NCCL tuning for multi-node
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-1800}

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
START_STAGE=${START_STAGE:-1}

#   RESUME_RUN: if interrupted, set to the unified run directory
#   (e.g. ckpts/helios/20260328_153000) to resume from START_STAGE.
RESUME_RUN="${RESUME_RUN:-}"

S1I_MERGED="${S1I_MERGED:-}"
OUR_BASE="${OUR_BASE:-}"
S2I_MERGED="${S2I_MERGED:-}"
OUR_MID="${OUR_MID:-}"
OUR_ODE="${OUR_ODE:-}"

ODE_DATA_DIR="${ODE_DATA_DIR:-}"
ODE_MAX_PROMPTS="${ODE_MAX_PROMPTS:-5000}"

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
        echo "  Set it via: $name=/path/to/merged_transformer bash run_cluster.sh"
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

echo ">>> START_STAGE=$START_STAGE  RESUME_RUN=${RESUME_RUN:-<new>}"
echo ""

# ============================================================
#  Session directory — all 6 stages under one timestamped folder
#  Layout:  $CKPTS/<timestamp>/stage1_init/
#                              stage1_post/
#                              stage2_init/  ...
#  Node 0 creates the dir; other nodes learn via broadcast.
# ============================================================
if [ "$NODE_RANK" -eq 0 ]; then
    if [ -n "$RESUME_RUN" ]; then
        RUN_DIR="$RESUME_RUN"
    else
        RUN_DIR="$CKPTS/$(date +%Y%m%d_%H%M%S)"
    fi
    RUN_OUTPUTS="$OUTPUTS/$(basename "$RUN_DIR")"
    mkdir -p "$RUN_DIR" "$RUN_OUTPUTS"
fi

# ============================================================
#  Multi-node launch: use torchrun (no SSH needed)
#  DeepSpeed is enabled via env vars for stages that need it.
# ============================================================
echo ">>> [Node $NODE_RANK] Using torchrun for multi-node launch (no SSH)"

# ============================================================
#  File-based barrier for inter-stage synchronization
#  (uses shared filesystem visible to all nodes)
# ============================================================
BARRIER_DIR="$CKPTS/.cluster_barriers"
if [ "$NODE_RANK" -eq 0 ]; then
    rm -rf "$BARRIER_DIR" 2>/dev/null || true
    mkdir -p "$BARRIER_DIR"
    touch "$BARRIER_DIR/.ready"
else
    local_waited=0
    while [ ! -f "$BARRIER_DIR/.ready" ]; do
        sleep 2
        local_waited=$((local_waited + 2))
        if [ "$local_waited" -ge 120 ]; then
            echo "ERROR: [Node $NODE_RANK] Timed out waiting for Node 0 to initialize barrier dir"
            exit 1
        fi
    done
fi

barrier_signal() {
    local name="$1"
    sync 2>/dev/null || true
    touch "$BARRIER_DIR/${name}.done"
    echo ">>> [Node $NODE_RANK] Barrier '$name' — signaled"
}

barrier_wait() {
    local name="$1" timeout="${2:-7200}"
    local waited=0
    echo ">>> [Node $NODE_RANK] Barrier '$name' — waiting..."
    while [ ! -f "$BARRIER_DIR/${name}.done" ]; do
        sleep 5
        waited=$((waited + 5))
        if [ "$waited" -ge "$timeout" ]; then
            echo "ERROR: [Node $NODE_RANK] Barrier '$name' timed out after ${timeout}s"
            exit 1
        fi
    done
    sleep 1
    echo ">>> [Node $NODE_RANK] Barrier '$name' — passed"
}

# Broadcast shell variables from node 0 to all other nodes.
#   Usage: broadcast_state "barrier_name" VAR1 VAR2 VAR3
broadcast_state() {
    local name="$1"; shift
    if [ "$NODE_RANK" -eq 0 ]; then
        local sf="$BARRIER_DIR/${name}.env"
        : > "$sf"
        for v in "$@"; do
            printf '%s=%q\n' "$v" "${!v}" >> "$sf"
        done
        barrier_signal "$name"
    else
        barrier_wait "$name"
        source "$BARRIER_DIR/${name}.env"
    fi
}

# Broadcast RUN_DIR so all nodes use the same session directory
broadcast_state "run_dir" RUN_DIR RUN_OUTPUTS
echo ">>> [Node $NODE_RANK] RUN_DIR     = $RUN_DIR"
echo ">>> [Node $NODE_RANK] RUN_OUTPUTS = $RUN_OUTPUTS"

# ============================================================
#  Helper functions
# ============================================================

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

make_config() {
    local tpl="$1" out="$2" exp="$3" val="$4"
    local tf="${5:-}" pipe="${6:-$MODELS/Helios-Base}" extra="${7:-}"

    "$PY" -c "
import yaml, sys, os

def is_flat_local_dir(p):
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

$extra

rsm = c.get('model_config', {}).get('real_score_model_name_or_path', '')
if rsm:
    validate_local_path(rsm, 'real_score_model_name_or_path')
    if is_flat_local_dir(rsm):
        c['model_config']['critic_subfolder'] = ''

# Scale gradient_accumulation_steps for multi-node
divisor = int('$GRAD_ACCUM_DIVISOR')
if divisor > 1:
    ga = c.get('training_config', {}).get('gradient_accumulation_steps', 1)
    c['training_config']['gradient_accumulation_steps'] = max(1, ga // divisor)

with open('$out', 'w') as f:
    yaml.dump(c, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
print(f'Config saved: $out')
"
}

do_merge() {
    local base="$1" sub="$2" ckpt="$3" out="$4"
    echo ""
    echo ">>> MERGE: $ckpt → $out"

    # Clean up incomplete merge from previous attempt
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

do_train() {
    local stage_name="$1" cfg="$2" ds_json="${3:-}"
    echo ""
    echo "════════════════════════════════════════════"
    echo "  TRAINING:  $stage_name  [Node $NODE_RANK/$NUM_NODES, ${NUM_PROCESSES} GPUs]"
    echo "  Config:    $cfg"
    echo "  DeepSpeed: ${ds_json:-off}"
    echo "════════════════════════════════════════════"

    if [ -n "$ds_json" ]; then
        export ACCELERATE_USE_DEEPSPEED=true
        export ACCELERATE_DEEPSPEED_CONFIG_FILE="$(pwd)/$ds_json"
    else
        unset ACCELERATE_USE_DEEPSPEED 2>/dev/null || true
        unset ACCELERATE_DEEPSPEED_CONFIG_FILE 2>/dev/null || true
    fi

    "$PY" -m torch.distributed.run \
        --nproc_per_node="$GPUS_PER_NODE" \
        --nnodes="$NUM_NODES" \
        --node_rank="$NODE_RANK" \
        --master_addr="$MASTER_ADDR" \
        --master_port="$MASTER_PORT" \
        train_helios.py --config "$cfg"

    echo ">>> [Node $NODE_RANK] Training '$stage_name' finished, cleaning up..."
    sleep 5
}

extract_prompts() {
    local out="$1"
    echo ">>> Extracting prompts from training data → $out  (max=${ODE_MAX_PROMPTS})"
    "$PY" -u -c "
import os, glob, torch, random, time

max_prompts = int('${ODE_MAX_PROMPTS}')
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

gen_ode_data() {
    local teacher="$1" prompt_txt="$2" ode_out="$3" port="${4:-9503}"
    echo ""
    echo "════════════════════════════════════════════"
    echo "  ODE DATA GENERATION (node 0 only, ${GPUS_PER_NODE} GPUs, mode=${ODE_SAMPLE_TYPE:-t2v})"
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

    torchrun --nproc_per_node "$GPUS_PER_NODE" --master_port "$port" \
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

setup_exp_dir() {
    local prefix="$1"
    _EXP="$RUN_DIR/$prefix"
    _VAL="$RUN_OUTPUTS/$prefix"
    mkdir -p "$_EXP"
    _CFG="$_EXP/train_config.yaml"
}


# ############################################################
#  STAGE 1-INIT : Learn AR generation (LR=5e-5, rank=128)
# ############################################################
if [ "$START_STAGE" -le 1 ]; then
    if [ "$NODE_RANK" -eq 0 ]; then
        setup_exp_dir "stage1_init"
        make_config "$CONFIGS/stage_1_action_init.yaml" "$_CFG" \
            "$_EXP" "$_VAL" "$MODELS/Helios-Base"
    fi
    broadcast_state "s1i_cfg" _EXP _VAL _CFG

    do_train "Stage 1-init" "$_CFG"

    if [ "$NODE_RANK" -eq 0 ]; then
        S1I_CKPT=$(latest_ckpt "$_EXP")
        S1I_MERGED="$_EXP/merged_transformer"
        do_merge "$MODELS/Helios-Base" "transformer" "$S1I_CKPT" "$S1I_MERGED"
    fi
    broadcast_state "s1i_merge" S1I_MERGED
fi
echo ">>> S1I_MERGED = $S1I_MERGED"


# ############################################################
#  STAGE 1-POST : Refine AR (LR=3e-5, rank=128)
# ############################################################
if [ "$START_STAGE" -le 2 ]; then
    if [ "$NODE_RANK" -eq 0 ]; then
        setup_exp_dir "stage1_post"
        make_config "$CONFIGS/stage_1_action_post.yaml" "$_CFG" \
            "$_EXP" "$_VAL" "$S1I_MERGED"
    fi
    broadcast_state "s1p_cfg" _EXP _VAL _CFG

    do_train "Stage 1-post" "$_CFG"

    if [ "$NODE_RANK" -eq 0 ]; then
        S1P_CKPT=$(latest_ckpt "$_EXP")
        S1P_MERGED="$_EXP/merged_transformer"
        do_merge "$S1I_MERGED" "" "$S1P_CKPT" "$S1P_MERGED"
        OUR_BASE="$S1P_MERGED"
    fi
    broadcast_state "s1p_merge" OUR_BASE
fi
echo ">>> Our-Helios-Base = $OUR_BASE"


# ############################################################
#  STAGE 2-INIT : Learn pyramid (LR=1e-4, rank=256)
# ############################################################
if [ "$START_STAGE" -le 3 ]; then
    if [ "$NODE_RANK" -eq 0 ]; then
        setup_exp_dir "stage2_init"
        make_config "$CONFIGS/stage_2_action_init.yaml" "$_CFG" \
            "$_EXP" "$_VAL" "$OUR_BASE"
    fi
    broadcast_state "s2i_cfg" _EXP _VAL _CFG

    do_train "Stage 2-init" "$_CFG"

    if [ "$NODE_RANK" -eq 0 ]; then
        S2I_CKPT=$(latest_ckpt "$_EXP")
        S2I_MERGED="$_EXP/merged_transformer"
        do_merge "$OUR_BASE" "" "$S2I_CKPT" "$S2I_MERGED"
    fi
    broadcast_state "s2i_merge" S2I_MERGED
fi
echo ">>> S2I_MERGED = $S2I_MERGED"


# ############################################################
#  STAGE 2-POST : Refine pyramid (LR=3e-5, rank=256)
# ############################################################
if [ "$START_STAGE" -le 4 ]; then
    if [ "$NODE_RANK" -eq 0 ]; then
        setup_exp_dir "stage2_post"
        make_config "$CONFIGS/stage_2_action_post.yaml" "$_CFG" \
            "$_EXP" "$_VAL" "$S2I_MERGED"
    fi
    broadcast_state "s2p_cfg" _EXP _VAL _CFG

    do_train "Stage 2-post" "$_CFG"

    if [ "$NODE_RANK" -eq 0 ]; then
        S2P_CKPT=$(latest_ckpt "$_EXP")
        S2P_MERGED="$_EXP/merged_transformer"
        do_merge "$S2I_MERGED" "" "$S2P_CKPT" "$S2P_MERGED"
        OUR_MID="$S2P_MERGED"
    fi
    broadcast_state "s2p_merge" OUR_MID
fi
echo ">>> Our-Helios-Mid = $OUR_MID"


# ############################################################
#  ODE DATA GENERATION (between S2 and S3)
#  Node 0 only — data goes to shared filesystem
# ############################################################
if [ "$START_STAGE" -le 5 ] && [ -z "$ODE_DATA_DIR" ]; then
    ODE_PROMPT_TXT="data/helios/ode_prompts.txt"
    ODE_DATA_DIR="data/helios/ode_pairs"

    if [ "$NODE_RANK" -eq 0 ]; then
        if [ -d "$ODE_DATA_DIR" ] && [ -n "$(ls -A "$ODE_DATA_DIR" 2>/dev/null)" ]; then
            echo ">>> ODE data already exists at $ODE_DATA_DIR, skipping generation."
        else
            extract_prompts "$ODE_PROMPT_TXT"
            gen_ode_data "$OUR_MID" "$ODE_PROMPT_TXT" "$ODE_DATA_DIR" 9503
        fi
    fi
    broadcast_state "ode_data" ODE_DATA_DIR
fi
echo ">>> ODE_DATA_DIR = ${ODE_DATA_DIR:-N/A}"


# ############################################################
#  STAGE 3-ODE : ODE regression only (LR=2e-6, rank=256)
# ############################################################
if [ "$START_STAGE" -le 5 ]; then
    if [ "$NODE_RANK" -eq 0 ]; then
        setup_exp_dir "stage3_ode"
        make_config "$CONFIGS/stage_3_action_ode.yaml" "$_CFG" \
            "$_EXP" "$_VAL" "$OUR_MID" "" \
            "c['data_config']['ode_data_root'] = ['$ODE_DATA_DIR']"
    fi
    broadcast_state "s3o_cfg" _EXP _VAL _CFG

    do_train "Stage 3-ode" "$_CFG" "scripts/accelerate_configs/zero2.json"

    if [ "$NODE_RANK" -eq 0 ]; then
        S3O_CKPT=$(latest_ckpt "$_EXP")
        S3O_MERGED="$_EXP/merged_transformer"
        do_merge "$OUR_MID" "" "$S3O_CKPT" "$S3O_MERGED"
        OUR_ODE="$S3O_MERGED"
    fi
    broadcast_state "s3o_merge" OUR_ODE
fi
echo ">>> Our-Helios-ODE = $OUR_ODE"


# ############################################################
#  STAGE 3-POST : DMD adversarial distillation (LR=2e-6)
# ############################################################
if [ "$START_STAGE" -le 6 ]; then
    if [ "$NODE_RANK" -eq 0 ]; then
        setup_exp_dir "stage3_post"
        make_config "$CONFIGS/stage_3_action_post.yaml" "$_CFG" \
            "$_EXP" "$_VAL" "$OUR_ODE" "$MODELS/Helios-Base" \
            "c['model_config']['real_score_model_name_or_path'] = '$OUR_BASE'"
    fi
    broadcast_state "s3p_cfg" _EXP _VAL _CFG

    do_train "Stage 3-post" "$_CFG" "scripts/accelerate_configs/zero3.json"

    if [ "$NODE_RANK" -eq 0 ]; then
        S3P_CKPT=$(latest_ckpt "$_EXP")
        S3P_MERGED="$_EXP/merged_transformer"
        do_merge "$OUR_ODE" "" "$S3P_CKPT" "$S3P_MERGED"
        OUR_DISTILLED="$S3P_MERGED"
    fi
    broadcast_state "s3p_merge" OUR_DISTILLED
fi

# All nodes burn local GPUs to hold the allocation after training
cd /mnt/bn/voyager-sg-l3/zhexiao.xiong
"$PY" runner.py --local

# ============================================================
echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║  PIPELINE COMPLETE  [Node $NODE_RANK/$NUM_NODES]"
echo "╠══════════════════════════════════════════════════╣"
echo "║  Run directory:        $RUN_DIR"
echo "║  Our-Helios-Base:      ${OUR_BASE:-N/A}"
echo "║  Our-Helios-Mid:       ${OUR_MID:-N/A}"
echo "║  Our-Helios-Distilled: ${OUR_DISTILLED:-N/A}"
echo "╚══════════════════════════════════════════════════╝"
