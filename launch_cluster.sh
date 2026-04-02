#!/bin/bash
# ╔══════════════════════════════════════════════════════════════╗
# ║  Cluster Job Bootstrap — submit this to Arnold/METIS/SLURM  ║
# ║                                                              ║
# ║  This script runs on EVERY node. It:                        ║
# ║    1) Installs system deps (apt)                            ║
# ║    2) Sets up Python environment (pip, only on node 0)      ║
# ║    3) Launches run_cluster.sh (multi-node training)         ║
# ║                                                              ║
# ║  Arnold example (4 nodes × 8 GPUs):                        ║
# ║    Startup command = bash Helios/launch_cluster.sh          ║
# ║                                                              ║
# ║  Training config env vars (optional, pass via job config):  ║
# ║    START_STAGE=5  OUR_MID=/path/to/mid  ...                ║
# ╚══════════════════════════════════════════════════════════════╝
set -euo pipefail

WORK="/mnt/bn/voyager-sg-l3/zhexiao.xiong"
cd "$WORK"

# ============================================================
#  Detect node rank early (needed for install gating)
# ============================================================
if [ -n "${ARNOLD_WORKER_HOSTS:-}" ]; then
    _NODE_RANK=${ARNOLD_ID:-${INDEX:-0}}
elif [ -n "${METIS_WORKER_0_HOST:-}" ]; then
    _NODE_RANK=${METIS_TASK_INDEX:-${INDEX:-0}}
elif [ -n "${SLURM_JOB_ID:-}" ]; then
    _NODE_RANK=${SLURM_NODEID:-0}
else
    _NODE_RANK=${NODE_RANK:-${RANK:-${INDEX:-0}}}
fi
echo ">>> [Node $_NODE_RANK] Bootstrap starting on $(hostname)"

# ============================================================
#  1) System packages (every node, idempotent)
# ============================================================
sudo apt-get update -qq
sudo apt-get install -y -qq libgl1 git git-lfs curl tmux htop zip 2>/dev/null || true

# ============================================================
#  2) Python environment setup
#     pip installs go to shared filesystem — only node 0 runs
#     them to avoid concurrent pip corruption.
# ============================================================
MINICONDA="$WORK/miniconda3"
export PATH="$MINICONDA/bin:$PATH"
export PYTHONNOUSERSITE=1

INSTALL_LOCK="$WORK/Helios/ckpts/helios/.install_done"

if [ "$_NODE_RANK" -eq 0 ]; then
    if [ ! -f "$INSTALL_LOCK" ] || [ "${FORCE_INSTALL:-0}" = "1" ]; then
        echo ">>> [Node 0] Installing Python dependencies..."
        cd "$WORK/Helios"
        pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
            --index-url https://download.pytorch.org/whl/cu126 2>&1 | tail -5
        bash install.sh
        pip install -r requirements_wm.txt 2>&1 | tail -5
        cd "$WORK"
        mkdir -p "$(dirname "$INSTALL_LOCK")"
        touch "$INSTALL_LOCK"
        echo ">>> [Node 0] Install complete."
    else
        echo ">>> [Node 0] Dependencies already installed (rm $INSTALL_LOCK to reinstall)."
    fi
else
    echo ">>> [Node $_NODE_RANK] Waiting for node 0 to finish install..."
    for i in $(seq 1 600); do
        [ -f "$INSTALL_LOCK" ] && break
        sleep 5
    done
    if [ ! -f "$INSTALL_LOCK" ]; then
        echo "ERROR: [Node $_NODE_RANK] Timed out waiting for install"
        exit 1
    fi
    echo ">>> [Node $_NODE_RANK] Install lock found, continuing."
fi

# ============================================================
#  3) Launch multi-node training
# ============================================================
echo ">>> [Node $_NODE_RANK] Launching run_cluster.sh..."
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export TORCH_DISTRIBUTED_DEBUG=${TORCH_DISTRIBUTED_DEBUG:-DETAIL}

cd "$WORK/Helios"
exec bash run_cluster.sh 2>&1 | tee "$WORK/Helios/ckpts/helios/node${_NODE_RANK}.log"
