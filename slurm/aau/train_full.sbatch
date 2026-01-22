#!/bin/bash
#SBATCH --job-name=soprano-danish
#SBATCH --output=soprano_train_%j.log
#SBATCH --error=soprano_train_%j.err
#SBATCH --partition=batch
#SBATCH --qos=allgpus
#SBATCH --gres=gpu:l40s:4
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00

# ============================================
# Soprano Danish TTS - Full Training
# AAU AI Cloud - 4x L40S GPUs
# ============================================
# Full pipeline:
#   1. Download full CoRal-TTS dataset (~18k samples)
#   2. Generate audio tokens
#   3. Train for 10k steps with DDP
# ============================================

set -eo pipefail

WORK_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$WORK_DIR"

mkdir -p logs outputs coral_danish_dataset

echo "============================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPUs: 4x L40S"
echo "Working dir: ${WORK_DIR}"
echo "============================================"

# Container configuration
CONTAINER="${CONTAINER:-/home/container/pytorch/pytorch_25.03.sif}"

if [ ! -f "$CONTAINER" ]; then
    echo "ERROR: Container not found at $CONTAINER"
    exit 1
fi

echo "Using container: $CONTAINER"

# Scratch directory
WORKDIR="${SLURM_TMPDIR:-${TMPDIR:-$(mktemp -d -t soprano-XXXXXX)}}"
mkdir -p "$WORKDIR"
echo "Temp workdir: $WORKDIR"

# Run everything inside the container
singularity exec --nv --cleanenv \
    --env WORKDIR="$WORKDIR",HF_HOME="$WORKDIR/hf-cache",TRANSFORMERS_CACHE="$WORKDIR/hf-cache" \
    "$CONTAINER" bash --noprofile --norc -c '
    set -eo pipefail
    cd "'"$WORK_DIR"'"

    echo "[soprano] Setting up Python environment..."
    python3 -m venv --system-site-packages "$WORKDIR/venv"
    . "$WORKDIR/venv/bin/activate"

    echo "[soprano] Installing dependencies..."
    python3 -m pip install --no-cache-dir -U pip
    python3 -m pip install --no-cache-dir transformers "datasets<3.0" huggingface_hub tqdm soundfile numpy scipy wandb librosa

    python3 -c "import torch; print(f\"[soprano] PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}\")"

    mkdir -p "$HF_HOME"

    # =========================================
    # Step 1: Prepare full dataset
    # =========================================
    echo "[soprano] Step 1: Preparing full dataset..."
    if [ ! -f "coral_danish_dataset/metadata.txt" ]; then
        python3 prepare_coral_dataset.py \
            --output-dir coral_danish_dataset
        # No --max-samples = use full dataset
    else
        echo "[soprano] Dataset already prepared, skipping..."
    fi

    # =========================================
    # Step 2: Generate audio tokens
    # =========================================
    echo "[soprano] Step 2: Generating audio tokens..."
    if [ ! -f "coral_danish_dataset/train.json" ]; then
        python3 generate_dataset.py \
            --input-dir coral_danish_dataset
    else
        echo "[soprano] Audio tokens already generated, skipping..."
    fi

    # =========================================
    # Step 3: Distributed training
    # =========================================
    echo "[soprano] Step 3: Training with 4 GPUs..."

    # DDP settings
    export MASTER_ADDR=localhost
    export MASTER_PORT=29500

    torchrun --nproc_per_node=4 train_distributed.py \
        --input-dir coral_danish_dataset \
        --save-dir outputs/soprano-danish-$(date +%Y%m%d) \
        --max-steps 20000 \
        --batch-size 12 \
        --lr 1e-4 \
        --val-freq 500 \
        --save-freq 2000 \
        --wandb \
        --wandb-project soprano-danish \
        --wandb-run "danish-tts-${SLURM_JOB_ID}"

    echo "[soprano] Training complete!"
    ls -la outputs/
'

echo "============================================"
echo "Training complete!"
echo "Model saved to outputs/"
echo "============================================"
