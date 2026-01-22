#!/bin/bash
#SBATCH --job-name=soprano-test
#SBATCH --output=soprano_test_output.log
#SBATCH --error=soprano_test_error.log
#SBATCH --partition=batch
#SBATCH --qos=short
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00

# ============================================
# Soprano Danish TTS - Quick Pipeline Test
# AAU AI Cloud - Uses system container
# ============================================
# This script tests the full pipeline with minimal data:
#   1. Download 100 samples from CoRal-TTS
#   2. Generate audio tokens
#   3. Train for 50 steps
# ============================================

set -eo pipefail

WORK_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$WORK_DIR"

mkdir -p logs outputs test_data test_output

echo "============================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Working dir: ${WORK_DIR}"
echo "============================================"

# Container configuration - use system container like hviske
CONTAINER="${CONTAINER:-/home/container/pytorch/pytorch_25.03.sif}"

if [ ! -f "$CONTAINER" ]; then
    echo "ERROR: Container not found at $CONTAINER"
    echo "Available containers:"
    ls -la /home/container/pytorch/ 2>/dev/null || echo "  (container dir not accessible)"
    exit 1
fi

echo "Using container: $CONTAINER"

# Scratch/work directory for temp files
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

    # Verify torch is available
    python3 -c "import torch; print(f\"[soprano] PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}\")"

    mkdir -p "$HF_HOME"

    echo "[soprano] Step 1: Preparing dataset (100 samples)..."
    if [ ! -f "test_data/metadata.txt" ]; then
        python3 prepare_coral_dataset.py \
            --output-dir test_data \
            --max-samples 100
    else
        echo "[soprano] Dataset already prepared, skipping..."
    fi

    echo "[soprano] Step 2: Generating audio tokens..."
    if [ ! -f "test_data/train.json" ]; then
        python3 generate_dataset.py \
            --input-dir test_data
    else
        echo "[soprano] Audio tokens already generated, skipping..."
    fi

    echo "[soprano] Step 3: Training for 50 steps..."
    python3 train_distributed.py \
        --input-dir test_data \
        --save-dir test_output \
        --max-steps 50 \
        --batch-size 4 \
        --val-freq 25

    echo "[soprano] Pipeline test complete!"
    echo "[soprano] Model saved to: test_output/"
    ls -la test_output/
'

echo "============================================"
echo "Test complete!"
echo "Check test_output/ for trained model"
echo "============================================"
