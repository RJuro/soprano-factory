#!/bin/bash
#SBATCH --job-name=soprano-a40
#SBATCH --output=logs/soprano_a40_%j.out
#SBATCH --error=logs/soprano_a40_%j.err
#SBATCH --partition=batch
#SBATCH --qos=short
#SBATCH --mem=128G
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:a40:4
#SBATCH --time=02:59:00

# ============================================
# Soprano Danish TTS - 4x A40 GPU Training
# AAU AI Cloud (CLAAUDIA)
# ============================================
# Uses pre-existing PyTorch containers from cluster.
# Assumes dataset is already prepared in coral_danish_dataset/
#
# Usage: sbatch slurm/aau/train_a40_simple.sbatch
# ============================================

set -e
WORK_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$WORK_DIR"

# Configuration
NUM_GPUS=4
MAX_STEPS=2500
BATCH_SIZE=48
INPUT_DIR="coral_danish_dataset"
OUTPUT_DIR="outputs/soprano-danish-a40-$(date +%Y%m%d_%H%M)"

mkdir -p logs outputs

echo "============================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPUs: ${NUM_GPUS}"
echo "Working dir: ${WORK_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "============================================"

# Show GPU info
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "============================================"

# Use pre-existing container from cluster
CONTAINER_PATH="/home/container/pytorch/pytorch_24.09.sif"

if [ ! -f "${CONTAINER_PATH}" ]; then
    echo "ERROR: Container not found at ${CONTAINER_PATH}"
    echo "Available containers:"
    ls -la /home/container/pytorch/ 2>/dev/null || echo "  (directory not accessible)"
    exit 1
fi

echo "Using container: ${CONTAINER_PATH}"

# Verify dataset exists
if [ ! -f "${INPUT_DIR}/train.json" ]; then
    echo "ERROR: Dataset not found at ${INPUT_DIR}/train.json"
    echo "Please prepare the dataset first."
    exit 1
fi

# Set Singularity environment (with fallback to /tmp if /raid not available)
if [ -d "/raid/${USER}" ]; then
    export SINGULARITY_TMPDIR="/raid/${USER}/singularity_tmp"
    export SINGULARITY_CACHEDIR="/raid/${USER}/singularity_cache"
else
    export SINGULARITY_TMPDIR="/tmp/${USER}/singularity_tmp"
    export SINGULARITY_CACHEDIR="/tmp/${USER}/singularity_cache"
fi
mkdir -p "${SINGULARITY_TMPDIR}" "${SINGULARITY_CACHEDIR}" 2>/dev/null || true

# Run training inside container
# PYTHONNOUSERSITE=1 prevents loading packages from ~/.local
echo "Running training inside container..."
singularity exec --nv \
    --bind "${WORK_DIR}:${WORK_DIR}" \
    --pwd "${WORK_DIR}" \
    --env "PYTHONNOUSERSITE=1" \
    "${CONTAINER_PATH}" \
    bash -c "
set -e

echo 'Installing compatible dependencies...'
pip install --quiet --no-cache-dir \
    'transformers>=4.40.0,<4.45.0' \
    'tokenizers>=0.19.0' \
    'huggingface_hub>=0.20.0' \
    'einops>=0.7.0'

echo 'Versions:'
python -c \"import transformers, tokenizers, torch; print(f'transformers={transformers.__version__}, tokenizers={tokenizers.__version__}, torch={torch.__version__}')\"

echo 'Starting distributed training on ${NUM_GPUS} GPUs...'
torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=29500 \
    train_distributed.py \
    --input-dir ${INPUT_DIR} \
    --save-dir ${OUTPUT_DIR} \
    --max-steps ${MAX_STEPS} \
    --batch-size ${BATCH_SIZE} \
    --save-freq 500
"

echo "============================================"
echo "Training complete!"
echo "Model saved to: ${OUTPUT_DIR}"
echo "============================================"
