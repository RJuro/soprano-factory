#!/bin/bash
#SBATCH --job-name=soprano-a40
#SBATCH --output=logs/soprano_a40_%j.out
#SBATCH --error=logs/soprano_a40_%j.err
#SBATCH --partition=batch
#SBATCH --qos=short
#SBATCH --mem=128G
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=03:00:00
#SBATCH --nodelist=i256-a40-01,i256-a40-02

# ============================================
# Soprano Danish TTS - A40 Training
# AAU AI Cloud (CLAAUDIA)
# ============================================
# Targets A40 nodes: i256-a40-01, i256-a40-02
# 4 GPUs per node, 48GB VRAM each
#
# Prerequisites:
#   1. Container: sbatch slurm/aau/pull_container.sbatch
#   2. Dataset prepared in coral_danish_dataset/
# ============================================

set -e
WORK_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$WORK_DIR"

# Configuration
NUM_GPUS=4
MAX_STEPS=2500
BATCH_SIZE=48
MASTER_PORT=$((29500 + RANDOM % 1000))  # Random port to avoid conflicts
OUTPUT_DIR="outputs/soprano-danish-a40-$(date +%Y%m%d_%H%M)"
INPUT_DIR="coral_danish_dataset"

mkdir -p logs outputs

echo "============================================"
echo "Soprano TTS Training - A40"
echo "============================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPUs: ${NUM_GPUS}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "--------------------------------------------"
echo "Batch size: ${BATCH_SIZE} per GPU"
echo "Effective batch: $((BATCH_SIZE * NUM_GPUS))"
echo "Max steps: ${MAX_STEPS}"
echo "Master port: ${MASTER_PORT}"
echo "Input: ${INPUT_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "============================================"

# Verify container
if [ ! -f "containers/pytorch.sif" ]; then
    echo "ERROR: Container not found at containers/pytorch.sif"
    echo "Run first: sbatch slurm/aau/pull_container.sbatch"
    exit 1
fi

# Verify dataset
if [ ! -f "${INPUT_DIR}/train.json" ]; then
    echo "ERROR: Dataset not found at ${INPUT_DIR}/train.json"
    echo "Run data preparation first."
    exit 1
fi

# Set Singularity environment
export SINGULARITY_TMPDIR="/raid/${USER}/tmp"
export SINGULARITY_CACHEDIR="/raid/${USER}/cache"
mkdir -p "${SINGULARITY_TMPDIR}" "${SINGULARITY_CACHEDIR}"

# Install requirements in container (once)
echo "Installing Python dependencies..."
singularity exec --nv \
    --bind "$WORK_DIR:$WORK_DIR" \
    containers/pytorch.sif \
    pip install --user -q datasets soundfile huggingface_hub einops tqdm

# Run distributed training
echo ""
echo "Starting training on ${NUM_GPUS} A40 GPUs..."
echo "--------------------------------------------"

singularity exec --nv \
    --bind "$WORK_DIR:$WORK_DIR" \
    containers/pytorch.sif \
    torchrun \
        --nproc_per_node=${NUM_GPUS} \
        --master_port=${MASTER_PORT} \
    train_distributed.py \
        --input-dir "${INPUT_DIR}" \
        --save-dir "${OUTPUT_DIR}" \
        --max-steps "${MAX_STEPS}" \
        --batch-size "${BATCH_SIZE}" \
        --save-freq 500

echo ""
echo "============================================"
echo "Training complete!"
echo "Model saved to: ${OUTPUT_DIR}"
echo "============================================"
