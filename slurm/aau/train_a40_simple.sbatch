#!/bin/bash
#SBATCH --job-name=soprano-a40
#SBATCH --output=soprano_a40_%j.out
#SBATCH --error=soprano_a40_%j.err
#SBATCH --partition=batch
#SBATCH --qos=short
#SBATCH --mem=128G
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a40:4
#SBATCH --time=03:00:00

# ============================================
# Soprano Danish TTS - A40 4-GPU Training
# AAU AI Cloud - Uses shared container
# ============================================

cd $SLURM_SUBMIT_DIR

# Shared PyTorch container on AAU (older version for CUDA 12.0 driver compatibility)
CONTAINER="/home/container/pytorch/pytorch_24.12.sif"

# Configuration
NUM_GPUS=4
MAX_STEPS=2500
BATCH_SIZE=48
MASTER_PORT=$((29500 + RANDOM % 1000))
OUTPUT_DIR="outputs/soprano-danish-a40-$(date +%Y%m%d_%H%M)"
INPUT_DIR="coral_danish_dataset"

echo "============================================"
echo "Soprano TTS Training - A40 x${NUM_GPUS}"
echo "============================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "Container: ${CONTAINER}"
echo "--------------------------------------------"

# Verify dataset exists
if [ ! -f "${INPUT_DIR}/train.json" ]; then
    echo "ERROR: Dataset not found at ${INPUT_DIR}/train.json"
    exit 1
fi

# Singularity setup
export SINGULARITY_TMPDIR="/raid/${USER}/tmp"
export SINGULARITY_CACHEDIR="/raid/${USER}/cache"
mkdir -p "${SINGULARITY_TMPDIR}" "${SINGULARITY_CACHEDIR}"

# Install deps in container
echo "Installing dependencies..."
singularity exec --nv "${CONTAINER}" \
    pip install --user -q transformers "datasets<3.0" huggingface_hub tqdm soundfile einops

echo ""
echo "Starting distributed training..."
echo "Batch: ${BATCH_SIZE}/GPU, effective: $((BATCH_SIZE * NUM_GPUS))"
echo "Steps: ${MAX_STEPS}"
echo "Output: ${OUTPUT_DIR}"
echo "--------------------------------------------"

# Run distributed training
singularity exec --nv "${CONTAINER}" \
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
echo "Done! Model saved to ${OUTPUT_DIR}/"
ls -la "${OUTPUT_DIR}/"
