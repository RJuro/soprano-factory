#!/bin/bash
#SBATCH --job-name=soprano-1gpu
#SBATCH --output=logs/soprano_%j.out
#SBATCH --error=logs/soprano_%j.err
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00

# ============================================
# Soprano Danish TTS - Single GPU Training
# AAU AI Cloud (CLAAUDIA)
# ============================================
# QoS options:
#   short:    max 4 GPU, 3 hours
#   normal:   max 1 GPU, 2 days
#   1gpulong: max 1 GPU, 14 days
#   allgpus:  max 8 GPU, 21 days (if available)
# ============================================

set -e
WORK_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$WORK_DIR"

# Configuration
MAX_SAMPLES=18863          # Full dataset
MAX_STEPS=10000
BATCH_SIZE=48
OUTPUT_DIR="outputs/soprano-danish-1gpu-$(date +%Y%m%d)"

mkdir -p logs outputs

echo "============================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Output: ${OUTPUT_DIR}"
echo "============================================"

# Check container exists
if [ ! -f "containers/pytorch.sif" ]; then
    echo "ERROR: Container not found. Run first:"
    echo "  sbatch slurm/aau/pull_container.sbatch"
    exit 1
fi

# Set Singularity environment
export SINGULARITY_TMPDIR="/raid/${USER}/tmp"
export SINGULARITY_CACHEDIR="/raid/${USER}/cache"
mkdir -p "${SINGULARITY_TMPDIR}" "${SINGULARITY_CACHEDIR}"

# Prepare dataset if needed
if [ ! -f "coral_danish_dataset/train.json" ]; then
    echo "Preparing dataset..."
    singularity exec --nv containers/pytorch.sif \
        pip install --user datasets soundfile huggingface_hub

    singularity exec --nv containers/pytorch.sif \
        python prepare_coral_dataset.py \
        --output-dir coral_danish_dataset \
        --max-samples "${MAX_SAMPLES}"

    echo "Generating audio tokens..."
    singularity exec --nv containers/pytorch.sif \
        python generate_dataset.py \
        --input-dir coral_danish_dataset
fi

# Run training
echo "Starting training..."
singularity exec --nv containers/pytorch.sif \
    python train_distributed.py \
    --input-dir coral_danish_dataset \
    --save-dir "${OUTPUT_DIR}" \
    --max-steps "${MAX_STEPS}" \
    --batch-size "${BATCH_SIZE}"

echo "============================================"
echo "Training complete!"
echo "Model saved to: ${OUTPUT_DIR}"
echo "============================================"
