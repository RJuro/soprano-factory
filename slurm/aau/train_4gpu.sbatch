#!/bin/bash
#SBATCH --job-name=soprano-4gpu
#SBATCH --output=logs/soprano_%j.out
#SBATCH --error=logs/soprano_%j.err
#SBATCH --partition=batch
#SBATCH --qos=short
#SBATCH --mem=128G
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=03:00:00

# ============================================
# Soprano Danish TTS - 4 GPU Training
# AAU AI Cloud (CLAAUDIA)
# ============================================
# Max 4 GPUs per job on AAU AI Cloud
# Using QoS=short (3 hour limit, 4 GPU max)
#
# For longer runs, use multiple sequential jobs
# or request --qos=allgpus (if you have access)
#
# To target L40S nodes specifically, add:
#   #SBATCH --nodelist=a768-l40s-01
# ============================================

set -e
WORK_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$WORK_DIR"

# Configuration
NUM_GPUS=4
MAX_SAMPLES=18863          # Full dataset
MAX_STEPS=2500             # Reduced for 3hr limit (adjust based on speed)
BATCH_SIZE=48              # Per GPU, effective = 192
OUTPUT_DIR="outputs/soprano-danish-4gpu-$(date +%Y%m%d)"

mkdir -p logs outputs

echo "============================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPUs: ${NUM_GPUS}"
nvidia-smi --query-gpu=name --format=csv,noheader
echo "Effective batch size: $((BATCH_SIZE * NUM_GPUS))"
echo "Output: ${OUTPUT_DIR}"
echo "============================================"

# Check container exists
if [ ! -f "containers/pytorch.sif" ]; then
    echo "ERROR: Container not found. Run first:"
    echo "  sbatch slurm/aau/pull_container.sbatch"
    exit 1
fi

# Set Singularity environment - use node scratch for temp
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

# Run distributed training
echo "Starting distributed training on ${NUM_GPUS} GPUs..."
singularity exec --nv containers/pytorch.sif \
    torchrun --nproc_per_node=${NUM_GPUS} --master_port=29500 \
    train_distributed.py \
    --input-dir coral_danish_dataset \
    --save-dir "${OUTPUT_DIR}" \
    --max-steps "${MAX_STEPS}" \
    --batch-size "${BATCH_SIZE}" \
    --save-freq 500

echo "============================================"
echo "Training complete!"
echo "Model saved to: ${OUTPUT_DIR}"
echo "============================================"
