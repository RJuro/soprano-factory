#!/bin/bash
#SBATCH --job-name=soprano-4gpu-long
#SBATCH --output=logs/soprano_%j.out
#SBATCH --error=logs/soprano_%j.err
#SBATCH --partition=batch
#SBATCH --qos=allgpus
#SBATCH --mem=128G
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00

# ============================================
# Soprano Danish TTS - 4 GPU Long Training
# AAU AI Cloud (CLAAUDIA)
# ============================================
# Uses QoS=allgpus for longer runtime (up to 21 days)
# Note: allgpus QoS may require special access
#
# To target specific L40S nodes:
#   #SBATCH --nodelist=a768-l40s-01
# ============================================

set -e
WORK_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$WORK_DIR"

# Configuration
NUM_GPUS=4
MAX_SAMPLES=18863          # Full dataset
MAX_STEPS=10000            # Full training
BATCH_SIZE=48              # Per GPU
OUTPUT_DIR="outputs/soprano-danish-4gpu-$(date +%Y%m%d-%H%M)"

mkdir -p logs outputs

echo "============================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPUs: ${NUM_GPUS}"
nvidia-smi --query-gpu=name --format=csv,noheader
echo "Effective batch size: $((BATCH_SIZE * NUM_GPUS))"
echo "Max steps: ${MAX_STEPS}"
echo "Output: ${OUTPUT_DIR}"
echo "============================================"

# Check container
if [ ! -f "containers/pytorch.sif" ]; then
    echo "ERROR: Container not found. Run: sbatch slurm/aau/pull_container.sbatch"
    exit 1
fi

# Singularity temp dirs on scratch
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
echo "Starting distributed training..."
singularity exec --nv containers/pytorch.sif \
    torchrun --nproc_per_node=${NUM_GPUS} --master_port=29500 \
    train_distributed.py \
    --input-dir coral_danish_dataset \
    --save-dir "${OUTPUT_DIR}" \
    --max-steps "${MAX_STEPS}" \
    --batch-size "${BATCH_SIZE}" \
    --save-freq 1000

echo "============================================"
echo "Training complete: ${OUTPUT_DIR}"
echo "============================================"
