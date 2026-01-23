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

# ============================================
# Soprano Danish TTS - 4x A40 GPU Training
# AAU AI Cloud (CLAAUDIA)
# ============================================
# Uses pre-existing PyTorch containers from cluster.
# Available containers: /home/container/pytorch/
#
# Usage: sbatch slurm/aau/train_a40_simple.sbatch
# ============================================

set -e
WORK_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$WORK_DIR"

# Configuration
NUM_GPUS=4
MAX_SAMPLES=18863
MAX_STEPS=2500
BATCH_SIZE=48
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
# pytorch_24.09.sif works well with transformers 4.44.x
CONTAINER_PATH="/home/container/pytorch/pytorch_24.09.sif"

if [ ! -f "${CONTAINER_PATH}" ]; then
    echo "ERROR: Container not found at ${CONTAINER_PATH}"
    echo "Available containers:"
    ls -la /home/container/pytorch/ 2>/dev/null || echo "  (directory not accessible)"
    exit 1
fi

echo "Using container: ${CONTAINER_PATH}"

# Set Singularity environment
export SINGULARITY_TMPDIR="/raid/${USER}/singularity_tmp"
export SINGULARITY_CACHEDIR="/raid/${USER}/singularity_cache"
mkdir -p "${SINGULARITY_TMPDIR}" "${SINGULARITY_CACHEDIR}"

# Define a script to run inside the container
SETUP_AND_RUN=$(cat << 'ENDSCRIPT'
#!/bin/bash
set -e

echo "Installing compatible dependencies..."
# Pin transformers to version compatible with container's PyTorch
# transformers 4.44.x works with PyTorch 2.4.x (in 24.09 container)
pip install --quiet --no-cache-dir \
    "transformers>=4.40.0,<4.45.0" \
    "huggingface_hub>=0.20.0" \
    "datasets>=2.14.0" \
    "soundfile>=0.12.0" \
    "einops>=0.7.0" \
    "tqdm>=4.66.0"

echo "Transformers version: $(python -c 'import transformers; print(transformers.__version__)')"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"

# Check if dataset exists
if [ ! -f "${INPUT_DIR}/train.json" ]; then
    echo "Preparing Coral Danish dataset..."
    python prepare_coral_dataset.py \
        --output-dir "${INPUT_DIR}" \
        --max-samples "${MAX_SAMPLES}"

    echo "Generating audio tokens..."
    python generate_dataset.py \
        --input-dir "${INPUT_DIR}"
fi

echo "Starting distributed training on ${NUM_GPUS} GPUs..."
torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=29500 \
    train_distributed.py \
    --input-dir "${INPUT_DIR}" \
    --save-dir "${OUTPUT_DIR}" \
    --max-steps "${MAX_STEPS}" \
    --batch-size "${BATCH_SIZE}" \
    --save-freq 500

ENDSCRIPT
)

# Export variables for the script
export INPUT_DIR="coral_danish_dataset"
export OUTPUT_DIR
export MAX_SAMPLES
export MAX_STEPS
export BATCH_SIZE
export NUM_GPUS

# Run the script inside the container
echo "Running training inside container..."
singularity exec --nv \
    --bind "${WORK_DIR}:${WORK_DIR}" \
    --pwd "${WORK_DIR}" \
    --env "INPUT_DIR=${INPUT_DIR}" \
    --env "OUTPUT_DIR=${OUTPUT_DIR}" \
    --env "MAX_SAMPLES=${MAX_SAMPLES}" \
    --env "MAX_STEPS=${MAX_STEPS}" \
    --env "BATCH_SIZE=${BATCH_SIZE}" \
    --env "NUM_GPUS=${NUM_GPUS}" \
    "${CONTAINER_PATH}" \
    bash -c "${SETUP_AND_RUN}"

echo "============================================"
echo "Training complete!"
echo "Model saved to: ${OUTPUT_DIR}"
echo "============================================"
