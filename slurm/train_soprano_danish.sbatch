#!/bin/bash
#SBATCH --job-name=soprano-danish
#SBATCH --output=logs/soprano-danish_%j.out
#SBATCH --error=logs/soprano-danish_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:l40s:4
#SBATCH --time=24:00:00
#SBATCH --partition=gpu

# ============================================
# Soprano Danish TTS Training on SLURM
# ============================================
# Adjust these SBATCH parameters for your cluster:
#   --gres=gpu:l40s:N  where N = 2, 4, or 6
#   --partition=gpu    your GPU partition name
#   --time=24:00:00    adjust based on dataset size
# ============================================

# Configuration
NUM_GPUS=${SLURM_GPUS_ON_NODE:-4}
MAX_SAMPLES=18863                    # Full dataset (set lower for testing)
MAX_STEPS=10000                      # Increase for better results
BATCH_SIZE_PER_GPU=48                # L40S 48GB can handle this
EFFECTIVE_BATCH_SIZE=$((BATCH_SIZE_PER_GPU * NUM_GPUS))

# Paths
WORK_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
DATA_DIR="${WORK_DIR}/coral_danish_dataset"
OUTPUT_DIR="${WORK_DIR}/outputs/soprano-danish-$(date +%Y%m%d-%H%M%S)"
LOG_DIR="${WORK_DIR}/logs"

# Create directories
mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

echo "============================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPUs: ${NUM_GPUS} x L40S"
echo "Effective batch size: ${EFFECTIVE_BATCH_SIZE}"
echo "Max samples: ${MAX_SAMPLES}"
echo "Max steps: ${MAX_STEPS}"
echo "Output: ${OUTPUT_DIR}"
echo "============================================"

# Load modules (adjust for your cluster)
module purge
module load cuda/12.1 || true
module load python/3.10 || true

# Activate virtual environment (adjust path)
if [ -f "${WORK_DIR}/venv/bin/activate" ]; then
    source "${WORK_DIR}/venv/bin/activate"
elif [ -f "${HOME}/.venv/soprano/bin/activate" ]; then
    source "${HOME}/.venv/soprano/bin/activate"
fi

# Install dependencies if needed
pip install -q torch torchaudio transformers datasets huggingface_hub tqdm soundfile

# Step 1: Prepare dataset (if not already done)
if [ ! -f "${DATA_DIR}/train.json" ]; then
    echo "Preparing dataset..."
    python "${WORK_DIR}/prepare_coral_dataset.py" \
        --output-dir "${DATA_DIR}" \
        --max-samples "${MAX_SAMPLES}"

    echo "Generating audio tokens..."
    python "${WORK_DIR}/generate_dataset.py" \
        --input-dir "${DATA_DIR}"
fi

# Step 2: Run training
echo "Starting training..."
if [ "${NUM_GPUS}" -gt 1 ]; then
    # Multi-GPU training with torchrun
    torchrun \
        --nproc_per_node="${NUM_GPUS}" \
        --master_port=29500 \
        "${WORK_DIR}/train_distributed.py" \
        --input-dir "${DATA_DIR}" \
        --save-dir "${OUTPUT_DIR}" \
        --max-steps "${MAX_STEPS}" \
        --batch-size "${BATCH_SIZE_PER_GPU}"
else
    # Single GPU training
    python "${WORK_DIR}/train_a100.py" \
        --input-dir "${DATA_DIR}" \
        --save-dir "${OUTPUT_DIR}" \
        --max-steps "${MAX_STEPS}"
fi

echo "============================================"
echo "Training complete!"
echo "Model saved to: ${OUTPUT_DIR}"
echo "============================================"
