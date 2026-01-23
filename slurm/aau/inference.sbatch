#!/bin/bash
#SBATCH --job-name=soprano-infer
#SBATCH --output=logs/inference_%j.out
#SBATCH --error=logs/inference_%j.err
#SBATCH --partition=batch
#SBATCH --qos=short
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00

# ============================================
# Soprano TTS Inference - AAU AI Cloud
# ============================================
# Usage:
#   sbatch slurm/aau/inference.sbatch "Hej, mit navn er Soprano."
#   sbatch slurm/aau/inference.sbatch "Rød grød med fløde." danish_dessert
# ============================================

set -e
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs outputs/audio

# Configuration
MODEL_PATH="${MODEL_PATH:-outputs/soprano-danish-a40-20260123_1228_step500}"
TEXT="${1:-Hej, dette er en test af Soprano.}"
OUTPUT_NAME="${2:-output_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_PATH="outputs/audio/${OUTPUT_NAME}.mp3"

echo "============================================"
echo "Soprano TTS Inference"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "============================================"
echo "Model: ${MODEL_PATH}"
echo "Text: ${TEXT}"
echo "Output: ${OUTPUT_PATH}"
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

# Install soprano-tts and run inference
singularity exec --nv containers/pytorch.sif bash -c "
    pip install -q soprano-tts pydub soundfile 2>/dev/null || true
    python inference.py \
        --model '${MODEL_PATH}' \
        --text '${TEXT}' \
        --output '${OUTPUT_PATH}' \
        --temperature 0.7 \
        --top-p 0.95
"

echo "============================================"
echo "Done! Output saved to ${OUTPUT_PATH}"
echo "============================================"
