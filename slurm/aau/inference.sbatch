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
CONTAINER="/home/container/pytorch/pytorch_24.09.sif"
MODEL_PATH="${MODEL_PATH:-outputs/soprano-danish-a40-20260123_1228_step500}"
TEXT="${1:-Hej, dette er en test af Soprano.}"
OUTPUT_NAME="${2:-output_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_PATH="outputs/audio/${OUTPUT_NAME}.mp3"

echo "============================================"
echo "Soprano TTS Inference"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "============================================"
echo "Model: ${MODEL_PATH}"
echo "Text: ${TEXT}"
echo "Output: ${OUTPUT_PATH}"
echo "============================================"

# Clean old packages (same as training script)
rm -rf ~/.local/lib/python*/site-packages/tokenizers* 2>/dev/null || true
rm -rf ~/.local/lib/python*/site-packages/transformers* 2>/dev/null || true

singularity exec --nv "$CONTAINER" bash -c '
set -e

# Install compatible versions (same as training script)
pip install --user --no-cache-dir --force-reinstall -q \
    "numpy<2" \
    "tokenizers>=0.20" \
    "transformers>=4.51" \
    "huggingface_hub>=0.20" \
    pydub soundfile einops 2>/dev/null || true

python inference.py \
    --model "'"${MODEL_PATH}"'" \
    --text "'"${TEXT}"'" \
    --output "'"${OUTPUT_PATH}"'" \
    --temperature 0.7 \
    --top-p 0.95
'

echo "============================================"
echo "Done! Output saved to ${OUTPUT_PATH}"
echo "============================================"
