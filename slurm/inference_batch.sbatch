#!/bin/bash
#SBATCH --job-name=soprano-batch
#SBATCH --output=logs/inference_batch_%j.out
#SBATCH --error=logs/inference_batch_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --partition=gpu

# Batch inference script for Soprano TTS
# Usage: sbatch slurm/inference_batch.sbatch texts.txt
#
# Input file format (one text per line):
#   Hej, mit navn er Soprano.
#   Rød grød med fløde.
#   København er Danmarks hovedstad.

set -e
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs outputs/audio

source venv/bin/activate 2>/dev/null || true

# Install dependencies if needed
pip install -q soprano-tts pydub soundfile 2>/dev/null || true

# Configuration
MODEL_PATH="${MODEL_PATH:-outputs/soprano-danish-a40-20260123_1228_step500}"
INPUT_FILE="${1:-texts.txt}"
OUTPUT_DIR="outputs/audio/batch_$(date +%Y%m%d_%H%M%S)"

echo "============================================"
echo "Soprano TTS Batch Inference"
echo "============================================"
echo "Model: ${MODEL_PATH}"
echo "Input: ${INPUT_FILE}"
echo "Output dir: ${OUTPUT_DIR}"
echo "============================================"

python inference.py \
    --model "${MODEL_PATH}" \
    --file "${INPUT_FILE}" \
    --output-dir "${OUTPUT_DIR}" \
    --output "audio.mp3" \
    --temperature 0.7 \
    --top-p 0.95

echo "Done! Outputs saved to ${OUTPUT_DIR}/"
