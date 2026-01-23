#!/bin/bash
#SBATCH --job-name=soprano-infer
#SBATCH --output=logs/inference_%j.out
#SBATCH --error=logs/inference_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --partition=gpu

# Inference script for Soprano TTS
# Usage: sbatch slurm/inference.sbatch "Your text here" [output_name]
#
# Examples:
#   sbatch slurm/inference.sbatch "Hej, mit navn er Soprano."
#   sbatch slurm/inference.sbatch "Rød grød med fløde." danish_dessert

set -e
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs outputs/audio

source venv/bin/activate 2>/dev/null || true

# Install dependencies if needed
pip install -q soprano-tts pydub soundfile 2>/dev/null || true

# Configuration
MODEL_PATH="${MODEL_PATH:-outputs/soprano-danish-a40-20260123_1228_step500}"
TEXT="${1:-Hej, dette er en test af Soprano.}"
OUTPUT_NAME="${2:-output_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_PATH="outputs/audio/${OUTPUT_NAME}.mp3"

echo "============================================"
echo "Soprano TTS Inference"
echo "============================================"
echo "Model: ${MODEL_PATH}"
echo "Text: ${TEXT}"
echo "Output: ${OUTPUT_PATH}"
echo "============================================"

python inference.py \
    --model "${MODEL_PATH}" \
    --text "${TEXT}" \
    --output "${OUTPUT_PATH}" \
    --temperature 0.7 \
    --top-p 0.95

echo "Done! Output saved to ${OUTPUT_PATH}"
