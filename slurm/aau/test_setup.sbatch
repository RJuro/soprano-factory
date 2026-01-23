#!/bin/bash
#SBATCH --job-name=test-soprano
#SBATCH --output=logs/test_soprano_%j.out
#SBATCH --error=logs/test_soprano_%j.err
#SBATCH --partition=batch
#SBATCH --qos=short
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00

# ============================================
# Test Script - Verify Container & Imports
# Run this first to check everything works
# ============================================

set -e
WORK_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$WORK_DIR"

mkdir -p logs

echo "============================================"
echo "Testing Soprano training setup"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "============================================"

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo "============================================"

# Use pre-existing container from cluster
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

echo "Running tests inside container..."
singularity exec --nv "${CONTAINER_PATH}" bash -c '
set -e
echo "=== Installing dependencies ==="
pip install --quiet --no-cache-dir \
    "transformers>=4.40.0,<4.45.0" \
    "huggingface_hub>=0.20.0" \
    "einops>=0.7.0"

echo ""
echo "=== Version check ==="
python -c "
import torch
import transformers
print(f\"PyTorch: {torch.__version__}\")
print(f\"CUDA available: {torch.cuda.is_available()}\")
print(f\"CUDA version: {torch.version.cuda}\")
print(f\"GPU: {torch.cuda.get_device_name(0)}\")
print(f\"Transformers: {transformers.__version__}\")
"

echo ""
echo "=== Testing imports ==="
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
print(\"AutoModelForCausalLM: OK\")
print(\"AutoTokenizer: OK\")
"

echo ""
echo "=== Testing model load ==="
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

print(\"Loading Soprano-80M tokenizer...\")
tokenizer = AutoTokenizer.from_pretrained(\"ekwek/Soprano-80M\")
print(f\"Vocab size: {tokenizer.vocab_size}\")

print(\"Loading Soprano-80M model...\")
model = AutoModelForCausalLM.from_pretrained(\"ekwek/Soprano-80M\")
model = model.to(torch.bfloat16).cuda()
print(f\"Model params: {sum(p.numel() for p in model.parameters()):,}\")
print(f\"Model device: {next(model.parameters()).device}\")

print(\"\")
print(\"=== ALL TESTS PASSED ===\")
"
'

echo "============================================"
echo "Setup verification complete!"
echo "You can now run: sbatch slurm/aau/train_a40_simple.sbatch"
echo "============================================"
