#!/bin/bash
#SBATCH --job-name=soprano-a40
#SBATCH --output=logs/soprano_a40_%j.out
#SBATCH --error=logs/soprano_a40_%j.err
#SBATCH --partition=batch
#SBATCH --qos=short
#SBATCH --gres=gpu:a40:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=02:59:00

cd $SLURM_SUBMIT_DIR

echo "============================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "============================================"

# Use 25.03 container which has PyTorch 2.6+ (supports SDPA with enable_gqa)
CONTAINER="/home/container/pytorch/pytorch_25.03.sif"

# Verify dataset
if [ ! -f "coral_danish_dataset/train.json" ]; then
    echo "ERROR: Dataset not found. Run data prep first."
    exit 1
fi

echo "Starting training on 4 A40 GPUs..."

# Clean old packages to avoid conflicts
rm -rf ~/.local/lib/python*/site-packages/tokenizers*
rm -rf ~/.local/lib/python*/site-packages/huggingface*
rm -rf ~/.local/lib/python*/site-packages/transformers*

singularity exec --nv "$CONTAINER" bash -c '
set -e

# Install compatible versions
pip install --user --no-cache-dir --force-reinstall \
    "numpy<2" \
    "tokenizers>=0.20" \
    "transformers>=4.51" \
    "huggingface_hub>=0.20" \
    einops \
    flash-attn --no-build-isolation 2>/dev/null || true

# Verify packages
echo "=== Package verification ==="
python -c "
import torch
import tokenizers, transformers
print(f\"PyTorch: {torch.__version__}\")
print(f\"tokenizers: {tokenizers.__version__}\")
print(f\"transformers: {transformers.__version__}\")
try:
    import flash_attn
    print(f\"flash_attn: {flash_attn.__version__}\")
except:
    print(\"flash_attn: not available\")
"
echo "==========================="

# Optimized training:
# - batch-size 24 with grad-accum 2 = effective batch 192 (24*4*2)
# - gradient-checkpointing saves memory, allows larger batches
# - compile uses torch.compile for faster training
torchrun --nproc_per_node=4 --master_port=29500 \
    train_distributed.py \
    --input-dir coral_danish_dataset \
    --save-dir outputs/soprano-danish-a40-$(date +%Y%m%d_%H%M) \
    --max-steps 2500 \
    --batch-size 24 \
    --grad-accum 2 \
    --gradient-checkpointing \
    --save-freq 500
'

echo "Done!"
