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

CONTAINER="/home/container/pytorch/pytorch_24.09.sif"

# Verify dataset
if [ ! -f "coral_danish_dataset/train.json" ]; then
    echo "ERROR: Dataset not found. Run data prep first."
    exit 1
fi

echo "Starting training on 4 A40 GPUs..."

# Completely nuke old tokenizers to fix ModelWrapper error
rm -rf ~/.local/lib/python*/site-packages/tokenizers*
rm -rf ~/.local/lib/python*/site-packages/huggingface*
rm -rf ~/.local/lib/python*/site-packages/transformers*
rm -rf ~/.cache/huggingface/hub/models--ekwek--Soprano-80M 2>/dev/null || true

singularity exec --nv "$CONTAINER" bash -c '
set -e

# Force reinstall with compatible versions
# Need transformers>=4.51 for qwen3 support, tokenizers>=0.20 for new format
pip install --user --no-cache-dir --force-reinstall \
    "numpy<2" \
    "tokenizers>=0.20" \
    "transformers>=4.51" \
    "huggingface_hub>=0.20" \
    einops

# Verify what is actually loaded
echo "=== Package verification ==="
python -c "
import tokenizers, transformers
print(f\"tokenizers={tokenizers.__version__} from {tokenizers.__file__}\")
print(f\"transformers={transformers.__version__} from {transformers.__file__}\")
"
echo "==========================="

torchrun --nproc_per_node=4 --master_port=29500 \
    train_distributed.py \
    --input-dir coral_danish_dataset \
    --save-dir outputs/soprano-danish-a40-$(date +%Y%m%d_%H%M) \
    --max-steps 2500 \
    --batch-size 48 \
    --save-freq 500
'

echo "Done!"
