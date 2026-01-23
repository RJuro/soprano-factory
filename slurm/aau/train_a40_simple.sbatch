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

# Verify dataset
if [ ! -f "coral_danish_dataset/train.json" ]; then
    echo "ERROR: Dataset not found. Run data prep first."
    exit 1
fi

echo "Starting training on 4 A40 GPUs..."

# Force reinstall tokenizers to get a working version (uninstall first to clear any old .so files)
pip uninstall -y tokenizers transformers 2>/dev/null || true
pip install --user --no-cache-dir \
    "numpy<2" \
    "tokenizers>=0.20" \
    "transformers>=4.44,<4.50" \
    huggingface_hub \
    einops \
    tqdm \
    soundfile

# Verify versions
echo "=== Package verification ==="
python -c "
import numpy, torch, transformers, tokenizers
print(f'numpy={numpy.__version__}')
print(f'torch={torch.__version__}')
print(f'transformers={transformers.__version__}')
print(f'tokenizers={tokenizers.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"
echo "==========================="

# Run distributed training
torchrun --nproc_per_node=4 --master_port=29500 \
    train_distributed.py \
    --input-dir coral_danish_dataset \
    --save-dir outputs/soprano-danish-a40-$(date +%Y%m%d_%H%M) \
    --max-steps 2500 \
    --batch-size 48 \
    --save-freq 500

echo "Done!"
