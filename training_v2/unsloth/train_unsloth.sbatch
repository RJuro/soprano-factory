#!/bin/bash
#SBATCH --job-name=soprano-unsloth
#SBATCH --output=logs/soprano_unsloth_%j.out
#SBATCH --error=logs/soprano_unsloth_%j.err
#SBATCH --partition=batch
#SBATCH --qos=short
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:59:00

# Unsloth is optimized for single GPU - uses it very efficiently
# Can train 2-5x faster with 50-80% less memory

cd $SLURM_SUBMIT_DIR

echo "============================================"
echo "Soprano TTS Training with Unsloth"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "============================================"

CONTAINER="/home/container/pytorch/pytorch_25.03.sif"

mkdir -p logs outputs

singularity exec --nv "$CONTAINER" bash -c '
set -e

# Install Unsloth and dependencies
pip install --user --no-cache-dir \
    "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" \
    "transformers>=4.51" \
    "tokenizers>=0.20" \
    "trl>=0.9" \
    "peft" \
    "datasets" \
    "huggingface_hub"

echo "=== Package versions ==="
python -c "
import unsloth
import transformers
print(f\"Unsloth: {unsloth.__version__}\")
print(f\"Transformers: {transformers.__version__}\")
"
echo "========================"

# Run training
python train_soprano_unsloth.py \
    --data-dir ../../coral_danish_dataset \
    --output-dir outputs/soprano-unsloth-$(date +%Y%m%d_%H%M) \
    --max-steps 2500 \
    --batch-size 4 \
    --grad-accum 4 \
    --lora-r 32
'

echo "Done!"
