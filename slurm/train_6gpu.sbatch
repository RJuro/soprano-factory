#!/bin/bash
#SBATCH --job-name=soprano-6gpu
#SBATCH --output=logs/soprano_%j.out
#SBATCH --error=logs/soprano_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=192G
#SBATCH --gres=gpu:l40s:6
#SBATCH --time=08:00:00
#SBATCH --partition=gpu

# 6x L40S configuration
# Effective batch size: 288 (48 per GPU)

set -e
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs outputs

source venv/bin/activate 2>/dev/null || true

# Prepare data if needed
if [ ! -f "coral_danish_dataset/train.json" ]; then
    python prepare_coral_dataset.py --output-dir coral_danish_dataset
    python generate_dataset.py --input-dir coral_danish_dataset
fi

# Train
torchrun --nproc_per_node=6 --master_port=29500 \
    train_distributed.py \
    --input-dir coral_danish_dataset \
    --save-dir outputs/soprano-danish-6gpu \
    --max-steps 10000 \
    --batch-size 48
