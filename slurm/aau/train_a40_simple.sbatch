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

# Use the container since pip/python not available on compute nodes
CONTAINER="/home/container/pytorch/pytorch_24.09.sif"

# Verify dataset
if [ ! -f "coral_danish_dataset/train.json" ]; then
    echo "ERROR: Dataset not found. Run data prep first."
    exit 1
fi

# Run distributed training on 4 GPUs inside container
# Install to local dir since container is read-only and ~/.local has old versions
echo "Starting training on 4 A40 GPUs..."
mkdir -p .pip_packages

singularity exec --nv \
    --env "PYTHONNOUSERSITE=1" \
    --env "PYTHONPATH=$PWD/.pip_packages" \
    "$CONTAINER" bash -c "
    pip install --quiet --no-cache-dir --target=.pip_packages \
        --no-deps transformers 'tokenizers>=0.19.0' huggingface_hub einops safetensors regex filelock pyyaml requests packaging

    torchrun --nproc_per_node=4 --master_port=29500 \
        train_distributed.py \
        --input-dir coral_danish_dataset \
        --save-dir outputs/soprano-danish-a40-\$(date +%Y%m%d_%H%M) \
        --max-steps 2500 \
        --batch-size 48 \
        --save-freq 500
"

echo "Done!"
