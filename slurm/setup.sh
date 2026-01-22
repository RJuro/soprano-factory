#!/bin/bash
# Setup script for Soprano training on SLURM cluster
# Run this once before submitting jobs

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

cd "$REPO_DIR"

echo "Setting up Soprano training environment..."

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install torch torchaudio transformers datasets huggingface_hub tqdm soundfile

# Create directories
mkdir -p logs outputs

# Prepare dataset (optional - can also be done in job)
read -p "Prepare Coral Danish dataset now? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Preparing dataset..."
    python prepare_coral_dataset.py --output-dir coral_danish_dataset
    echo "Generating audio tokens (this may take a while)..."
    python generate_dataset.py --input-dir coral_danish_dataset
fi

echo ""
echo "Setup complete!"
echo ""
echo "Submit a job with:"
echo "  sbatch slurm/train_2gpu.sbatch   # 2x L40S"
echo "  sbatch slurm/train_4gpu.sbatch   # 4x L40S"
echo "  sbatch slurm/train_6gpu.sbatch   # 6x L40S"
