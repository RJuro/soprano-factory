#!/bin/bash
#SBATCH --job-name=pull-pytorch
#SBATCH --output=logs/pull-pytorch_%j.out
#SBATCH --error=logs/pull-pytorch_%j.err
#SBATCH --partition=batch
#SBATCH --qos=short
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00

# ============================================
# Pull PyTorch NGC Container for AAU AI Cloud
# Run once before training jobs
# ============================================

set -e
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs containers

# Set Singularity cache directories (required on AAU)
export SINGULARITY_TMPDIR="${SLURM_SUBMIT_DIR}/tmp"
export SINGULARITY_CACHEDIR="${SLURM_SUBMIT_DIR}/cache"
mkdir -p "${SINGULARITY_TMPDIR}" "${SINGULARITY_CACHEDIR}"

echo "Pulling PyTorch container from NGC..."
echo "This may take 20-30 minutes..."

# Pull latest PyTorch NGC container
# See https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch for versions
singularity pull --force containers/pytorch.sif docker://nvcr.io/nvidia/pytorch:24.01-py3

echo "Container saved to: containers/pytorch.sif"
echo ""
echo "Clean up cache..."
rm -rf "${SINGULARITY_TMPDIR}" "${SINGULARITY_CACHEDIR}"

echo "Done! You can now submit training jobs."
