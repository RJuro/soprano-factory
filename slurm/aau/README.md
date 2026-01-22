# AAU AI Cloud (CLAAUDIA) Training Scripts

Scripts customized for Aalborg University's AI Cloud infrastructure.

## Quick Start

```bash
# 1. Pull the PyTorch container (run once)
sbatch slurm/aau/pull_container.sbatch

# 2. Wait for container download (~20-30 min)
squeue -u $USER

# 3. Submit training job
sbatch slurm/aau/train_4gpu.sbatch
```

## Available Scripts

| Script | GPUs | QoS | Time Limit | Use Case |
|--------|------|-----|------------|----------|
| `train_1gpu.sbatch` | 1 | normal | 48h | Development/testing |
| `train_4gpu.sbatch` | 4 | short | 3h | Quick training runs |
| `train_4gpu_allgpus.sbatch` | 4 | allgpus | 24h | Full training |
| `pull_container.sbatch` | 0 | short | 1h | Download NGC container |

## AAU AI Cloud Specifics

### GPU Types Available
- **L40S** (48GB): `a768-l40s-[01-06]` - 8 GPUs per node
- **A100** (40GB): `nv-ai-04` - 8 GPUs per node
- **A40** (48GB): Multiple nodes - 3-4 GPUs per node
- **V100** (32GB): `nv-ai-[02-03]` - 16 GPUs per node

### Resource Limits
- **Max 4 GPUs per job**
- **Max 8 GPUs per user** (across all jobs)

### QoS Options
| QoS | Max GPUs | Max Time | Notes |
|-----|----------|----------|-------|
| `short` | 4 | 3 hours | Good for testing |
| `normal` | 1 | 2 days | Single GPU work |
| `1gpulong` | 1 | 14 days | Long single-GPU jobs |
| `allgpus` | 8 | 21 days | May need approval |
| `deadline` | 8 | 14 days | Publication deadlines |

### Targeting Specific GPU Types

Add to your sbatch script:
```bash
# For L40S GPUs
#SBATCH --nodelist=a768-l40s-01

# For A100 GPUs
#SBATCH --nodelist=nv-ai-04
```

### Container System

AAU uses **Singularity** (not Docker). The scripts use NGC PyTorch containers:

```bash
# Container is pulled to:
containers/pytorch.sif

# Run commands inside container:
singularity exec --nv containers/pytorch.sif python script.py
```

### Scratch Space

Each node has fast NVMe scratch at `/raid`. Use for temp files:
```bash
export SINGULARITY_TMPDIR="/raid/${USER}/tmp"
```

## Monitoring Jobs

```bash
# Check queue
squeue -u $USER

# View job details
scontrol show job <jobid>

# Cancel job
scancel <jobid>

# View logs
tail -f logs/soprano_<jobid>.out
```

## Workflow for Full Training

Since `short` QoS has 3-hour limit, for full 10k steps:

### Option 1: Use allgpus QoS (if available)
```bash
sbatch slurm/aau/train_4gpu_allgpus.sbatch
```

### Option 2: Chain multiple short jobs
```bash
# First run (saves checkpoints every 500 steps)
sbatch slurm/aau/train_4gpu.sbatch

# Continue from checkpoint (edit script to load from checkpoint)
# ... modify train_distributed.py to support --resume
```

## Troubleshooting

### Container not found
```bash
sbatch slurm/aau/pull_container.sbatch
# Wait for completion, then retry
```

### Out of memory
Reduce batch size in the script:
```bash
BATCH_SIZE=32  # or 24
```

### NCCL timeout on multi-GPU
Add to script before singularity exec:
```bash
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
```

### Singularity cache errors
```bash
rm -rf /raid/${USER}/tmp /raid/${USER}/cache
mkdir -p /raid/${USER}/tmp /raid/${USER}/cache
```

## References

- [AAU AI Cloud System Overview](https://hpc.aau.dk/ai-cloud/system-overview/)
- [Running Jobs Guide](https://hpc.aau.dk/ai-lab/guides/running-jobs/)
- [Download Containers](https://hpc.aau.dk/ai-cloud/additional-guides/download-container-images/)
- [NGC PyTorch Catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)
