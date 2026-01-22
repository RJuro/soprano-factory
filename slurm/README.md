# SLURM Training Scripts

Scripts for training Soprano TTS on Danish (Coral TTS) using SLURM clusters with L40S GPUs.

## Quick Start

```bash
# 1. Setup environment (run once)
bash slurm/setup.sh

# 2. Submit job
sbatch slurm/train_4gpu.sbatch
```

## Available Configurations

| Script | GPUs | Batch Size | Est. Time | Memory |
|--------|------|------------|-----------|--------|
| `train_2gpu.sbatch` | 2x L40S | 96 | ~24h | 64GB |
| `train_4gpu.sbatch` | 4x L40S | 192 | ~12h | 128GB |
| `train_6gpu.sbatch` | 6x L40S | 288 | ~8h | 192GB |

## Manual Steps

### 1. Prepare Dataset (on login node)

```bash
# Download and convert Coral TTS to LJSpeech format
python prepare_coral_dataset.py --output-dir coral_danish_dataset

# Generate audio tokens (CPU-intensive, ~30 min)
python generate_dataset.py --input-dir coral_danish_dataset
```

### 2. Submit Training Job

```bash
# Choose based on available GPUs
sbatch slurm/train_4gpu.sbatch
```

### 3. Monitor Job

```bash
# Check job status
squeue -u $USER

# View logs
tail -f logs/soprano_<jobid>.out
```

## Customization

### Adjust for your cluster

Edit the SBATCH headers in the `.sbatch` files:

```bash
#SBATCH --partition=gpu        # Your GPU partition name
#SBATCH --gres=gpu:l40s:4      # GPU type and count
#SBATCH --account=myproject    # Add if needed
```

### Change training parameters

```bash
# In the sbatch file, modify:
--max-steps 10000      # More steps = better quality
--batch-size 48        # Reduce if OOM (try 32 or 24)
--lr 5e-4              # Learning rate
```

### Use different dataset size

```bash
# For testing (2000 samples)
python prepare_coral_dataset.py --output-dir coral_danish_dataset --max-samples 2000

# Full dataset (~18k samples)
python prepare_coral_dataset.py --output-dir coral_danish_dataset
```

## Output

Models are saved to `outputs/soprano-danish-Ngpu/`:
- `config.json` - Model config
- `model.safetensors` - Model weights
- `tokenizer.json` - Tokenizer

Checkpoints saved every 1000 steps to `outputs/soprano-danish-Ngpu_stepN/`

## Troubleshooting

### Out of Memory
Reduce batch size in the sbatch file:
```bash
--batch-size 32   # or 24
```

### Job killed / timeout
Increase time limit:
```bash
#SBATCH --time=48:00:00
```

### NCCL errors
Try setting:
```bash
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1  # If InfiniBand issues
```
