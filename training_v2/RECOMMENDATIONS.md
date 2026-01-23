# Training Framework Recommendations for Soprano TTS

## Executive Summary

For your Soprano Danish TTS training on AAU AI Cloud (A40 GPUs), I recommend:

**Primary: Unsloth** - Best speed/memory efficiency for your setup
**Alternative: Axolotl** - If you need multi-GPU scaling later

## Detailed Analysis

### 1. Unsloth (Recommended)

**Pros:**
- 2-5x faster training than standard PyTorch
- 50-80% less VRAM usage
- Native TTS fine-tuning support (added May 2025)
- Works great on single A40 (46GB)
- Built-in gradient checkpointing optimizations
- No approximations - same accuracy as standard training

**Cons:**
- Multi-GPU support is limited
- Newer framework, less battle-tested
- Some models may not be supported

**Best For:**
- Your current setup (single/few A40s)
- Quick iteration and experimentation
- Memory-constrained scenarios

**Key Features:**
```python
# Unsloth's optimized LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    use_gradient_checkpointing="unsloth",  # 3x less memory
)
```

### 2. Axolotl (Good Alternative)

**Pros:**
- Excellent multi-GPU support (DeepSpeed, FSDP)
- YAML-based config - easy to modify
- Large community, well-documented
- Supports many training methods (SFT, DPO, RLHF)
- Recently added audio model support (Voxtral)

**Cons:**
- Slightly slower than Unsloth on single GPU
- More abstraction layers
- Larger install footprint

**Best For:**
- Multi-GPU distributed training
- Production pipelines
- Teams with DevOps support

### 3. LLaMA-Factory (Alternative)

**Pros:**
- Web UI for easy management
- Supports 100+ models including Qwen3
- Good documentation
- Built-in evaluation tools

**Cons:**
- Heavier framework
- Less optimized for single GPU
- More focused on chat models than TTS

**Best For:**
- Users who prefer GUI
- Quick prototyping
- Educational purposes

## Performance Comparison

| Metric | Unsloth | Axolotl | LLaMA-Factory | Your Current |
|--------|---------|---------|---------------|--------------|
| Training Speed | 2-5x faster | 1x | 0.9x | 1x |
| VRAM Usage | 50-80% less | Standard | Standard | Standard |
| Multi-GPU | Limited | Excellent | Good | Manual DDP |
| TTS Support | Native | Via config | Limited | Manual |
| Ease of Use | High | High | Very High | Medium |

## Recommended Training Configuration

For your Soprano model on A40:

```yaml
# Optimal settings for Unsloth on A40
batch_size: 8          # Can go higher with Unsloth
grad_accum: 4          # Effective batch = 32
learning_rate: 2e-4
lora_r: 32             # Higher rank for TTS quality
lora_alpha: 32
max_seq_length: 2048
gradient_checkpointing: "unsloth"  # Critical for memory
```

## Migration Path

1. **Start with Unsloth** on single A40
   - Fastest iteration
   - ~1.5x-2x faster than current setup
   - Same or less memory usage

2. **Scale with Axolotl** if needed
   - When you need 4+ GPUs
   - For longer training runs
   - Configs are easily transferable

3. **Use current setup** as fallback
   - Already working
   - Full control
   - Good for debugging

## References

- [Unsloth TTS Docs](https://unsloth.ai/docs/basics/text-to-speech-tts-fine-tuning)
- [Unsloth Blog - TTS Fine-tuning](https://unsloth.ai/blog/tts)
- [Axolotl Documentation](https://docs.axolotl.ai/)
- [Framework Comparison 2025](https://blog.spheron.network/comparing-llm-fine-tuning-frameworks-axolotl-unsloth-and-torchtune-in-2025)
- [Modal Fine-tuning Guide](https://modal.com/blog/fine-tuning-llms)
