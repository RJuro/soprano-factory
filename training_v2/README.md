# Soprano TTS Training v2 - Modern Approaches

This folder contains training configurations using modern fine-tuning frameworks.
Each offers different tradeoffs in terms of speed, memory efficiency, and features.

## Framework Comparison

| Framework | Speed | Memory | Multi-GPU | TTS Support | Best For |
|-----------|-------|--------|-----------|-------------|----------|
| **Unsloth** | 2-5x faster | 50-80% less | Limited | Yes (native) | Single GPU, speed |
| **Axolotl** | Standard | Good | Excellent | Via Voxtral | Multi-GPU, flexibility |
| **LLaMA-Factory** | Good | Good | Excellent | Via Qwen2-Audio | Web UI, ease of use |

## Soprano Model Architecture

Soprano uses **Qwen3** architecture for text-to-speech:
- Model: `ekwek/Soprano-1.1-80M` (80M parameters)
- Architecture: Qwen3 (transformer-based)
- Task: Text → Audio token prediction

## Recommended Approach

### For Single GPU (A40/A100):
**Use Unsloth** - Native TTS support, fastest training, lowest memory

### For Multi-GPU (4+ GPUs):
**Use Axolotl or LLaMA-Factory** - Better distributed training support

## Quick Start

### Unsloth (Recommended for A40)
```bash
cd unsloth
pip install unsloth
python train_soprano_unsloth.py
```

### Axolotl
```bash
cd axolotl
pip install axolotl
accelerate launch -m axolotl.cli.train soprano_config.yaml
```

### LLaMA-Factory
```bash
cd llama_factory
pip install llamafactory
llamafactory-cli train soprano_config.yaml
```

## Sources

- [Unsloth TTS Fine-tuning](https://unsloth.ai/docs/basics/text-to-speech-tts-fine-tuning)
- [Axolotl Documentation](https://docs.axolotl.ai/)
- [LLaMA-Factory GitHub](https://github.com/hiyouga/LlamaFactory)
- [Framework Comparison](https://blog.spheron.network/comparing-llm-fine-tuning-frameworks-axolotl-unsloth-and-torchtune-in-2025)
