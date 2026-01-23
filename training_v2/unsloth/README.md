# Soprano Danish TTS Training with Unsloth

This guide covers fine-tuning Soprano TTS for Danish using [Unsloth](https://unsloth.ai/), a library that makes training 2-5x faster with 50-80% less memory.

## Why Unsloth?

| Feature | Unsloth | Standard PyTorch |
|---------|---------|------------------|
| Training Speed | 2-5x faster | Baseline |
| Memory Usage | 50-80% less | Baseline |
| TTS Support | Native (May 2025) | Manual |
| Gradient Checkpointing | Optimized ("unsloth" mode) | Standard |
| LoRA Training | Optimized | Standard |

## Quick Start

### Google Colab (Recommended for Testing)

1. Open `Soprano_Danish_Unsloth_Colab.ipynb` in Google Colab
2. Select T4 GPU runtime
3. Run all cells
4. Upload your CORAL dataset when prompted

### Local / Cluster

```bash
pip install -r requirements.txt
python train_soprano_unsloth.py --data-dir /path/to/coral_danish_dataset
```

## Training Configuration

### Recommended Settings for TTS

```python
# Model loading
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="ekwek/Soprano-1.1-80M",
    max_seq_length=2048,
    load_in_4bit=False,  # Full precision for TTS quality
)

# LoRA configuration (higher rank for TTS)
model = FastLanguageModel.get_peft_model(
    model,
    r=32,                    # Higher rank = better quality
    lora_alpha=32,
    lora_dropout=0.0,        # No dropout for TTS
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    use_gradient_checkpointing="unsloth",  # Critical: 3x memory savings
)
```

### Training Arguments

```python
TrainingArguments(
    per_device_train_batch_size=4,   # Adjust for GPU memory
    gradient_accumulation_steps=4,    # Effective batch = 16
    learning_rate=2e-4,
    max_steps=2500,                   # ~1-2 hours on T4
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    bf16=True,                        # Use bf16 if supported
    optim="adamw_8bit",               # Memory-efficient optimizer
)
```

## Data Preparation

### CORAL Danish Dataset

The CORAL dataset should be formatted as JSON with the Soprano token format:

```json
[
    {"text": "<|audio|>token1 token2 token3...<|endoftext|>"},
    {"text": "<|audio|>token1 token2 token3...<|endoftext|>"}
]
```

### Speaker Selection (Important!)

For best results, filter to a **single speaker**:

```python
# Filter to female speakers only (matches Soprano's English voice better)
dataset = dataset.filter(lambda x: x["speaker_gender"] == "female")

# Or filter to specific speaker ID
dataset = dataset.filter(lambda x: x["speaker_id"] == "coral_f01")
```

**Why single speaker?**
- Soprano doesn't handle multiple voices well
- Consistent voice = consistent output quality
- 3 hours of one speaker > 10 hours of mixed speakers

### Curriculum Learning (Optional)

For better results with limited high-quality data:

```python
# Phase 1: Train on diverse Danish data (lower quality OK)
trainer.train(dataset=diverse_danish, max_steps=5000)

# Phase 2: Fine-tune on high-quality CORAL
trainer.train(dataset=coral_filtered, max_steps=2500)
```

## Memory Requirements

| GPU | VRAM | Batch Size | Grad Accum | Notes |
|-----|------|------------|------------|-------|
| T4 | 16GB | 2 | 4 | Colab free tier |
| A10 | 24GB | 4 | 4 | Good balance |
| A40 | 48GB | 8 | 2 | Fast training |
| A100 | 40/80GB | 8-16 | 2 | Fastest |

If you run out of memory:
1. Reduce `per_device_train_batch_size`
2. Enable 4-bit: `load_in_4bit=True`
3. Reduce `max_seq_length`

## Saving & Loading

### Save LoRA Adapters

```python
model.save_pretrained("soprano-danish-lora")
tokenizer.save_pretrained("soprano-danish-lora")
```

### Save Merged Model

```python
# Merge LoRA into base model (standalone, no adapter needed)
model.save_pretrained_merged("soprano-danish-merged", tokenizer)
```

### Load for Inference

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    "soprano-danish-lora",  # or merged model path
    max_seq_length=2048,
)
FastLanguageModel.for_inference(model)  # Enable fast inference
```

## Inference Example

```python
text = "Hej, dette er en test af den danske stemme."
inputs = tokenizer(text, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True,
    )

# Decode audio tokens and convert to audio
# (Use Soprano's audio decoding pipeline)
```

## Troubleshooting

### "CUDA out of memory"
- Reduce batch size
- Enable `load_in_4bit=True`
- Use `gradient_checkpointing="unsloth"`

### "Model not found" / Import errors
- Ensure transformers >= 4.51 (for Qwen3 support)
- Ensure tokenizers >= 0.20 (for new tokenizer format)

### Poor audio quality
- Increase LoRA rank (`r=64` or higher)
- Train longer (more steps)
- Use single-speaker data
- Disable 4-bit quantization

## References

- [Unsloth Documentation](https://docs.unsloth.ai/)
- [Unsloth TTS Fine-tuning Guide](https://unsloth.ai/docs/basics/text-to-speech-tts-fine-tuning)
- [Soprano Model](https://huggingface.co/ekwek/Soprano-1.1-80M)
- [Training Framework Comparison](https://blog.spheron.network/comparing-llm-fine-tuning-frameworks-axolotl-unsloth-and-torchtune-in-2025)
