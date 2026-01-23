# Research: Training Soprano with Unsloth

## Executive Summary

This document analyzes the feasibility of using **Unsloth** for fine-tuning the Soprano TTS model as an alternative to the current "factory" training approach.

**Key Finding**: Soprano-80M uses **Qwen3ForCausalLM** architecture, which is **natively supported by Unsloth**. This makes migration feasible, though several challenges need to be addressed.

---

## 1. Current Factory Approach Analysis

### Architecture
- **Model**: `ekwek/Soprano-80M` - 79.7M parameters
- **Base Architecture**: `Qwen3ForCausalLM` (standard transformer)
- **Vocab Size**: 8192 tokens
  - Audio tokens: 3-8003 (8001 audio codes)
  - Special tokens: [STOP], [TEXT], [START], etc.

### Training Pipeline
```
LJSpeech Dataset → Audio Encoder (Vocos) → Token Sequences → CLM Training → Fine-tuned Model
```

### Data Format
```
[STOP][TEXT]{text_prompt}[START]{audio_tokens}[STOP]
```

### Key Training Features
- Custom loss separation (audio vs text loss)
- Token packing/sequence collation
- AdamW optimizer with WSD schedule
- bfloat16 mixed precision
- Batch size 4-48 depending on GPU

---

## 2. Unsloth Capabilities

### Native Support
- **Qwen3 models**: Fully supported with optimized kernels
- **TTS models**: sesame/csm-1b, Orpheus-3B, Whisper
- **Training modes**: Full fine-tuning, LoRA, QLoRA (4-bit, 8-bit)
- **Performance**: 2x faster training, 50-70% less VRAM

### TTS-Specific Features
- Multi-modal input support (text + audio)
- Custom `auto_model` parameter for non-standard architectures
- Works with any transformers-compatible model

---

## 3. Potential Challenges

### Challenge 1: Custom Tokenization Format
**Problem**: Soprano uses a custom token format where audio is represented as `[0]`, `[1]`, `[2]`, etc., wrapped in special delimiters.

**Impact**: Unsloth's standard data processing may not handle this format correctly.

**Severity**: Medium

**Solution**:
- Use Unsloth's `formatting_func` parameter in SFTTrainer
- Pre-tokenize data to match expected format
- Keep using the existing `AutoTokenizer.from_pretrained('ekwek/Soprano-80M')`

```python
def formatting_func(example):
    text, audio = example['text'], example['audio']
    return f"[STOP][TEXT]{text}[START]{''.join([f'[{x}]' for x in audio])}[STOP]"
```

---

### Challenge 2: Custom Loss Function
**Problem**: Factory approach separates audio loss (tokens 3-8003) from text loss with different weighting (`text_factor=0.0` or `0.01`).

**Impact**: Standard Unsloth/TRL SFTTrainer uses uniform cross-entropy loss across all tokens.

**Severity**: High - Could affect model quality

**Solutions**:

**Option A: Custom Trainer Subclass**
```python
from trl import SFTTrainer

class SopranoTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs["labels"]

        # Separate audio vs text loss
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), reduction='none')
        audio_mask = (labels >= 3) & (labels <= 8003)
        audio_loss = loss[audio_mask.view(-1)].mean()
        text_loss = loss[~audio_mask.view(-1)].mean()

        total_loss = audio_loss + self.text_factor * text_loss
        return (total_loss, outputs) if return_outputs else total_loss
```

**Option B: Mask Text Tokens**
- Set labels for text tokens to -100 (ignored in loss)
- Only train on audio token prediction
- Simpler but loses text conditioning signal

**Option C: Accept Uniform Loss**
- Use standard loss, accept potential quality differences
- May still work well due to audio tokens being majority of sequence
- Fastest path to test Unsloth compatibility

---

### Challenge 3: Token Packing / Sequence Collation
**Problem**: Factory uses custom `collate_pack` function that:
- Concatenates multiple samples to fill `seq_len=1024`
- Handles variable-length audio sequences efficiently

**Impact**: Unsloth's default data collation may waste GPU memory on padding.

**Severity**: Medium - Affects efficiency, not correctness

**Solution**:
- Use Unsloth's `packing=True` parameter (supported in TRL)
- Or implement custom data collator:

```python
from transformers import DataCollatorForLanguageModeling

class PackedDataCollator(DataCollatorForLanguageModeling):
    def __call__(self, features):
        # Custom packing logic
        packed = pack_sequences(features, self.seq_len)
        return super().__call__(packed)
```

---

### Challenge 4: Audio Encoder Integration
**Problem**: The Vocos audio encoder is separate from the LLM and must be run offline to convert audio → tokens.

**Impact**: None for LLM training, but end-to-end pipeline requires extra step.

**Severity**: Low

**Solution**:
- Keep using `generate_dataset.py` for audio encoding (unchanged)
- Unsloth only trains the LLM backbone, not the codec
- This matches how TTS models like CSM-1B work

---

### Challenge 5: No Standard TTS Processor
**Problem**: Unsloth TTS examples use `AutoProcessor` (e.g., for CSM). Soprano only has a tokenizer.

**Impact**: Can't directly use Unsloth's TTS notebook patterns.

**Severity**: Low

**Solution**:
- Load model with `auto_model=Qwen3ForCausalLM` explicitly
- Use tokenizer directly instead of processor
- Soprano is simpler (text-only input, no speaker embeddings)

```python
from unsloth import FastModel
from transformers import Qwen3ForCausalLM

model, tokenizer = FastModel.from_pretrained(
    model_name="ekwek/Soprano-80M",
    max_seq_length=1024,
    dtype=torch.bfloat16,
    auto_model=Qwen3ForCausalLM,
    load_in_4bit=False,  # Full precision for TTS quality
)
```

---

### Challenge 6: Learning Rate Schedule
**Problem**: Factory uses custom WSD (Warmup-Stable-Decay) schedule. Unsloth/TRL defaults to cosine.

**Impact**: Minor - both are effective schedules.

**Severity**: Low

**Solution**:
- Configure in TrainingArguments:
```python
training_args = TrainingArguments(
    learning_rate=5e-4,
    warmup_ratio=0.1,
    lr_scheduler_type="constant_with_warmup",  # Or "cosine"
)
```

---

## 4. Migration Strategy

### Phase 1: Validate Basic Loading
```python
from unsloth import FastModel
from transformers import Qwen3ForCausalLM, AutoTokenizer

# Test if Soprano loads correctly
model, tokenizer = FastModel.from_pretrained(
    model_name="ekwek/Soprano-80M",
    max_seq_length=1024,
    auto_model=Qwen3ForCausalLM,
)

# Verify tokenizer works
test = tokenizer("[STOP][TEXT]Hello[START][100][200][STOP]")
print(test)
```

### Phase 2: Test with LoRA (Fast Iteration)
```python
model = FastModel.get_peft_model(
    model,
    r=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    lora_dropout=0,
)
```

### Phase 3: Full Fine-tuning
```python
model, tokenizer = FastModel.from_pretrained(
    model_name="ekwek/Soprano-80M",
    full_finetuning=True,
    max_seq_length=1024,
)
```

### Phase 4: Custom Loss (If Needed)
Implement `SopranoTrainer` subclass if audio quality suffers with uniform loss.

---

## 5. Recommended Approach

### Quick Start (Recommended First)
1. Use Unsloth with standard SFTTrainer
2. Keep existing data preparation (`generate_dataset.py`)
3. Use `packing=True` for efficiency
4. Full fine-tuning (model is only 80M params)
5. Skip custom loss initially - test if standard loss works

### Training Script Outline
```python
from unsloth import FastModel
from transformers import Qwen3ForCausalLM
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import json

# Load model
model, tokenizer = FastModel.from_pretrained(
    model_name="ekwek/Soprano-80M",
    max_seq_length=1024,
    dtype=None,  # auto
    auto_model=Qwen3ForCausalLM,
)

# Apply LoRA (optional - can also do full fine-tuning)
model = FastModel.get_peft_model(
    model,
    r=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)

# Load pre-processed data
def load_soprano_data(path):
    with open(path) as f:
        data = json.load(f)
    return [{"text": f"[STOP][TEXT]{t}[START]{''.join([f'[{x}]' for x in a])}[STOP]"}
            for t, a in data]

train_data = load_soprano_data("dataset/train.json")
val_data = load_soprano_data("dataset/val.json")

# Convert to HF dataset
from datasets import Dataset
train_dataset = Dataset.from_list(train_data)

# Training
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    args=SFTConfig(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        max_seq_length=1024,
        packing=True,
        learning_rate=5e-4,
        warmup_ratio=0.1,
        num_train_epochs=3,
        bf16=True,
        output_dir="outputs",
    ),
)

trainer.train()
model.save_pretrained("fine_tuned_soprano")
```

---

## 6. Expected Benefits

| Aspect | Factory Approach | Unsloth |
|--------|-----------------|---------|
| Training Speed | Baseline | ~2x faster |
| VRAM Usage | ~16GB | ~8-10GB (50% less) |
| Code Complexity | Custom training loop | Standard TRL interface |
| LoRA Support | Not implemented | Built-in |
| Quantized Training | Not supported | 4-bit, 8-bit supported |
| Multi-GPU | Custom DDP | Built-in accelerate |

---

## 7. Potential Risks

1. **Audio Quality**: Uniform loss may affect output quality - requires testing
2. **Tokenizer Compatibility**: Custom tokens need validation
3. **Unsupported Operations**: Some Unsloth optimizations may not apply to small 80M model
4. **Version Compatibility**: Ensure Unsloth version supports Qwen3

---

## 8. Next Steps

1. [ ] Create Unsloth training notebook/script
2. [ ] Validate model loading with FastModel
3. [ ] Test with small dataset (100 samples)
4. [ ] Compare audio quality: factory vs Unsloth
5. [ ] Benchmark training speed and VRAM
6. [ ] If quality differs, implement custom loss function
7. [ ] Scale to full dataset

---

## References

- [Unsloth GitHub](https://github.com/unslothai/unsloth)
- [Unsloth TTS Documentation](https://docs.unsloth.ai/basics/text-to-speech-tts-fine-tuning)
- [Unsloth TTS Blog Post](https://unsloth.ai/blog/tts)
- [Soprano-80M on HuggingFace](https://huggingface.co/ekwek/Soprano-80M)
- [Soprano GitHub](https://github.com/ekwek1/soprano)
