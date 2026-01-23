# Research: TTS Training with Unsloth

## Executive Summary

This document analyzes alternatives to the current "factory" training approach for TTS fine-tuning using **Unsloth**.

**Recommendation**: Use **Orpheus-TTS 3B** instead of Soprano-80M. Orpheus is:
- Llama-3B based (native Unsloth support with optimized kernels)
- Well-documented fine-tuning workflow with SNAC audio codec
- Much more capable (3B vs 80M parameters)
- Active community with many successful fine-tunes
- Supports emotion tags, zero-shot voice cloning

---

# PART 1: RECOMMENDED APPROACH - Orpheus-TTS 3B

## Why Orpheus over Soprano?

| Aspect | Soprano-80M | Orpheus-TTS 3B |
|--------|-------------|----------------|
| Parameters | 80M | 3B |
| Architecture | Qwen3 (less common) | Llama-3B (native Unsloth) |
| Audio Codec | Custom Vocos | SNAC (well-documented) |
| Unsloth Support | Requires workarounds | First-class support |
| Community | Limited | Active, many fine-tunes |
| Quality | Lightweight/fast | Human-like, emotional |
| Voice Cloning | No | Zero-shot supported |

## Orpheus-TTS Overview

- **Base Model**: `Meta-Llama/Llama-3.2-3B-Instruct`
- **Audio Codec**: SNAC at 24kHz
- **Latency**: ~200ms streaming (reducible to ~100ms)
- **Emotion Tags**: `<laugh>`, `<sigh>`, `<gasp>`, etc.
- **HuggingFace**: `unsloth/orpheus-3b-0.1-ft`

## Data Format for Orpheus

### Dataset Structure
```python
# HuggingFace dataset with columns:
{
    "text": "Your transcription text here",
    "audio": {
        "array": [...],  # Audio waveform
        "sampling_rate": 24000
    }
}
```

### Audio Tokenization with SNAC
```python
from snac import SNAC
import torchaudio.transforms as T

snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to("cuda")

def tokenise_audio(waveform, source_sample_rate):
    waveform = torch.from_numpy(waveform).unsqueeze(0).float()
    resample = T.Resample(source_sample_rate, 24000)
    waveform = resample(waveform).unsqueeze(0).to("cuda")

    with torch.inference_mode():
        codes = snac_model.encode(waveform)

    # Interleave 7 code streams (Orpheus format)
    all_codes = []
    for i in range(codes[0].shape[1]):
        all_codes.extend([
            codes[0][0][i].item() + 128266,
            codes[1][0][2*i].item() + 128266 + 4096,
            codes[2][0][4*i].item() + 128266 + (2*4096),
            codes[2][0][(4*i)+1].item() + 128266 + (3*4096),
            codes[1][0][(2*i)+1].item() + 128266 + (4*4096),
            codes[2][0][(4*i)+2].item() + 128266 + (5*4096),
            codes[2][0][(4*i)+3].item() + 128266 + (6*4096)
        ])
    return all_codes
```

### Special Tokens
```python
start_of_text = 128000
end_of_text = 128009
start_of_speech = 128257
end_of_speech = 128258
start_of_human = 128259
end_of_human = 128260
start_of_ai = 128261
end_of_ai = 128262
pad_token = 128263
```

### Input Format Assembly
```python
def create_input_ids(text, audio_codes, tokenizer, source=None):
    text_prompt = f"{source}: {text}" if source else text
    text_ids = tokenizer.encode(text_prompt, add_special_tokens=True)
    text_ids.append(end_of_text)

    input_ids = (
        [start_of_human] + text_ids + [end_of_human] +
        [start_of_ai, start_of_speech] + audio_codes +
        [end_of_speech, end_of_ai]
    )
    return {
        "input_ids": input_ids,
        "labels": input_ids,
        "attention_mask": [1] * len(input_ids)
    }
```

## Complete Training Script for Orpheus

```python
"""
train_orpheus_danish.py - Fine-tune Orpheus-TTS on Coral Danish dataset
"""
from unsloth import FastLanguageModel
from transformers import Trainer, TrainingArguments
from datasets import load_dataset, Dataset
from snac import SNAC
import torchaudio.transforms as T
import torch

# ============== Configuration ==============
MODEL_NAME = "unsloth/orpheus-3b-0.1-ft"
MAX_SEQ_LENGTH = 2048
LORA_R = 64
BATCH_SIZE = 1
GRAD_ACCUM = 4
MAX_STEPS = 500
LEARNING_RATE = 2e-4

# ============== Load Model ==============
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=False,  # 16-bit for quality
)

# Apply LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=LORA_R,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# ============== Load SNAC Codec ==============
snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to("cuda")

# ============== Special Tokens ==============
START_OF_HUMAN = 128259
END_OF_HUMAN = 128260
START_OF_AI = 128261
END_OF_AI = 128262
START_OF_SPEECH = 128257
END_OF_SPEECH = 128258
END_OF_TEXT = 128009

# ============== Audio Tokenization ==============
def tokenize_audio(waveform, source_sr):
    """Convert audio waveform to SNAC tokens"""
    waveform = torch.from_numpy(waveform).unsqueeze(0).float()
    if source_sr != 24000:
        resample = T.Resample(source_sr, 24000)
        waveform = resample(waveform)
    waveform = waveform.unsqueeze(0).to("cuda")

    with torch.inference_mode():
        codes = snac_model.encode(waveform)

    all_codes = []
    for i in range(codes[0].shape[1]):
        all_codes.extend([
            codes[0][0][i].item() + 128266,
            codes[1][0][2*i].item() + 128266 + 4096,
            codes[2][0][4*i].item() + 128266 + 2*4096,
            codes[2][0][4*i+1].item() + 128266 + 3*4096,
            codes[1][0][2*i+1].item() + 128266 + 4*4096,
            codes[2][0][4*i+2].item() + 128266 + 5*4096,
            codes[2][0][4*i+3].item() + 128266 + 6*4096,
        ])
    return all_codes

# ============== Dataset Preparation ==============
def prepare_example(example):
    """Convert Coral TTS example to Orpheus format"""
    text = example["text"]
    audio = example["audio"]

    # Tokenize audio with SNAC
    audio_codes = tokenize_audio(audio["array"], audio["sampling_rate"])

    # Tokenize text
    text_ids = tokenizer.encode(text, add_special_tokens=True)
    text_ids.append(END_OF_TEXT)

    # Assemble input sequence
    input_ids = (
        [START_OF_HUMAN] + text_ids + [END_OF_HUMAN] +
        [START_OF_AI, START_OF_SPEECH] + audio_codes +
        [END_OF_SPEECH, END_OF_AI]
    )

    # Truncate if too long
    if len(input_ids) > MAX_SEQ_LENGTH:
        return None

    return {
        "input_ids": input_ids,
        "labels": input_ids,
        "attention_mask": [1] * len(input_ids),
    }

# Load Coral TTS Danish
print("Loading Coral TTS Danish dataset...")
dataset = load_dataset("alexandrainst/coral-tts", split="train")

# Process dataset
print("Processing dataset with SNAC tokenization...")
processed = []
for i, example in enumerate(dataset):
    if i >= 2000:  # Limit for testing
        break
    result = prepare_example(example)
    if result:
        processed.append(result)
    if i % 100 == 0:
        print(f"Processed {i}/{min(len(dataset), 2000)}")

train_dataset = Dataset.from_list(processed)
print(f"Final dataset size: {len(train_dataset)}")

# ============== Training ==============
trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    args=TrainingArguments(
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        warmup_steps=10,
        max_steps=MAX_STEPS,
        learning_rate=LEARNING_RATE,
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs_orpheus_danish",
        save_steps=100,
        bf16=True,
        report_to="none",
    ),
)

print("Starting training...")
trainer.train()

# Save model
model.save_pretrained("orpheus-danish-lora")
tokenizer.save_pretrained("orpheus-danish-lora")
print("Training complete!")
```

## Converting Existing Coral Dataset

If you already have the Coral TTS dataset prepared in LJSpeech format:

```python
"""
convert_ljspeech_to_orpheus.py - Convert existing dataset to Orpheus format
"""
import os
import json
import torchaudio
from pathlib import Path
from datasets import Dataset
from snac import SNAC
import torch
import torchaudio.transforms as T

DATASET_PATH = "coral_danish_dataset"  # Your LJSpeech-format dataset
OUTPUT_PATH = "coral_danish_orpheus.json"

snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to("cuda")

def tokenize_audio_file(audio_path):
    waveform, sr = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != 24000:
        resample = T.Resample(sr, 24000)
        waveform = resample(waveform)

    waveform = waveform.unsqueeze(0).to("cuda")

    with torch.inference_mode():
        codes = snac_model.encode(waveform)

    all_codes = []
    for i in range(codes[0].shape[1]):
        all_codes.extend([
            codes[0][0][i].item() + 128266,
            codes[1][0][2*i].item() + 128266 + 4096,
            codes[2][0][4*i].item() + 128266 + 2*4096,
            codes[2][0][4*i+1].item() + 128266 + 3*4096,
            codes[1][0][2*i+1].item() + 128266 + 4*4096,
            codes[2][0][4*i+2].item() + 128266 + 5*4096,
            codes[2][0][4*i+3].item() + 128266 + 6*4096,
        ])
    return all_codes

# Read metadata
metadata_path = Path(DATASET_PATH) / "metadata.txt"
wavs_dir = Path(DATASET_PATH) / "wavs"

processed = []
with open(metadata_path) as f:
    for line in f:
        parts = line.strip().split("|")
        file_id = parts[0]
        text = parts[1] if len(parts) > 1 else parts[0]

        audio_path = wavs_dir / f"{file_id}.wav"
        if audio_path.exists():
            try:
                codes = tokenize_audio_file(str(audio_path))
                processed.append({"text": text, "audio_codes": codes})
            except Exception as e:
                print(f"Error processing {file_id}: {e}")

with open(OUTPUT_PATH, "w") as f:
    json.dump(processed, f)

print(f"Converted {len(processed)} samples to Orpheus format")
```

## Requirements

```
# requirements_orpheus.txt
unsloth
torch>=2.0
torchaudio
transformers>=4.40
datasets
snac
accelerate
bitsandbytes
```

## VRAM Requirements

| Configuration | VRAM Required |
|--------------|---------------|
| LoRA 16-bit | ~12-16 GB |
| LoRA 8-bit | ~8-10 GB |
| QLoRA 4-bit | ~6-8 GB |
| Full fine-tuning | ~24+ GB |

---

# PART 2: ALTERNATIVE - Soprano with Unsloth (Not Recommended)

The following section documents the original analysis of using Soprano with Unsloth, kept for reference.

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

### Orpheus-TTS (Recommended)
- [Orpheus-TTS GitHub](https://github.com/canopyai/Orpheus-TTS)
- [unsloth/orpheus-3b-0.1-ft on HuggingFace](https://huggingface.co/unsloth/orpheus-3b-0.1-ft)
- [Unsloth Orpheus Colab Notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Orpheus_(3B)-TTS.ipynb)
- [Sample Dataset Format](https://huggingface.co/datasets/canopylabs/zac-sample-dataset)
- [Hypa Orpheus Fine-tune Example](https://huggingface.co/hypaai/Hypa_Orpheus-3b-0.1-ft-unsloth-merged_16bit)

### Unsloth
- [Unsloth GitHub](https://github.com/unslothai/unsloth)
- [Unsloth TTS Documentation](https://docs.unsloth.ai/basics/text-to-speech-tts-fine-tuning)
- [Unsloth TTS Blog Post](https://unsloth.ai/blog/tts)

### Soprano (Original - Not Recommended)
- [Soprano-80M on HuggingFace](https://huggingface.co/ekwek/Soprano-80M)
- [Soprano GitHub](https://github.com/ekwek1/soprano)
