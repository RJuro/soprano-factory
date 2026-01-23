"""
Soprano TTS Training with Unsloth

Unsloth provides 2-5x faster training with 50-80% less memory.
This script fine-tunes Soprano (Qwen3-based) for Danish TTS.

Requirements:
    pip install unsloth transformers datasets trl

Usage:
    python train_soprano_unsloth.py --data-dir ../coral_danish_dataset
"""

import argparse
import os
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer

# Check for unsloth availability
try:
    from unsloth import FastLanguageModel, is_bfloat16_supported
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    print("WARNING: Unsloth not available, falling back to standard transformers")
    from transformers import AutoModelForCausalLM, AutoTokenizer


def get_args():
    parser = argparse.ArgumentParser(description="Train Soprano with Unsloth")
    parser.add_argument("--data-dir", type=Path, default=Path("../../coral_danish_dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("./outputs/soprano-unsloth"))
    parser.add_argument("--model-name", default="ekwek/Soprano-1.1-80M")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--full-finetune", action="store_true", help="Full fine-tuning instead of LoRA")
    parser.add_argument("--use-4bit", action="store_true", help="Use 4-bit quantization (QLoRA)")
    return parser.parse_args()


def load_model_unsloth(args):
    """Load model with Unsloth optimizations."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,  # Auto-detect
        load_in_4bit=args.use_4bit,
    )

    if not args.full_finetune:
        # Apply LoRA adapters
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_r,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha=args.lora_alpha,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",  # Unsloth optimized
            random_state=42,
        )

    return model, tokenizer


def load_model_standard(args):
    """Fallback to standard transformers loading."""
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
    )

    if not args.full_finetune:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    return model, tokenizer


def load_soprano_dataset(data_dir: Path, tokenizer):
    """Load and format the Soprano dataset."""
    # Load from JSON files
    dataset = load_dataset(
        "json",
        data_files={
            "train": str(data_dir / "train.json"),
            "validation": str(data_dir / "val.json"),
        }
    )

    def format_example(example):
        # Soprano uses a specific format for TTS
        # The dataset should already have the proper format with audio tokens
        return {"text": example.get("text", "")}

    dataset = dataset.map(format_example, remove_columns=dataset["train"].column_names)
    return dataset


def main():
    args = get_args()
    print(f"Training Soprano TTS with {'Unsloth' if UNSLOTH_AVAILABLE else 'Standard Transformers'}")
    print(f"  Model: {args.model_name}")
    print(f"  Data: {args.data_dir}")
    print(f"  Output: {args.output_dir}")
    print(f"  LoRA: {not args.full_finetune} (r={args.lora_r})")
    print(f"  4-bit: {args.use_4bit}")

    # Load model
    if UNSLOTH_AVAILABLE:
        model, tokenizer = load_model_unsloth(args)
    else:
        model, tokenizer = load_model_standard(args)

    # Load dataset
    dataset = load_soprano_dataset(args.data_dir, tokenizer)
    print(f"  Train samples: {len(dataset['train'])}")
    print(f"  Val samples: {len(dataset['validation'])}")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=250,
        eval_strategy="steps",
        eval_steps=250,
        bf16=is_bfloat16_supported() if UNSLOTH_AVAILABLE else torch.cuda.is_bf16_supported(),
        fp16=not (is_bfloat16_supported() if UNSLOTH_AVAILABLE else torch.cuda.is_bf16_supported()),
        optim="adamw_8bit",
        weight_decay=0.01,
        seed=42,
        report_to="none",  # or "wandb" for logging
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        args=training_args,
        max_seq_length=args.max_seq_length,
        packing=True,  # Pack multiple samples into one sequence
    )

    # Train
    print("\nStarting training...")
    trainer.train()

    # Save
    print(f"\nSaving model to {args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)

    print("Training complete!")


if __name__ == "__main__":
    main()
