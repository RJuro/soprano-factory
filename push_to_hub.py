#!/usr/bin/env python3
"""
Push a fine-tuned Soprano model to HuggingFace Hub.

Usage:
    python push_to_hub.py --model outputs/soprano-danish-a40-20260123_1228_step500 --repo your-username/soprano-danish

Requirements:
    pip install huggingface_hub transformers
    huggingface-cli login
"""

import argparse
import os
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Push Soprano model to HuggingFace Hub")
    parser.add_argument("--model", "-m", type=str, required=True,
                        help="Path to fine-tuned model directory")
    parser.add_argument("--repo", "-r", type=str, required=True,
                        help="HuggingFace repo ID (e.g., username/soprano-danish)")
    parser.add_argument("--token", "-t", type=str, default=os.environ.get("HF_TOKEN"),
                        help="HuggingFace token (or set HF_TOKEN env var)")
    parser.add_argument("--private", action="store_true",
                        help="Make the repo private")
    parser.add_argument("--include-decoder", action="store_true",
                        help="Include decoder.pth from base model")
    return parser.parse_args()


def main():
    args = parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import HfApi, hf_hub_download, login

    # Login if token provided
    if args.token:
        print("Logging in to HuggingFace...")
        login(token=args.token)
    else:
        print("Warning: No token provided. Set --token or HF_TOKEN env var")

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model path {model_path} does not exist")
        return 1

    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    # Include decoder if requested
    if args.include_decoder:
        decoder_path = model_path / "decoder.pth"
        if not decoder_path.exists():
            print("Downloading decoder from ekwek/Soprano-1.1-80M...")
            base_decoder = hf_hub_download("ekwek/Soprano-1.1-80M", "decoder.pth")
            shutil.copy(base_decoder, decoder_path)
            print(f"Copied decoder to {decoder_path}")

    print(f"Pushing to {args.repo}...")

    # Push model and tokenizer (pass token explicitly to override env var)
    model.push_to_hub(args.repo, private=args.private, token=args.token)
    tokenizer.push_to_hub(args.repo, private=args.private, token=args.token)

    # Push decoder separately if it exists
    decoder_path = model_path / "decoder.pth"
    if decoder_path.exists():
        print("Uploading decoder.pth...")
        api = HfApi(token=args.token)
        api.upload_file(
            path_or_fileobj=str(decoder_path),
            path_in_repo="decoder.pth",
            repo_id=args.repo,
            repo_type="model",
        )

    print(f"\nDone! Model available at: https://huggingface.co/{args.repo}")
    print(f"\nTo use in Colab:")
    print(f'  from soprano import SopranoTTS')
    print(f'  model = SopranoTTS(backend="{args.repo}")')
    print(f'  model.infer("Hej verden", "output.wav")')


if __name__ == "__main__":
    exit(main() or 0)
