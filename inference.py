#!/usr/bin/env python3
"""
Inference script for fine-tuned Soprano TTS models.
Uses transformers directly (not soprano-tts) for container compatibility.

Usage:
    python inference.py --model outputs/soprano-danish_step500 --text "Hej, mit navn er Soprano."
    python inference.py --model outputs/soprano-danish_step500 --text "Rød grød med fløde." -o output.mp3

Requirements:
    pip install torch transformers huggingface_hub soundfile pydub vocos
"""

import argparse
import os
import sys
import torch
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Soprano TTS Inference")
    parser.add_argument("--model", "-m", type=str, required=True,
                        help="Path to fine-tuned model or HuggingFace model ID")
    parser.add_argument("--text", "-t", type=str, required=True,
                        help="Text to synthesize")
    parser.add_argument("--output", "-o", type=str, default="output.wav",
                        help="Output file path (.wav or .mp3)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (default: 0.7)")
    parser.add_argument("--top-p", type=float, default=0.95,
                        help="Top-p sampling (default: 0.95)")
    parser.add_argument("--max-tokens", type=int, default=1500,
                        help="Max audio tokens to generate")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cuda, cpu (default: auto)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Device setup
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import hf_hub_download
    import shutil

    print(f"Loading model from {args.model}...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Load model with eager attention (avoids SDPA enable_gqa issue)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        attn_implementation="eager",  # Avoid SDPA compatibility issues
        torch_dtype=torch.bfloat16,
    )
    model.to(device)
    model.eval()

    # Try using soprano-tts with model replacement (recommended approach)
    print("Attempting to use soprano-tts with fine-tuned model...")
    try:
        from soprano import SopranoTTS

        # Initialize soprano with base model first
        tts = SopranoTTS(device=device)

        # Replace BOTH model and tokenizer with fine-tuned versions
        tts.pipeline.model = model
        tts.pipeline.tokenizer = tokenizer

        print(f"Generating audio for: {args.text}")
        audio = tts.infer(
            args.text,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        use_soprano = True

    except Exception as e:
        print(f"soprano-tts failed: {e}")
        print("Falling back to manual generation...")
        use_soprano = False

        # Manual fallback: generate tokens and use soprano decoder
        # Prepare prompt
        prompt = f"[STOP][TEXT]{args.text}[START]"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Get STOP token
        stop_token_id = tokenizer.encode('[STOP]')[0]

        print(f"Generating audio for: {args.text}")

        # Generate audio tokens
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=1.2,
                eos_token_id=stop_token_id,
                pad_token_id=tokenizer.pad_token_id or 0,
            )

        # Extract audio tokens (IDs 3-8003)
        generated_ids = outputs[0].tolist()
        start_token_id = tokenizer.encode('[START]')[0]
        try:
            start_idx = generated_ids.index(start_token_id) + 1
        except ValueError:
            start_idx = inputs.input_ids.shape[1]

        audio_token_ids = [t for t in generated_ids[start_idx:] if 3 <= t <= 8003]
        print(f"Generated {len(audio_token_ids)} audio tokens")

        if len(audio_token_ids) < 10:
            print("Warning: Very few audio tokens generated.")
            sys.exit(1)

        # Use soprano's decoder directly
        try:
            from soprano import SopranoTTS
            base_tts = SopranoTTS(device=device)
            decoder = base_tts.pipeline.decoder

            # Get hidden states for decoder from fine-tuned model
            audio_input = torch.tensor([audio_token_ids], device=device)
            with torch.no_grad():
                out = model(audio_input, output_hidden_states=True)
                hidden_states = out.hidden_states[-1].float()

            # Decode to audio
            with torch.no_grad():
                audio = decoder(hidden_states)
            audio = audio.cpu().numpy().squeeze()
        except Exception as e2:
            print(f"Decoder failed: {e2}")
            sys.exit(1)

    # Save audio
    if use_soprano:
        # Audio from soprano is already numpy array
        if hasattr(audio, 'cpu'):
            audio = audio.cpu().numpy().squeeze()
    else:
        if hasattr(audio, 'cpu'):
            audio = audio.cpu().numpy().squeeze()

    # Normalize
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio)) * 0.95

    import soundfile as sf
    # Soprano outputs at 32kHz
    sample_rate = 32000
    wav_path = args.output.replace('.mp3', '.wav') if args.output.endswith('.mp3') else args.output
    sf.write(wav_path, audio, sample_rate)
    print(f"Saved: {wav_path}")

    # Convert to MP3 if requested
    if args.output.endswith('.mp3'):
        try:
            from pydub import AudioSegment
            sound = AudioSegment.from_wav(wav_path)
            sound.export(args.output, format="mp3", bitrate="192k")
            os.remove(wav_path)
            print(f"Saved: {args.output}")
        except Exception as e:
            print(f"MP3 conversion failed: {e}")

    print("Done!")


if __name__ == "__main__":
    main()
