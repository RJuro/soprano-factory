#!/usr/bin/env python3
"""
Inference script for fine-tuned Soprano TTS models.

Usage:
    python inference.py --model outputs/soprano-danish_step500 --text "Hej, mit navn er Soprano."
    python inference.py --model outputs/soprano-danish_step500 --text "Rød grød med fløde." -o output.mp3

Requirements:
    pip install soprano-tts soundfile pydub
"""

import argparse
import os
import sys


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
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cuda, cpu (default: auto)")
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        from soprano import SopranoTTS
    except ImportError:
        print("Error: soprano-tts not installed.")
        print("Install with: pip install soprano-tts")
        sys.exit(1)

    from huggingface_hub import hf_hub_download
    import shutil

    print(f"Loading model: {args.model}")

    # For fine-tuned models, copy decoder from Soprano-1.1-80M (768-dim)
    if os.path.isdir(args.model):
        decoder_path = os.path.join(args.model, "decoder.pth")

        # Always ensure we have the right decoder (delete old wrong one if needed)
        if os.path.exists(decoder_path):
            # Check if it's the wrong decoder (512-dim vs 768-dim)
            import torch
            state = torch.load(decoder_path, map_location='cpu', weights_only=False)
            embed_key = 'decoder.embed.weight' if 'decoder.embed.weight' in state else 'embed.weight'
            if embed_key in state and state[embed_key].shape[0] == 512:
                print("Found old 512-dim decoder, replacing with 768-dim...")
                os.remove(decoder_path)

        if not os.path.exists(decoder_path):
            print("Downloading decoder from ekwek/Soprano-1.1-80M...")
            base_decoder = hf_hub_download("ekwek/Soprano-1.1-80M", "decoder.pth")
            shutil.copy(base_decoder, decoder_path)
            print(f"Copied decoder to {decoder_path}")

        model = SopranoTTS(model_path=args.model, device=args.device)
    else:
        model = SopranoTTS(backend=args.model, device=args.device)

    print("Model loaded!")

    # Generate
    print(f"Generating: {args.text}")
    wav_path = args.output.replace('.mp3', '.wav') if args.output.endswith('.mp3') else args.output

    audio = model.infer(
        args.text,
        output_path=wav_path,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=1.2,
    )
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
