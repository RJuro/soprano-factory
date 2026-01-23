#!/usr/bin/env python3
"""
Inference script for fine-tuned Soprano TTS models.

Usage:
    # Basic usage
    python inference.py --model outputs/soprano-danish_step500 --text "Hej, mit navn er Soprano."

    # Save as MP3
    python inference.py --model outputs/soprano-danish_step500 --text "Rød grød med fløde." -o output.mp3

    # Use base model
    python inference.py --model ekwek/Soprano-80M --text "Hello world"

    # Batch from file
    python inference.py --model outputs/soprano-danish_step500 --file texts.txt --output-dir outputs/

Requirements:
    pip install soprano-tts soundfile pydub
    # For MP3: apt-get install ffmpeg (or brew install ffmpeg on macOS)
"""

import argparse
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Soprano TTS Inference")
    parser.add_argument("--model", "-m", type=str, required=True,
                        help="Path to fine-tuned model or HuggingFace model ID (e.g., ekwek/Soprano-80M)")

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--text", "-t", type=str,
                             help="Text to synthesize")
    input_group.add_argument("--file", "-f", type=str,
                             help="File with texts (one per line)")

    # Output options
    parser.add_argument("--output", "-o", type=str, default="output.wav",
                        help="Output file path (.wav or .mp3)")
    parser.add_argument("--output-dir", type=str,
                        help="Output directory for batch processing")

    # Generation parameters
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (default: 0.7)")
    parser.add_argument("--top-p", type=float, default=0.95,
                        help="Top-p sampling (default: 0.95)")
    parser.add_argument("--repetition-penalty", type=float, default=1.2,
                        help="Repetition penalty (default: 1.2)")

    # Device options
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cuda, cpu (default: auto)")

    return parser.parse_args()


def convert_to_mp3(wav_path, mp3_path, bitrate="192k"):
    """Convert WAV to MP3 using pydub/ffmpeg."""
    try:
        from pydub import AudioSegment
        sound = AudioSegment.from_wav(wav_path)
        sound.export(mp3_path, format="mp3", bitrate=bitrate)
        return True
    except ImportError:
        print("Warning: pydub not installed. Run: pip install pydub")
        return False
    except Exception as e:
        print(f"Warning: MP3 conversion failed: {e}")
        print("Make sure ffmpeg is installed: apt-get install ffmpeg")
        return False


def main():
    args = parse_args()

    # Import soprano
    try:
        from soprano import SopranoTTS
    except ImportError:
        print("Error: soprano-tts not installed.")
        print("Install with: pip install soprano-tts")
        sys.exit(1)

    # Load model
    print(f"Loading model: {args.model}")

    # Check if it's a local path or HuggingFace ID
    if os.path.isdir(args.model):
        # Local fine-tuned model - need to copy decoder from base model
        decoder_path = os.path.join(args.model, "decoder.pth")
        if not os.path.exists(decoder_path):
            print("Decoder not found in fine-tuned model, downloading from base model...")
            from huggingface_hub import hf_hub_download
            import shutil
            base_decoder = hf_hub_download("ekwek/Soprano-80M", "decoder.pth")
            shutil.copy(base_decoder, decoder_path)
            print(f"Copied decoder to {decoder_path}")

        model = SopranoTTS(
            model_path=args.model,
            device=args.device,
        )
    else:
        # HuggingFace model ID
        model = SopranoTTS(
            backend=args.model,
            device=args.device,
        )

    print("Model loaded successfully!")

    # Get texts to synthesize
    if args.text:
        texts = [args.text]
    else:
        with open(args.file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(texts)} texts from {args.file}")

    # Generate audio
    for i, text in enumerate(texts):
        print(f"\n[{i+1}/{len(texts)}] Generating: {text[:50]}{'...' if len(text) > 50 else ''}")

        # Determine output path
        if len(texts) == 1:
            output_path = args.output
        else:
            output_dir = args.output_dir or "outputs"
            os.makedirs(output_dir, exist_ok=True)
            ext = os.path.splitext(args.output)[1] or ".wav"
            output_path = os.path.join(output_dir, f"audio_{i:04d}{ext}")

        # Generate
        wav_path = output_path.replace('.mp3', '.wav') if output_path.endswith('.mp3') else output_path

        try:
            audio = model.infer(
                text,
                output_path=wav_path,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
            )
            print(f"  Saved: {wav_path}")

            # Convert to MP3 if requested
            if output_path.endswith('.mp3'):
                if convert_to_mp3(wav_path, output_path):
                    print(f"  Saved: {output_path}")
                    os.remove(wav_path)  # Remove intermediate WAV

        except Exception as e:
            print(f"  Error: {e}")
            continue

    print("\nDone!")


if __name__ == "__main__":
    main()
