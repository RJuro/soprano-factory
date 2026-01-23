#!/usr/bin/env python3
"""
Simple inference script that works directly with transformers + decoder.
Use this if the soprano-tts package doesn't work with your fine-tuned model.

Usage:
    python inference_simple.py --model outputs/soprano-danish_step500 --text "Hej verden"
    python inference_simple.py --model outputs/soprano-danish_step500 --text "Rød grød" -o output.mp3

Requirements:
    pip install torch transformers vocos soundfile pydub huggingface_hub
"""

import argparse
import os
import re
import sys
import torch
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Simple Soprano TTS Inference")
    parser.add_argument("--model", "-m", type=str, required=True,
                        help="Path to fine-tuned model directory")
    parser.add_argument("--text", "-t", type=str, required=True,
                        help="Text to synthesize")
    parser.add_argument("--output", "-o", type=str, default="output.wav",
                        help="Output file (.wav or .mp3)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=1500,
                        help="Max audio tokens to generate")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def load_decoder(model_path, device):
    """Load Vocos decoder from model directory or download from base model."""
    from vocos import Vocos
    from huggingface_hub import hf_hub_download

    # Try local decoder first
    local_decoder = os.path.join(model_path, "decoder.pth")
    if os.path.exists(local_decoder):
        print(f"Loading decoder from {local_decoder}")
        decoder = torch.load(local_decoder, map_location=device)
    else:
        # Download from base model
        print("Downloading decoder from ekwek/Soprano-80M...")
        decoder_path = hf_hub_download("ekwek/Soprano-80M", "decoder.pth")
        decoder = torch.load(decoder_path, map_location=device)

    return decoder


def extract_audio_tokens(text, tokenizer):
    """Extract audio token IDs from generated text."""
    # Audio tokens are in range [3, 8003]
    tokens = tokenizer.encode(text, add_special_tokens=False)
    audio_tokens = [t for t in tokens if 3 <= t <= 8003]
    return audio_tokens


def tokens_to_codes(tokens):
    """Convert token IDs to FSQ codes (0-7999 range)."""
    # Subtract 3 to get back to codec range
    codes = [t - 3 for t in tokens]
    return codes


def decode_with_vocos(decoder, codes, device):
    """Decode FSQ codes to audio waveform."""
    # This requires the Vocos model structure
    # The decoder expects codes in a specific format
    codes_tensor = torch.tensor(codes, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        # Vocos decodes from codes to audio
        audio = decoder.decode(codes_tensor)

    return audio.cpu().numpy().squeeze()


def main():
    args = parse_args()

    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load model and tokenizer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model from {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.to(device).to(torch.bfloat16)
    model.eval()

    # Prepare prompt
    # Soprano format: [STOP][TEXT]<text>[START]<audio tokens>[STOP]
    prompt = f"[STOP][TEXT]{args.text}[START]"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    print(f"Generating audio for: {args.text}")

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=1.2,
            eos_token_id=tokenizer.encode('[STOP]')[0],
            pad_token_id=tokenizer.pad_token_id or 0,
        )

    # Decode output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)

    # Extract audio tokens (IDs between 3 and 8003)
    generated_ids = outputs[0].tolist()
    audio_token_ids = [t for t in generated_ids if 3 <= t <= 8003]

    print(f"Generated {len(audio_token_ids)} audio tokens")

    if len(audio_token_ids) == 0:
        print("Error: No audio tokens generated!")
        print("Generated output:", generated_text[:200])
        sys.exit(1)

    # Try to decode using soprano-tts if available
    try:
        from soprano.decoder import Decoder
        from huggingface_hub import hf_hub_download

        # Download decoder if needed
        decoder_path = hf_hub_download("ekwek/Soprano-80M", "decoder.pth")

        print("Decoding audio...")
        decoder = Decoder(decoder_path, device=str(device))

        # Convert token IDs to codes (subtract 3)
        codes = torch.tensor([[t - 3 for t in audio_token_ids]], dtype=torch.long, device=device)

        with torch.no_grad():
            audio = decoder(codes)

        audio = audio.cpu().numpy().squeeze()

    except ImportError:
        print("\nNote: soprano-tts not installed. Cannot decode audio tokens to waveform.")
        print("Install with: pip install soprano-tts")
        print("\nGenerated audio token IDs saved to output.txt")
        with open("output_tokens.txt", "w") as f:
            f.write(f"Text: {args.text}\n")
            f.write(f"Audio tokens ({len(audio_token_ids)}): {audio_token_ids}\n")
        sys.exit(1)

    # Save audio
    import soundfile as sf

    # Normalize
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio)) * 0.95

    wav_path = args.output.replace('.mp3', '.wav') if args.output.endswith('.mp3') else args.output
    sf.write(wav_path, audio, 32000)
    print(f"Saved: {wav_path}")

    # Convert to MP3 if needed
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
