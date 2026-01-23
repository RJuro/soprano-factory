#!/usr/bin/env python3
"""
Colab-ready inference script for fine-tuned Soprano TTS models.

Copy this entire file into a Colab cell and run it!

Usage in Colab:
    !pip install soprano-tts transformers torch soundfile

    # Then run this script
    %run inference_colab.py
"""

import torch
import numpy as np

# Configuration - EDIT THESE
MODEL_ID = "RJuro/soprano-danish"  # Your fine-tuned model on HuggingFace
TEXT = "Hej, mit navn er Soprano. Jeg taler dansk nu."
OUTPUT_FILE = "output.wav"
TEMPERATURE = 0.7
TOP_P = 0.9

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Method 1: Use soprano-tts with model replacement (RECOMMENDED)
    print("\n=== Method 1: soprano-tts with model replacement ===")
    try:
        from soprano import SopranoTTS
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading fine-tuned model from {MODEL_ID}...")

        # Load fine-tuned model and tokenizer
        finetuned_model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
        ).to(device).eval()

        finetuned_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

        # Initialize soprano (loads base model + decoder)
        print("Initializing soprano-tts...")
        tts = SopranoTTS(device=device)

        # CRITICAL: Replace BOTH model AND tokenizer
        tts.pipeline.model = finetuned_model
        tts.pipeline.tokenizer = finetuned_tokenizer

        print(f"Generating: {TEXT}")
        audio = tts.infer(TEXT, temperature=TEMPERATURE, top_p=TOP_P)

        # Save
        import soundfile as sf
        sf.write(OUTPUT_FILE, audio, 32000)
        print(f"Saved to {OUTPUT_FILE}")

        # Play in Colab
        try:
            from IPython.display import Audio, display
            display(Audio(audio, rate=32000))
        except:
            pass

        return True

    except Exception as e:
        print(f"Method 1 failed: {e}")
        import traceback
        traceback.print_exc()

    # Method 2: Manual generation with soprano decoder
    print("\n=== Method 2: Manual generation ===")
    try:
        from soprano import SopranoTTS
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Load fine-tuned model
        print(f"Loading fine-tuned model from {MODEL_ID}...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
        ).to(device).eval()

        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

        # Load soprano just for the decoder
        print("Loading soprano decoder...")
        base_tts = SopranoTTS(device=device)
        decoder = base_tts.pipeline.decoder

        # Prepare input (Soprano format)
        prompt = f"[STOP][TEXT]{TEXT}[START]"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Get special token IDs
        stop_token_id = tokenizer.encode('[STOP]')[0]
        start_token_id = tokenizer.encode('[START]')[0]

        print(f"Generating tokens...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1500,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                repetition_penalty=1.2,
                eos_token_id=stop_token_id,
                pad_token_id=tokenizer.pad_token_id or 0,
            )

        # Extract audio tokens (IDs 3-8003)
        generated_ids = outputs[0].tolist()
        try:
            start_idx = generated_ids.index(start_token_id) + 1
        except ValueError:
            start_idx = inputs.input_ids.shape[1]

        # Stop token marks end
        try:
            stop_idx = generated_ids[start_idx:].index(stop_token_id) + start_idx
        except ValueError:
            stop_idx = len(generated_ids)

        audio_token_ids = [t for t in generated_ids[start_idx:stop_idx] if 3 <= t <= 8003]
        print(f"Generated {len(audio_token_ids)} audio tokens")

        if len(audio_token_ids) < 10:
            print("Warning: Very few audio tokens generated!")
            return False

        # Get hidden states from fine-tuned model
        audio_input = torch.tensor([audio_token_ids], device=device)
        with torch.no_grad():
            out = model(audio_input, output_hidden_states=True)
            hidden_states = out.hidden_states[-1].float()

        print(f"Hidden states shape: {hidden_states.shape}")

        # Decode to audio
        with torch.no_grad():
            audio = decoder(hidden_states)

        audio = audio.cpu().numpy().squeeze()

        # Normalize
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.95

        # Save
        import soundfile as sf
        sf.write(OUTPUT_FILE, audio, 32000)
        print(f"Saved to {OUTPUT_FILE}")

        # Play in Colab
        try:
            from IPython.display import Audio, display
            display(Audio(audio, rate=32000))
        except:
            pass

        return True

    except Exception as e:
        print(f"Method 2 failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    main()
