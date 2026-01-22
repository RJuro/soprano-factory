"""
Converts a dataset in LJSpeech format into audio tokens that can be used to train/fine-tune Soprano.
This script creates two JSON files for train and test splits in the provided directory.

Usage:
python generate_dataset.py --input-dir path/to/files

Args:
--input-dir: Path to directory of LJSpeech-style dataset. If none is provided this defaults to the provided example dataset.
"""
import argparse
import pathlib
import random
import json

import torch
import soundfile as sf
import numpy as np
from scipy import signal
from tqdm import tqdm
from huggingface_hub import hf_hub_download

from encoder.codec import Encoder


SAMPLE_RATE = 32000


def load_audio(filepath):
    """
    Load audio file and return normalized float tensor in range [-1, 1].
    Handles stereo to mono conversion automatically.
    Uses soundfile to avoid torchcodec/FFmpeg issues.
    """
    # Use soundfile instead of torchaudio to avoid torchcodec dependency
    audio, sr = sf.read(filepath, dtype='float32')

    # Convert to torch tensor
    audio = torch.from_numpy(audio)

    # Handle stereo to mono (soundfile returns (samples, channels) for stereo)
    if audio.dim() > 1:
        audio = audio.mean(dim=1)

    return audio, sr


SEED = 42
VAL_PROP = 0.1
VAL_MAX = 512

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir",
        required=False,
        default="./example_dataset",
        type=pathlib.Path
    )
    return parser.parse_args()

def main():
    args = get_args()
    input_dir = args.input_dir

    print("Loading model.")
    encoder = Encoder()
    encoder_path = hf_hub_download(repo_id='ekwek/Soprano-Encoder', filename='encoder.pth')
    encoder.load_state_dict(torch.load(encoder_path))
    print("Model loaded.")


    print("Reading metadata.")
    files = []
    with open(f'{input_dir}/metadata.txt', encoding='utf-8') as f:
        data = f.read().split('\n')
        for line in data:
            line = line.strip()
            if not line or '|' not in line:
                continue
            filename, transcript = line.split('|', maxsplit=1)
            files.append((filename, transcript))
    print(f'{len(files)} samples located in directory.')

    print("Encoding audio.")
    dataset = []
    skipped = []
    for sample in tqdm(files):
        filename, transcript = sample
        audio_path = f'{input_dir}/wavs/{filename}.wav'
        try:
            audio, sr = load_audio(audio_path)
            if sr != SAMPLE_RATE:
                # Resample using scipy instead of torchaudio
                num_samples = int(len(audio) * SAMPLE_RATE / sr)
                audio_np = audio.numpy()
                audio_np = signal.resample(audio_np, num_samples)
                audio = torch.from_numpy(audio_np.astype(np.float32))
            audio = audio.unsqueeze(0)  # Add batch dimension: (samples,) -> (1, samples)
            with torch.no_grad():
                audio_tokens = encoder(audio)
            dataset.append([transcript, audio_tokens.squeeze(0).tolist()])
        except Exception as e:
            skipped.append((filename, str(e)))
            continue

    if skipped:
        print(f"Warning: Skipped {len(skipped)} files due to errors:")
        for filename, error in skipped[:10]:  # Show first 10
            print(f"  - {filename}: {error}")
        if len(skipped) > 10:
            print(f"  ... and {len(skipped) - 10} more")

    print("Generating train/test splits.")
    random.seed(SEED)
    random.shuffle(dataset)
    num_val = min(int(VAL_PROP * len(dataset)) + 1, VAL_MAX)
    train_dataset = dataset[num_val:]
    val_dataset = dataset[:num_val]
    print(f'# train samples: {len(train_dataset)}')
    print(f'# val samples: {len(val_dataset)}')

    print("Saving datasets.")
    with open(f'{input_dir}/train.json', 'w', encoding='utf-8') as f:
        json.dump(train_dataset, f, indent=2)
    with open(f'{input_dir}/val.json', 'w', encoding='utf-8') as f:
        json.dump(val_dataset, f, indent=2)
    print("Datasets saved.")


if __name__ == '__main__':
    main()
