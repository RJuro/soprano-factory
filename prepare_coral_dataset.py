"""
Prepare Coral TTS Danish dataset for Soprano training.
Downloads from HuggingFace and converts to LJSpeech format.

Usage:
    python prepare_coral_dataset.py --output-dir coral_danish_dataset
    python prepare_coral_dataset.py --output-dir coral_danish_dataset --max-samples 2000
"""
import argparse
import os

from datasets import load_dataset
import soundfile as sf
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser(description="Prepare Coral TTS dataset")
    parser.add_argument("--output-dir", default="coral_danish_dataset", help="Output directory")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of samples (default: all)")
    parser.add_argument("--dataset", default="CoRal-project/coral-tts", help="HuggingFace dataset name")
    parser.add_argument("--split", default="train", help="Dataset split")
    return parser.parse_args()


def main():
    args = get_args()

    output_dir = args.output_dir
    wavs_dir = os.path.join(output_dir, "wavs")
    os.makedirs(wavs_dir, exist_ok=True)

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    dataset = load_dataset(args.dataset, split=args.split)
    print(f"Full dataset: {len(dataset)} samples")

    # Limit samples if specified
    num_samples = len(dataset)
    if args.max_samples:
        num_samples = min(args.max_samples, len(dataset))
        print(f"Limiting to {num_samples} samples")

    # Convert to LJSpeech format
    print(f"Converting to LJSpeech format in {output_dir}/")
    metadata_lines = []
    skipped = 0

    for i, sample in enumerate(tqdm(dataset.select(range(num_samples)), total=num_samples)):
        # Get audio and text
        audio_array = sample['audio']['array']
        sample_rate = sample['audio']['sampling_rate']
        text = sample.get('text', '')

        if not text or not text.strip():
            skipped += 1
            continue

        # Save audio file
        filename = f"coral_{i:06d}"
        audio_path = os.path.join(wavs_dir, f"{filename}.wav")
        sf.write(audio_path, audio_array, sample_rate)

        # Add to metadata (clean text of pipes)
        clean_text = text.replace('|', ' ').strip()
        metadata_lines.append(f"{filename}|{clean_text}")

    # Write metadata file
    metadata_path = os.path.join(output_dir, "metadata.txt")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(metadata_lines))

    print(f"\nDataset prepared:")
    print(f"  - Samples: {len(metadata_lines)}")
    print(f"  - Skipped: {skipped}")
    print(f"  - metadata.txt: {metadata_path}")
    print(f"  - wavs/: {wavs_dir}")
    print(f"\nNext step: python generate_dataset.py --input-dir {output_dir}")


if __name__ == '__main__':
    main()
