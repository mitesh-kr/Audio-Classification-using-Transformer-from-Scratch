"""
Script to extract and prepare the dataset.
"""

import os
import zipfile
import argparse
from pathlib import Path
import pandas as pd


def extract_dataset(zip_path, extract_path):
    """
    Extract the dataset zip file.
    
    Args:
        zip_path: Path to the zip file
        extract_path: Path to extract the files to
    """
    print(f"Extracting {zip_path} to {extract_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("Extraction complete.")


def check_dataset(extract_path):
    """
    Check the extracted dataset.
    
    Args:
        extract_path: Path where the dataset was extracted
    """
    meta_path = os.path.join(extract_path, 'meta', 'esc50.csv')
    if not os.path.exists(meta_path):
        print(f"Error: Metadata file not found at {meta_path}")
        return False
    
    # Load the metadata
    df = pd.read_csv(meta_path)
    print(f"Dataset metadata loaded: {len(df)} entries")
    
    # Check for audio files
    audio_path = os.path.join(extract_path, 'audio')
    if not os.path.exists(audio_path):
        print(f"Error: Audio directory not found at {audio_path}")
        return False
    
    # Count audio files
    audio_files = list(Path(audio_path).glob('*.wav'))
    print(f"Found {len(audio_files)} audio files")
    
    # Check for ESC-10 subset
    esc10_count = df[df['esc10'] == True].shape[0]
    print(f"ESC-10 subset contains {esc10_count} entries")
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and prepare the ESC-50 dataset")
    
    parser.add_argument("--zip_path", type=str, required=True,
                       help="Path to the dataset zip file")
    parser.add_argument("--extract_path", type=str, default="./data",
                       help="Path to extract the dataset to")
    
    args = parser.parse_args()
    
    # Create the extraction directory if it doesn't exist
    os.makedirs(args.extract_path, exist_ok=True)
    
    # Extract and check the dataset
    extract_dataset(args.zip_path, args.extract_path)
    if check_dataset(args.extract_path):
        print("Dataset preparation complete.")
    else:
        print("Dataset preparation failed.")
