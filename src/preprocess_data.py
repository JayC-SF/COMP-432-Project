import zipfile
import os
import librosa
import numpy as np
from pathlib import Path
from tqdm import tqdm
import gdown
import zipfile


def convert_audio_dataset_to_mel(audio_root, mel_root, sr=16000, n_mels=128, hop_length=512):
    """
    This function converts an audio dataset to mel spectrograms while 
    preserving the same folder structure.

    Args:
        audio_root (str): base root path of the audio dataset.
        mel_root (str): base root path of the mel spectogram dataset destination
        sr (int, optional): number > 0 [scalar] sampling rate of `y`. Defaults to 16000.
        n_mels (int, optional): int > 0 [scalar] number of Mel bands to generate. Defaults to 128.
        hop_length (int, optional): int > 0 [scalar] number of samples between successive frames. See `librosa.stft`. Defaults to 512.

    """
    # Convert strings to Path objects
    audio_root = Path(audio_root)
    mel_root = Path(mel_root)

    if mel_root.exists():
        print(f"{mel_root} already exists, skipping convertion...")
        return

    # Find all audio files (wav, mp3, etc.)
    extensions = ['.wav', '.mp3', '.flac']
    audio_files = [f for f in audio_root.rglob('*') if f.suffix.lower() in extensions]

    print(f"Found {len(audio_files)} files. Converting to .npy...")

    for audio_path in tqdm(audio_files):
        try:
            # 1. Load audio (forced to 16kHz for ICSD baseline)
            y, _ = librosa.load(audio_path, sr=sr)

            # 2. Compute Mel Spectrogram
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)

            # 3. Convert to Log Scale (dB)
            S_db = librosa.power_to_db(S, ref=np.max)

            # 4. Min-Max Normalize to [0, 1]
            S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8)

            # 5. Mirror the folder structure in the output directory
            relative_path = audio_path.relative_to(audio_root)
            target_path = mel_root / relative_path.with_suffix('.npy')
            target_path.parent.mkdir(parents=True, exist_ok=True)

            # 6. Save as float32 to save space while keeping precision
            np.save(target_path, S_norm.astype(np.float32))

        except Exception as e:
            print(f"Error skipping {audio_path.name}: {e}")


def download_and_extract(zip_file_path, google_file_id, data_path):
    """
    This function downloads the audio.zip and metadata.zip files from google drive
    It further then extracts its content to the appropriate locations 
    """
    if not data_path.exists():
        if not zip_file_path.exists():
            print(f'{zip_file_path} does not exists, proceeding to download...')
            gdown.download(id=google_file_id, output=str(zip_file_path), quiet=False)
        else:
            print(f'{zip_file_path} already exists.')

        print(f'Extracting contents from {zip_file_path} into {data_path}.')
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(data_path)

        print(f"Successfully extracted to {data_path}")
    else:
        print(f'{data_path} already exists.')
