import zipfile
import os
import librosa
import numpy as np
from pathlib import Path
from tqdm import tqdm
import gdown
import zipfile
import matplotlib.pyplot as plt
import glob


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
            # load audio (forced to 16khz for ICSD baseline)
            y, _ = librosa.load(audio_path, sr=sr)

            # compute mel spectrogram
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)

            # convert to log scale (db)
            S_db = librosa.power_to_db(S, ref=np.max)

            # min-max normalize to [0, 1]
            S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8)

            # mirror the folder structure in the output directory
            relative_path = audio_path.relative_to(audio_root)
            target_path = mel_root / relative_path.with_suffix('.npy')
            target_path.parent.mkdir(parents=True, exist_ok=True)

            # save as float32 to save space while keeping precision
            np.save(target_path, S_norm.astype(np.float32))

        except Exception as e:
            print(f"Error skipping {audio_path.name}: {e}")


def download_google_file(zip_file_path, google_file_id):
    if not zip_file_path.exists():
        print(f'{zip_file_path} does not exists, proceeding to download...')
        gdown.download(id=google_file_id, output=str(zip_file_path), quiet=False)
    else:
        print(f'{zip_file_path} already exists.')


def download_and_extract(zip_file_path, google_file_id, data_path):
    """
    This function downloads the audio.zip and metadata.zip files from google drive
    It further then extracts its content to the appropriate locations
    """
    if not data_path.exists():
        download_google_file(zip_file_path, google_file_id)
        print(f'Extracting contents from {zip_file_path} into {data_path}.')
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(data_path)

        print(f"Successfully extracted to {data_path}")
    else:
        print(f'{data_path} already exists.')


def get_durations(base_path: str):
    file_list = get_file_list(base_path, "wav")
    durations = np.empty(file_list.shape)

    for i, file_path in enumerate(file_list):
        y, sr = librosa.load(file_path)
        durations[i] = librosa.get_duration(y=y, sr=sr)
    return file_list, durations


def get_file_list(base_path: str, wildcard: str):
    return np.array(glob.glob(os.path.join(base_path, wildcard), recursive=True))


def count_durations(durations):
    counter = {}
    for d in durations:
        counter[d] = counter.get(d, 0) + 1
    return counter


def plot_durations(durations, title):
    plt.figure(figsize=(10, 6))
    plt.hist(durations, bins=30, color='skyblue', edgecolor='black')
    plt.title(title)
    plt.xlabel("Durations (s)")
    plt.ylabel("Counts")
    plt.grid(axis='y', alpha=0.75)
    plt.show()


def load_mel_spec_to_dataset(base_path: str, mel_dim_size=128, timedim=-1, channels=1):
    file_list = get_file_list(base_path, '**/*.npy')

    maxtimedim = -1
    for fp in file_list:
        data = np.load(fp, mmap_mode='r')
        maxtimedim = max(data.shape[1], maxtimedim)

    if timedim == -1:
        timedim = maxtimedim

    # create numpy dataset
    X = np.empty((len(file_list), channels, mel_dim_size, max(timedim, 0)))

    y = np.empty(len(file_list))

    for i, fp in enumerate(file_list):
        y[i] = 1 if "Infantcry" in fp else 0
        X[i, 0, :, :] = adjust_mel_spectogram_length(np.load(fp), timedim)

    return X, y


def adjust_mel_spectogram_length(spec, target_len):

    # If it's already perfect, just return it
    if spec.shape[1] == target_len:
        return spec
    # PADDING: If it's too short, add zeros to the end
    elif spec.shape[1] < target_len:
        pad_width = target_len - spec.shape[1]
        # np.pad(array, (before, after))
        return np.pad(spec, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)

    # TRIMMING: If it's too long, cut it
    else:
        return spec[:, :target_len]
