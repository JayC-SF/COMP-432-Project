from pathlib import Path
from dotenv import load_dotenv
import os

try:
    from google.colab import userdata, drive
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# take sensitive data from dotenv or colab secrets
if IN_COLAB:
    AUDIO_ZIP_GID = userdata.get('AUDIO_ZIP_GID')
    METADATA_ZIP_GID = userdata.get('METADATA_ZIP_GID')
    MEL_SPECTOGRAM_ZIP_GID = userdata.get('MEL_SPECTOGRAM_ZIP_GID')
    MEL_SPECTOGRAM_NPZ_GID = userdata.get('MEL_SPECTOGRAM_NPZ_GID')
    drive.mount('/content/drive/')
    RUNS_PATH = Path('/content/drive/MyDrive/')/userdata.get('DRIVE_FOLDER')/'runs'

else:
    load_dotenv()
    AUDIO_ZIP_GID = os.getenv('AUDIO_ZIP_GID')
    METADATA_ZIP_GID = os.getenv('METADATA_ZIP_GID')
    MEL_SPECTOGRAM_ZIP_GID = os.getenv('MEL_SPECTOGRAM_ZIP_GID')
    MEL_SPECTOGRAM_NPZ_GID = os.getenv('MEL_SPECTOGRAM_NPZ_GID')
    RUNS_PATH = Path('runs')

DATA_PATH = Path('data')

AUDIO_DATA_PATH = DATA_PATH/'audio'
AUDIO_ZIP_FILE_PATH = DATA_PATH/'audio.zip'

METADATA_PATH = DATA_PATH/'metadata'
METADATA_ZIP_FILE_PATH = DATA_PATH/'metadata.zip'

MEL_SPECTOGRAM_PATH = DATA_PATH/'mel_spectogram'
MEL_SPECTOGRAM_ZIP_FILE_PATH = DATA_PATH/'mel_spectogram.zip'
MEL_SPECTOGRAM_NPZ_FILE_PATH = DATA_PATH/'mel_spectogram_audio_length_adjusted.npz'

SEED = 42

CLASSES = ['Snoring', 'InfantCry']
