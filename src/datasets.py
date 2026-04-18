from torch.utils.data import Dataset
import torch


class ICSD_MelSpectogram(Dataset):
    def __init__(self, X, y):
        # Convert numpy arrays to float tensors
        # Audio data is usually float32; Labels are usually Long (integers)
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

        # Audio CNNs expect a 'channel' dimension: (Batch, Channels, Height, Width)
        # If your X is (Samples, H, W), we need to add a 1 for the channel
        if self.X.ndimension() == 3:
            self.X = self.X.unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
