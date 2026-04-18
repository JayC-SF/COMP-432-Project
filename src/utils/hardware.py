import torch


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")  # For Mac M1/M2/M3 users
    else:
        return torch.device("cpu")
