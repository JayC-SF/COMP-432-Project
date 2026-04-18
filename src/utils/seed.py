import torch
import numpy as np
import random
import os


def set_seed(seed=42):
    # 1. Python's built-in random module
    random.seed(seed)

    # 2. Environment variable for hash randomization
    os.environ['PYTHONHASHSEED'] = str(seed)

    # 3. NumPy
    np.random.seed(seed)

    # 4. PyTorch CPU
    torch.manual_seed(seed)

    # 5. PyTorch GPU (all of them)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # safe for multi-gpu

    # 6. CuDNN (The "Deterministic" trade-off)
    # This ensures convolution algorithms are consistent
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"✅ Seed set to: {seed}")
