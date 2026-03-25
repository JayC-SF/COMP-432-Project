import torch
import sys

print("-" * 30)
print(f"Python Version: {sys.version}")
print(f"PyTorch Version: {torch.__version__}")
print("-" * 30)

# 1. Check if CUDA (GPU support) is even installed in this environment
cuda_available = torch.cuda.is_available()
print(f"Is CUDA available? {cuda_available}")

if cuda_available:
    # 2. Get GPU Details
    device_count = torch.cuda.device_count()
    current_device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(current_device)
    
    print(f"Total GPUs found: {device_count}")
    print(f"Current Device ID: {current_device}")
    print(f"GPU Model: {device_name}")
    
    # 3. Test a simple math operation on the GPU
    try:
        x = torch.tensor([1.0, 2.0]).to("cuda")
        y = x * 2
        print(f"Successfully ran a calculation on your {device_name}!")
    except Exception as e:
        print(f"Error running GPU calculation: {e}")
else:
    print("RESULT: PyTorch is running on your CPU ONLY.")
    print("FIX: You likely installed the CPU version of Torch.")

print("-" * 30)