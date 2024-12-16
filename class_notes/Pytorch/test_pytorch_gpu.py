import torch

def test_pytorch_gpu():
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        print("CUDA is not available. PyTorch will run on the CPU.")
        return
    
    # Print CUDA and GPU details
    print("CUDA is available.")
    device_count = torch.cuda.device_count()
    print(f"Number of GPUs available: {device_count}")
    for i in range(device_count):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Test tensor computation on GPU
    device = torch.device("cuda" if cuda_available else "cpu")
    try:
        print("\nTesting tensor operations on GPU...")
        x = torch.rand(1000, 1000).to(device)
        y = torch.rand(1000, 1000).to(device)
        result = torch.matmul(x, y)  # Matrix multiplication
        print("Tensor operation successful on GPU.")
    except Exception as e:
        print(f"Tensor operation failed on GPU: {e}")
    else:
        print("GPU is working correctly for PyTorch.")

if __name__ == "__main__":
    test_pytorch_gpu()
