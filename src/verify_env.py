import torch

print("PyTorch:", torch.__version__, "| CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("GPU:", p.name, f"| VRAM: {p.total_memory/1024**2:.0f} MB")
    print("Compiled for:", torch.cuda.get_arch_list())
    x = torch.randn(1000, 1000, device="cuda")
    print("GPU math works:", (x @ x).sum().item())
    