import sys

def check_environment():
    print("==========================================")
    print("ImageStudio Environment Diagnostic")
    print("==========================================\n")
    
    print(f"Python Version: {sys.version.split()[0]}\n")
    
    # List of core dependencies to check
    packages = [
        "torch", "torchvision", "diffusers", "transformers", 
        "accelerate", "huggingface_hub", "safetensors", "gradio"
    ]
    
    missing = []
    
    print("--- Package Check ---")
    for pkg in packages:
        try:
            module = __import__(pkg)
            version = getattr(module, "__version__", "Unknown")
            print(f"[OK] {pkg:<15} v{version}")
        except ImportError:
            print(f"[FAILED] {pkg:<11} NOT INSTALLED")
            missing.append(pkg)
            
    print("\n--- Hardware Check ---")
    if "torch" not in missing:
        import torch
        if torch.cuda.is_available():
            print(f"[OK] CUDA Available! Device: {torch.cuda.get_device_name(0)}")
            print(f"     VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            print("[WARNING] CUDA is NOT available. PyTorch cannot see your GPU.")
    else:
        print("[FAILED] Cannot check hardware. PyTorch is missing.")
        
    print("\n==========================================")
    if missing:
        print(f"Status: OFFLINE. Missing packages: {', '.join(missing)}")
    else:
        print("Status: ALL SYSTEMS GO.")
    print("==========================================")

if __name__ == "__main__":
    check_environment()