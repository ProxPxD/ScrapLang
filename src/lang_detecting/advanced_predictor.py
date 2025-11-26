try:
    import torch
    has_torch = True
except ImportError:
    has_torch = False


print(f'Torch is {("un", "")[has_torch]}available')
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device count:", torch.cuda.device_count())
    print("Current device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))


class AdvancedDetector:
    ...
