def init_device_and_seeds(seed: int = 42):
    import warnings
    warnings.filterwarnings("ignore")

    import torch
    import numpy as np

    if torch.cuda.is_available():
        device = torch.device("cuda")
        try:
            gpu_name = torch.cuda.get_device_name(0)
            print(f"Using CUDA: {gpu_name}")
        except Exception:
            print("Using CUDA")
    else:
        device = torch.device("cpu")
        print("CUDA not available. Using CPU.")

    torch.manual_seed(seed)
    np.random.seed(seed)
    return device


