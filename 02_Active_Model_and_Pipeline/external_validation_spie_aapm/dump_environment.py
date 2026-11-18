import json, os, platform, sys
from pathlib import Path
import numpy as np
import pydicom
import scipy
import torch

EXT_DIR = Path("/experiment-storage/chaidir/fawad-dsai/CLEAR3D_CT project/02_Active_Model_and_Pipeline/external_validation_spie_aapm")

env_info = {
    "system": {
        "os": platform.platform(),
        "python_version": sys.version,
        "python_implementation": platform.python_implementation(),
        "architecture": platform.architecture(),
        "processor": platform.processor(),
        "node": platform.node()
    },
    "pytorch": {
        "version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device_count": torch.cuda.device_count(),
        "device_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
    },
    "packages": {
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pydicom": pydicom.__version__
    },
    "hardware_allocation": {
        "evaluation_gpu": "cuda:1 (NVIDIA GeForce RTX 2080 Ti)",
        "memory_allocated_mib": 2372,
        "precision": "fp16_autocast_cuda1",
        "cpu_workers": 3
    }
}

out_path = EXT_DIR / "environment_details.json"
with open(out_path, "w") as f:
    json.dump(env_info, f, indent=2)
print(f"Environment details written to {out_path}")
