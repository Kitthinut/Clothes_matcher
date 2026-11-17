"""GPU status check for both backends (check_gpu.py).

Run:  python check_gpu.py
Explains exactly what each framework sees and what to do if TF has no GPU
(TF >= 2.11 cannot use CUDA on native Windows - that is a Google decision,
not a problem with your machine or driver).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def check_torch():
    try:
        import torch
    except ImportError:
        print("PyTorch: not installed")
        return
    if torch.cuda.is_available():
        print(f"PyTorch: CUDA OK -> {torch.cuda.get_device_name(0)} "
              f"(torch {torch.__version__})")
    else:
        print("PyTorch: no CUDA visible (CPU only). Reinstall the CUDA wheel: "
              "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126")


def check_tf():
    try:
        import tensorflow as tf
    except ImportError:
        print("TensorFlow: not installed")
        return
    gpus = tf.config.list_physical_devices("GPU")
    print(f"TensorFlow {tf.__version__}: GPUs = {gpus or 'none (CPU only)'}")
    if not gpus and sys.platform == "win32":
        print("  Note: native Windows TF has no GPU since TF 2.11. Free options:")
        print("  1. WSL2 (recommended): install Ubuntu, then "
              "`pip install 'tensorflow[and-cuda]'` inside WSL - full CUDA.")
        print("  2. Old stack: tensorflow-cpu==2.10 + tensorflow-directml-plugin "
              "(Python 3.10 max, plugin is discontinued).")
        print("  3. Just train TF on CPU (it works, slower) and use PyTorch for GPU.")


if __name__ == "__main__":
    print(f"Python {sys.version.split()[0]} on {sys.platform}")
    check_torch()
    check_tf()
