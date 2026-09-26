"""GPU detection and CUDA library bootstrapping.

The CUDA runtime that CTranslate2 needs (cuBLAS, cuDNN) ships inside pip
packages under ``site-packages/nvidia/*/lib``, which is not on the dynamic
loader's search path.  Without help, CTranslate2 fails with

    Library libcublas.so.12 is not found or cannot be loaded

even though the library is installed.  Rather than make the user export
LD_LIBRARY_PATH before every launch, the shared objects are opened here with
RTLD_GLOBAL so they are already resident when CTranslate2 looks for them.
"""

import ctypes
import glob
import os
import site
import sys

# Loaded in dependency order — cuDNN links against cuBLAS.
_LIBRARY_PATTERNS = (
    "libcublasLt.so*",
    "libcublas.so*",
    "libnvrtc.so*",
    "libcudnn*.so*",
)

_bootstrapped: bool | None = None


def _nvidia_lib_dirs() -> list[str]:
    """Every site-packages/nvidia/*/lib directory that exists."""
    roots = list(site.getsitepackages())
    user_site = site.getusersitepackages()
    if isinstance(user_site, str):
        roots.append(user_site)
    roots.append(os.path.dirname(os.path.dirname(sys.executable)))

    dirs = []
    for root in roots:
        dirs.extend(glob.glob(os.path.join(root, "nvidia", "*", "lib")))
    # Deduplicate while keeping order
    return list(dict.fromkeys(d for d in dirs if os.path.isdir(d)))


def preload_cuda_libraries() -> bool:
    """Make the pip-installed CUDA libraries loadable. Safe to call repeatedly."""
    global _bootstrapped
    if _bootstrapped is not None:
        return _bootstrapped

    loaded_any = False
    for lib_dir in _nvidia_lib_dirs():
        for pattern in _LIBRARY_PATTERNS:
            for path in sorted(glob.glob(os.path.join(lib_dir, pattern))):
                try:
                    ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                    loaded_any = True
                except OSError:
                    continue  # a variant we do not need

    _bootstrapped = loaded_any
    return loaded_any


def cuda_available() -> bool:
    """True if CTranslate2 can actually see a usable GPU."""
    try:
        import ctranslate2
    except ImportError:
        return False

    try:
        if ctranslate2.get_cuda_device_count() <= 0:
            return False
    except Exception:
        return False

    preload_cuda_libraries()
    return True


def resolve_device(preference: str = "auto", use_fp16: bool = True) -> tuple[str, str]:
    """Pick the device and compute type to load the model with.

    Returns (device, compute_type) ready to hand to faster-whisper.
    "auto" uses the GPU when one is usable and falls back to CPU otherwise,
    so the app runs on a machine without CUDA without any config change.
    """
    if preference == "cpu":
        return "cpu", "int8"

    if preference in ("auto", "cuda") and cuda_available():
        return "cuda", "float16" if use_fp16 else "float32"

    if preference == "cuda":
        # Asked for explicitly but unavailable — say so rather than
        # silently running 8x slower.
        print("Warning: CUDA requested but not usable; falling back to CPU.")

    return "cpu", "int8"
