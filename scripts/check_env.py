#!/usr/bin/env python3
"""Is this machine ready? Reports Apple Silicon, Python version and MLX."""
import platform, shutil, sys

def main() -> int:
    ok = True
    machine, system = platform.machine(), platform.system()
    print(f"platform      {system} {platform.mac_ver()[0]} ({machine})")
    if system != "Darwin" or machine != "arm64":
        print("  ! MLX targets Apple Silicon (Darwin/arm64)")
        ok = False
    print(f"python        {sys.version.split()[0]}")
    if sys.version_info < (3, 11):
        print("  ! need Python 3.11+")
        ok = False
    try:
        import mlx.core as mx
        print(f"mlx           {mx.__version__ if hasattr(mx, '__version__') else 'present'}")
        import numpy as np
        a = mx.array(np.arange(6, dtype=np.float32).reshape(2, 3))
        print(f"mlx smoke     sum={float(mx.sum(a)):.1f} (expected 15.0)")
    except Exception as exc:  # pragma: no cover - environment probe
        print(f"mlx           MISSING ({exc.__class__.__name__}: {exc})")
        print("  -> pip install mlx")
        ok = False
    print(f"git           {shutil.which('git') or 'MISSING'}")
    print("READY" if ok else "NOT READY")
    return 0 if ok else 1

if __name__ == "__main__":
    raise SystemExit(main())
