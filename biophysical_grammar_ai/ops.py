from __future__ import annotations
import os
xp = None; gpu=False; reason=""
try:
    if os.environ.get("BGAI_DISABLE_GPU","0")=="0":
        import cupy as cp; _=cp.arange(1); xp=cp; gpu=True; reason="CuPy"
    else:
        raise ImportError("GPU disabled via env")
except Exception as e:
    import numpy as np; xp=np; gpu=False; reason=f"NumPy (GPU unavailable: {e})"
def to_cpu(a):
    try:
        import cupy as cp
        if isinstance(a, cp.ndarray): return a.get()
    except Exception: pass
    return a
def softmax(x, axis=-1, temperature=1.0):
    x = x / max(temperature,1e-8); x = x - x.max(axis=axis, keepdims=True)
    e = xp.exp(x); return e/(e.sum(axis=axis, keepdims=True)+1e-8)
def randn(*shape, scale=1.0): return xp.random.randn(*shape)*scale
def clip(x, lo, hi): return xp.minimum(xp.maximum(x, lo), hi)
