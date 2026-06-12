"""Safe inference device selection with CUDA fallback."""

from __future__ import annotations

from utils.logger import get_logger

log = get_logger("inference_device")


def resolve_device(requested: str | None = None, *, allow_cuda: bool = True) -> str:
    """Return a usable torch/ultralytics device string, falling back to CPU."""
    choice = (requested or "auto").lower().strip()
    if choice in {"", "auto"}:
        choice = "cuda" if allow_cuda else "cpu"
    if choice.startswith("cuda") and allow_cuda:
        try:
            import torch

            if torch.cuda.is_available():
                # Smoke tensor to catch driver/runtime errors early
                torch.zeros(1, device="cuda")
                return choice if choice != "auto" else "cuda:0"
        except Exception as exc:
            log.warning("CUDA unavailable ({}); using CPU", exc)
    return "cpu"


def run_with_cpu_fallback(fn, device: str):
    """Run callable(device) and retry on CPU if CUDA fails."""
    try:
        return fn(device)
    except Exception as exc:
        if device == "cpu":
            raise
        log.warning("GPU inference failed ({}); retrying on CPU", exc)
        return fn("cpu")
