"""Loggers: CSV always, TensorBoard when it imports."""
from __future__ import annotations

from pathlib import Path


def build_loggers(out_dir: str | Path, run_name: str) -> list:
    from lightning.pytorch.loggers import CSVLogger
    loggers = [CSVLogger(save_dir=str(out_dir), name=run_name, version="")]
    try:
        import tensorboard  # noqa: F401
        from lightning.pytorch.loggers import TensorBoardLogger
        loggers.append(TensorBoardLogger(save_dir=str(out_dir), name=run_name,
                                         version="", sub_dir="tb"))
    except ImportError:
        pass
    return loggers
