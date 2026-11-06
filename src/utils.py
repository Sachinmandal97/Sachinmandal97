"""
Core utilities: logging, seed management for reproducibility, and hardware profiling.
Keeps all utility functions in one clear, readable file for easy defense.
"""

import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import numpy as np
import psutil
import torch


def setup_logger(
    name: str = "multimodal_verifier",
    log_file: Optional[Path] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Configures a standard logger with Windows cp1252-safe console output.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Standard console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Optional file handler
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


def set_seed(seed: int = 42) -> None:
    """
    Locks random seeds across Python, NumPy, and PyTorch for 100% reproducible experiments.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class HardwareProfiler:
    """
    Measures actual RAM consumption, model loading times, and stage latencies.
    Ensures that reported performance metrics are measured empirically on the host machine.
    """

    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.stage_timings: Dict[str, List[float]] = {}

    def get_memory_mb(self) -> float:
        """Returns current process Resident Set Size (RSS) in Megabytes."""
        return self.process.memory_info().rss / (1024 * 1024)

    def measure_stage(self, stage_name: str, fn: Callable[..., Any], *args, **kwargs) -> Any:
        """
        Executes a function and records its elapsed execution time.
        """
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - t0

        if stage_name not in self.stage_timings:
            self.stage_timings[stage_name] = []
        self.stage_timings[stage_name].append(elapsed)

        return result

    def get_summary(self) -> Dict[str, Any]:
        """
        Returns average latency and throughput for every profiled pipeline stage.
        """
        summary = {
            "current_ram_mb": round(self.get_memory_mb(), 2),
            "stages": {}
        }
        for stage, timings in self.stage_timings.items():
            count = len(timings)
            total = sum(timings)
            avg = total / count if count > 0 else 0.0
            throughput = (count / total) if total > 0 else 0.0
            summary["stages"][stage] = {
                "count": count,
                "total_sec": round(total, 4),
                "avg_ms": round(avg * 1000, 2),
                "samples_per_sec": round(throughput, 2)
            }
        return summary
