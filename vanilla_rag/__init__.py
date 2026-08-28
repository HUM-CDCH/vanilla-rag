import os

# Must be set before torch/pymilvus are imported below, or embedding can
# segfault on macOS from a duplicate OpenMP runtime init.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from .config import DEFAULT_CONFIG, RAGConfig
from .pipeline import RAGPipeline

__all__ = ["DEFAULT_CONFIG", "RAGConfig", "RAGPipeline"]
