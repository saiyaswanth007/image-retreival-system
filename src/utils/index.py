"""
Utility: FAISS Retrieval Index Wrapper
=======================================
Task 7: FAISS Index Wrapper

Provides strict encapsulation over faiss.IndexFlatL2 for exact Euclidean distance
metric searches. Enforces C-contiguous float32 typing and allows deterministic
persistence to align with the project design philosophy.
"""

from pathlib import Path
import numpy as np

try:
    import faiss
except ImportError:
    raise ImportError("[FAIL] FAISS is not installed. Run `pip install faiss-cpu` or `faiss-gpu`.")

class RetrievalIndex:
    def __init__(self, feature_dim: int, use_gpu: bool = True):
        """
        Initializes an exact L2 distance index.
        Args:
            feature_dim: The dimensionality of the feature vectors.
            use_gpu: If True, attempts to use FAISS GPU indices.
        """
        self.feature_dim = feature_dim
        
        # We always use exact L2 distance to mathematically align with TripletMarginLoss
        self._cpu_index = faiss.IndexFlatL2(feature_dim)
        
        self.use_gpu = use_gpu and faiss.get_num_gpus() > 0
        if self.use_gpu:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self._cpu_index)
        else:
            self.index = self._cpu_index

    def _validate_data(self, vectors: np.ndarray) -> np.ndarray:
        """
        Enforce strict type validation:
        FAISS aggressively requires C-contiguous float32 arrays.
        """
        assert isinstance(vectors, np.ndarray), "[FAIL] Input must be a numpy array."
        assert vectors.ndim == 2, f"[FAIL] Expected 2D array, got {vectors.ndim}D."
        assert vectors.shape[1] == self.feature_dim, (
            f"[FAIL] Dimension mismatch. Index expects {self.feature_dim}, got {vectors.shape[1]}"
        )
        
        # Ensure strict typing required by C++ backend
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        
        if not vectors.flags.c_contiguous:
            vectors = np.ascontiguousarray(vectors)
            
        return vectors

    def add(self, vectors: np.ndarray):
        """Adds validated vectors to the index."""
        vectors = self._validate_data(vectors)
        self.index.add(vectors)

    def search(self, query_vectors: np.ndarray, k: int = 5):
        """
        Searches the index for the top-k nearest neighbors.
        Returns:
            distances: (N, K) array of L2 distances.
            indices: (N, K) array of retrieved vector IDs.
        """
        assert self.index.ntotal > 0, "[FAIL] Cannot search an empty index."
        
        query_vectors = self._validate_data(query_vectors)
        distances, indices = self.index.search(query_vectors, k)
        return distances, indices

    def save(self, path: Path):
        """Saves the index atomically to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        cpu_index = self.index
        if self.use_gpu:
            # Must map back to CPU for serialization
            cpu_index = faiss.index_gpu_to_cpu(self.index)
            
        faiss.write_index(cpu_index, str(path))

    def load(self, path: Path):
        """Loads the index deterministically from disk."""
        path = Path(path)
        assert path.exists(), f"[FAIL] FAISS index file not found: {path}"
        
        cpu_index = faiss.read_index(str(path))
        
        assert cpu_index.d == self.feature_dim, (
            f"[FAIL] Loaded index dimension ({cpu_index.d}) != expected ({self.feature_dim})"
        )
        
        if self.use_gpu:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        else:
            self.index = cpu_index
            
    @property
    def total_count(self) -> int:
        return self.index.ntotal
