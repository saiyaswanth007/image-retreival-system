import sys
from pathlib import Path

# Add src to pythonpath
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from utils.index import RetrievalIndex

def verify_faiss_wrapper():
    print("Testing FAISS RetrievalIndex Wrapper...")
    
    # 1. Initialize
    feature_dim = 128
    index = RetrievalIndex(feature_dim=feature_dim)
    print(f"[OK] Initialized. Using GPU: {index.use_gpu}")
    
    # 2. Add vectors (dummy gallery)
    # 5000 images, 128 D
    gallery = np.random.randn(5000, 128)
    index.add(gallery)
    print(f"[OK] Added 5000 vectors. Total count: {index.total_count}")
    
    # 3. Validation catches bad types
    try:
        index.add(np.random.randn(10, 64)) # Wrong dimension
        print("[FAIL] Failed to catch dimension mismatch")
        sys.exit(1)
    except AssertionError as e:
        print(f"[OK] Successfully caught validation error: {e}")
        
    # 4. Search
    queries = np.random.randn(10, 128)
    print(f"[OK] Searching for top-5 for 10 queries...")
    distances, indices = index.search(queries, k=5)
    
    assert distances.shape == (10, 5)
    assert indices.shape == (10, 5)
    print(f"[OK] Returned distances shape {distances.shape}")
    print(f"Sample Top-1 index: {indices[0][0]}, Distance: {distances[0][0]:.4f}")
    
    # 5. Exact Search check (Self-query)
    distances, indices = index.search(gallery[:1], k=1)
    assert indices[0][0] == 0, "[FAIL] Identity search did not return self!"
    assert np.isclose(distances[0][0], 0.0), "[FAIL] Self-distance is not 0"
    print("[OK] Self-query returned 0 distance exactly!")
    
    # 6. Serialization
    tmp_path = ROOT / "outputs" / "test_index.faiss"
    index.save(tmp_path)
    print(f"[OK] Saved to {tmp_path}")
    
    index2 = RetrievalIndex(feature_dim=128)
    index2.load(tmp_path)
    print(f"[OK] Loaded index from disk. Total count: {index2.total_count}")
    assert index2.total_count == 5000
    tmp_path.unlink() # Cleanup
    
    print("\n--- All FAISS Wrapper Tests Passed ---")

if __name__ == "__main__":
    verify_faiss_wrapper()
