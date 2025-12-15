import faiss
import numpy as np
import os
import math

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMB_PATH = os.path.join(BASE_DIR, "data", "embeddings", "embeddings.npy")
INDEX_DIR = os.path.join(BASE_DIR, "data", "faiss_index")
INDEX_FILE = os.path.join(INDEX_DIR, "index.bin")

def main():
    if not os.path.exists(INDEX_DIR):
        os.makedirs(INDEX_DIR)

    if not os.path.exists(EMB_PATH):
        print(f"Embeddings file not found: {EMB_PATH}")
        return

    print("Loading vectors...")
    data = np.load(EMB_PATH).astype('float32')
    n_samples, d = data.shape

    print(f"Loaded {n_samples} vectors (dim={d}).")

    # Use Flat index for small datasets, IVF for larger ones
    if n_samples < 2000:
        print("Small dataset detected. Using IndexFlatIP (Exact Search).")
        index = faiss.IndexFlatIP(d)
        index.add(data)
    else:
        nlist = int(4 * math.sqrt(n_samples))
        print(f"Building IVF Index (nlist={nlist})...")
        
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
        
        print("Training index...")
        index.train(data)
        print("Adding vectors...")
        index.add(data)

    print(f"Saving index to {INDEX_FILE}...")
    faiss.write_index(index, INDEX_FILE)
    print("Done.")

if __name__ == "__main__":
    main()