import pandas as pd
import numpy as np
import faiss
import pickle
import os
import random
from tqdm import tqdm

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(base_dir, "data", "styles.csv")
emb_path = os.path.join(base_dir, "data", "embeddings", "embeddings.npy")
names_path = os.path.join(base_dir, "data", "embeddings", "filenames.pkl")
index_path = os.path.join(base_dir, "data", "faiss_index", "index.bin")

def main():
    # 1. Load ground truth
    print("Reading CSV...")
    if not os.path.exists(csv_path):
        print("Missing styles.csv!")
        return

    df = pd.read_csv(csv_path, on_bad_lines='skip')
    df['id'] = df['id'].astype(str)
    
    gt_map = dict(zip(df.id, df.articleType))

    if not os.path.exists(index_path):
        print("Index not found, run indexer first.")
        return

    print("Loading index & vectors...")
    index = faiss.read_index(index_path)
    vecs = np.load(emb_path)
    
    with open(names_path, 'rb') as f:
        fnames = pickle.load(f)

    print("Mapping data...")
    vec_idx_to_cat = {}
    valid_idxs = []

    for i, f in enumerate(fnames):
        pid = f.split('.')[0]
        if pid in gt_map:
            vec_idx_to_cat[i] = gt_map[pid]
            valid_idxs.append(i)

    print(f"Found {len(valid_idxs)} images with valid labels.")
    
    n_sample = 1000
    if len(valid_idxs) < n_sample:
        n_sample = len(valid_idxs)

    print(f"Testing on {n_sample} random images...")
    test_idxs = random.sample(valid_idxs, n_sample)
    
    query_vecs = vecs[test_idxs]
    
    D, I = index.search(query_vecs, 11)

    r1, r5, r10 = 0, 0, 0

    for i in tqdm(range(len(test_idxs))):
        true_label = vec_idx_to_cat[test_idxs[i]]
        results = I[i][1:] 

        hits = []
        for rank, res_idx in enumerate(results):
            pred_label = vec_idx_to_cat.get(res_idx, "Unknown")
            if pred_label == true_label:
                hits.append(rank)

        if any(h < 1 for h in hits): r1 += 1
        if any(h < 5 for h in hits): r5 += 1
        if any(h < 10 for h in hits): r10 += 1

    print("-" * 30)
    print(f"Results (N={n_sample}):")
    print(f"R@1:  {r1/n_sample:.2%}")
    print(f"R@5:  {r5/n_sample:.2%}")
    print(f"R@10: {r10/n_sample:.2%}")
    print("-" * 30)

if __name__ == "__main__":
    main()