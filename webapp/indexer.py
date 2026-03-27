"""
Similar Products Index — FAISS-based nearest neighbor search.

Builds an index from training set embeddings and metadata.
"""
import sys, os, pickle
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import config

INDEX_PATH = os.path.join(config.DATA_DIR, "faiss_index.bin")
META_PATH = os.path.join(config.DATA_DIR, "faiss_meta.pkl")


class SimilarProductsIndex:
    """Builds and queries a FAISS index over training product embeddings."""

    def __init__(self):
        print("[SimilarIndex] Initializing...")
        import faiss

        if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
            print("[SimilarIndex] Loading cached index...")
            self.index = faiss.read_index(INDEX_PATH)
            with open(META_PATH, 'rb') as f:
                self.metadata = pickle.load(f)
            print(f"[SimilarIndex] Loaded {self.index.ntotal} products ✓")
        else:
            print("[SimilarIndex] Building index from scratch...")
            self._build_index(faiss)

    def _build_index(self, faiss):
        """Build FAISS index from training embeddings."""
        # Load embeddings (train portion only)
        text_embeds = np.load(config.TEXT_EMBED_FILE)
        img_embeds = np.load(config.IMG_EMBED_FILE)
        n_train = len(pd.read_parquet(config.LOG_PRICE_TRAIN))

        text_train = text_embeds[:n_train].astype(np.float32)
        img_train = img_embeds[:n_train].astype(np.float32)

        # Concatenate text + image as the search space
        combined = np.concatenate([text_train, img_train], axis=1)

        # L2 normalize for cosine similarity
        norms = np.linalg.norm(combined, axis=1, keepdims=True)
        norms[norms == 0] = 1
        combined = combined / norms

        # Build index (Inner Product = cosine similarity after normalization)
        dim = combined.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(combined)

        # Load metadata
        train_df = pd.read_csv(config.TRAIN_CSV)
        self.metadata = {
            'item_names': [],
            'prices': train_df['price'].values.tolist(),
            'image_links': train_df['image_link'].values.tolist(),
            'sample_ids': train_df['sample_id'].values.tolist(),
        }

        # Extract item names
        import re
        for text in train_df['catalog_content'].values:
            if not isinstance(text, str):
                self.metadata['item_names'].append("Unknown Product")
                continue
            m = re.search(
                r'[Ii]tem\s+[Nn]ame\s*:\s*(.+?)(?:\n|\\n|Bullet|Value:|Unit:|$)',
                text
            )
            if m:
                self.metadata['item_names'].append(m.group(1).strip())
            else:
                self.metadata['item_names'].append(' '.join(text.split()[:10]))

        # Save index
        faiss.write_index(self.index, INDEX_PATH)
        with open(META_PATH, 'wb') as f:
            pickle.dump(self.metadata, f)

        print(f"[SimilarIndex] Built index with {self.index.ntotal} products ✓")

    def find_similar(self, text_embedding: np.ndarray, image_embedding: np.ndarray, k: int = 8) -> list:
        """Find k most similar products given text + image embeddings."""
        # Concatenate and normalize query
        query = np.concatenate([text_embedding, image_embedding]).astype(np.float32)
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm
        query = query.reshape(1, -1)

        # Search
        scores, indices = self.index.search(query, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append({
                "name": self.metadata['item_names'][idx][:80],
                "price": round(self.metadata['prices'][idx], 2),
                "image_url": self.metadata['image_links'][idx],
                "similarity": round(float(score) * 100, 1),
            })

        return results


# ── Singleton ────────────────────────────────────────────────────────────────

_index = None

def get_index() -> SimilarProductsIndex:
    global _index
    if _index is None:
        _index = SimilarProductsIndex()
    return _index
