import sys, os
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

import config

FILES_TO_RESTORE = [
    "faiss_index.bin",
    "faiss_meta.pkl",
    "brand_vocab.pkl",
    "brand_stats.pkl",
    "best_model.pt",
    "meta_scaler.pkl"
]

def restore():
    print("🚀 Restoring local assets from Hugging Face Hub...")
    for filename in FILES_TO_RESTORE:
        path = config.get_asset(filename)
        if os.path.exists(path):
            print(f"✓ Restored {filename} to {path}")
        else:
            print(f"❌ Failed to restore {filename}")

if __name__ == "__main__":
    restore()
