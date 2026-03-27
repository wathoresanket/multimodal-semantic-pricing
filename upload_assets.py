import os
from huggingface_hub import HfApi, create_repo

# Configuration
REPO_ID = "wathoresanket/pricing-assets"  # Update this to your desired name
FILES_TO_UPLOAD = [
    "data/faiss_index.bin",
    "data/faiss_meta.pkl",
    "data/brand_vocab.pkl",
    "data/brand_stats.pkl",
    "checkpoints/best_model.pt",
    "checkpoints/meta_scaler.pkl"
]

def upload():
    api = HfApi()
    
    print(f"Checking/Creating repository: {REPO_ID}...")
    try:
        create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)
        print(f"✓ Repository {REPO_ID} is ready.")
    except Exception as e:
        print(f"Error creating repo: {e}")
        print("Make sure you are logged in using 'huggingface-cli login'")
        return

    for file_path in FILES_TO_UPLOAD:
        if os.path.exists(file_path):
            print(f"Uploading {file_path}...")
            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=file_path,
                repo_id=REPO_ID,
                repo_type="model"
            )
            print(f"✓ Uploaded {file_path}")
        else:
            print(f"Warning: {file_path} not found, skipping.")

if __name__ == "__main__":
    upload()
