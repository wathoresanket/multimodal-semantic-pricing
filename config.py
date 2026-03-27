"""
Central config — edit paths and hyperparams here, nowhere else.

Model upgrade notes
───────────────────
TEXT_MODEL_NAME:
  Current: "BAAI/bge-m3"            — 570M params, 1024-dim, excellent open-source
  Upgrade:  "Qwen3-Embedding-0.6B"  — similar size, newer training, marginal SMAPE gain
  Heavy:    "nvidia/NV-Embed-v2"    — 7.85B params, near 8B challenge limit

VISION_MODEL_NAME:
  Current: "google/siglip2-base-patch16-224"  — SigLIP-2 base, drop-in replacement
  Upgrade: "marqo-ai/marqo-ecommerce-embeddings-b"  — fine-tuned on e-comm images,
            requires OpenCLIP instead of transformers (see 03_image_embeddings.py)

GLiNER is REMOVED. Brand extraction now uses fast regex + frequency-vocab approach.
No GPU needed for Step 1. Runs in seconds instead of 30+ minutes.
"""
import os
import torch
from huggingface_hub import hf_hub_download

# ── Asset Management (External Hub) ───────────────────────────────────────────
ASSET_REPO = "wathoresanket/pricing-assets"

def get_asset(filename, repo_id=None):
    """Download asset from HF Hub if missing locally."""
    if repo_id is None: repo_id = ASSET_REPO
    
    # Map filenames to their repository/local subfolders
    if filename.endswith('.npy'):
        repo_path = f"embeddings/{filename}"
        local_dir = EMBED_DIR
    elif filename.endswith('.pt') or filename.endswith('scaler.pkl'):
        repo_path = f"checkpoints/{filename}"
        local_dir = CKPT_DIR
    else:
        repo_path = f"data/{filename}"
        local_dir = DATA_DIR
        
    local_path = os.path.join(local_dir, filename)
    
    if not os.path.exists(local_path):
        print(f"[Config] Fetching {repo_path} from {repo_id}...")
        try:
            return hf_hub_download(
                repo_id=repo_id, 
                filename=repo_path, 
                local_dir=BASE_DIR,  # This will place it in data/ or checkpoints/ correctly
            )
        except Exception as e:
            print(f"[Config] Warning: Failed to download {repo_path}: {e}")
            return local_path
    return local_path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "data")
EMBED_DIR = os.path.join(BASE_DIR, "embeddings")
LOG_DIR   = os.path.join(BASE_DIR, "logs")
CKPT_DIR  = os.path.join(BASE_DIR, "checkpoints")

TRAIN_CSV      = os.path.join(DATA_DIR, "train.csv")
TEST_CSV       = os.path.join(DATA_DIR, "test.csv")
SUBMISSION_CSV = os.path.join(DATA_DIR, "final_submission.csv")

# Step 1 outputs
BRAND_FEATS_TRAIN = os.path.join(DATA_DIR, "brand_features_train.parquet")
BRAND_FEATS_TEST  = os.path.join(DATA_DIR, "brand_features_test.parquet")
ITEM_NAME_TRAIN   = os.path.join(DATA_DIR, "item_name_train.parquet")
ITEM_NAME_TEST    = os.path.join(DATA_DIR, "item_name_test.parquet")
IPQ_TRAIN         = os.path.join(DATA_DIR, "ipq_train.parquet")
IPQ_TEST          = os.path.join(DATA_DIR, "ipq_test.parquet")
BRAND_VOCAB_PKL   = get_asset("brand_vocab.pkl")
BRAND_STATS_PKL   = get_asset("brand_stats.pkl")
LOG_PRICE_TRAIN   = os.path.join(DATA_DIR, "log_price_train.parquet")

# Step 2 / 3 outputs (Optional for webapp if FAISS_INDEX exists)
TEXT_EMBED_FILE = os.path.join(EMBED_DIR, "text_embeddings.npy")
IMG_EMBED_FILE  = os.path.join(EMBED_DIR, "image_embeddings.npy")

# Step 4 output
MODEL_CKPT  = get_asset("best_model.pt")
META_SCALER = get_asset("meta_scaler.pkl")

# Search Index
FAISS_INDEX = get_asset("faiss_index.bin")
FAISS_META  = get_asset("faiss_meta.pkl")

# ── Models ────────────────────────────────────────────────────────────────────
TEXT_MODEL_NAME   = "BAAI/bge-m3"
VISION_MODEL_NAME = "google/siglip2-base-patch16-224"   # SigLIP-2 (upgraded from v1)

# ── Hardware ──────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Embedding generation ──────────────────────────────────────────────────────
TEXT_BATCH_SIZE  = 32
TEXT_MAX_LEN     = 128    # item_name is short — 128 is plenty (was 512 for full catalog)
IMG_TIMEOUT_SECS = 5
IMG_MAX_RETRIES  = 2

# ── Brand extraction ──────────────────────────────────────────────────────────
BRAND_UNIGRAM_MIN_FREQ = 30
BRAND_BIGRAM_MIN_FREQ  = 15
BRAND_MIN_TOKEN_LEN    = 3
BAYESIAN_SMOOTHING_K   = 10

# ── Model architecture ────────────────────────────────────────────────────────
# Meta features: ipq_value (1) + 6 brand features = 7 total
META_DIM = 7

# ── Training ──────────────────────────────────────────────────────────────────
TRAIN_BATCH_SIZE = 128
EPOCHS           = 25
LR               = 1e-3
WEIGHT_DECAY     = 1e-4
SMAPE_EPS        = 1e-8
