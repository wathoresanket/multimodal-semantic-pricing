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
BRAND_VOCAB_PKL   = os.path.join(DATA_DIR, "brand_vocab.pkl")
BRAND_STATS_PKL   = os.path.join(DATA_DIR, "brand_stats.pkl")
LOG_PRICE_TRAIN   = os.path.join(DATA_DIR, "log_price_train.parquet")

# Step 2 / 3 outputs
TEXT_EMBED_FILE = os.path.join(EMBED_DIR, "text_embeddings.npy")
IMG_EMBED_FILE  = os.path.join(EMBED_DIR, "image_embeddings.npy")

# Step 4 output
MODEL_CKPT = os.path.join(CKPT_DIR, "best_model.pt")

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
