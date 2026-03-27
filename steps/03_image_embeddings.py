"""
Step 3 — Image Embeddings (SigLIP-2 SO400M High-Resolution)
    • Reads image URLs from train.csv + test.csv directly
    • Downloads each image with retry + timeout
    • Generates pooler embeddings with google/siglip2-so400m-patch14-384
    • Writes embeddings/image_embeddings.npy

WHY SIGLIP-2 SO400M INSTEAD OF MARQO/OPENCLIP:
    • While Marqo is fine-tuned for e-commerce, it relies on standard CLIP InfoNCE loss. 
      This contrastive loss creates batch-norm drift, which destroys the stability of the 
      vector space for continuous regression tasks.
    • SigLIP-2 uses pairwise sigmoid loss, which evaluates each image-text pair independently,
      providing the deterministic stability required for an MLP or XGBoost regressor.
    • Upgrading from the "base" model to the "SO400M" (approx 1B total parameters) and 
      increasing the resolution to 384x384 allows the Vision Transformer to capture fine-grained 
      packaging text and material textures that dictate premium vs. budget pricing.

Run:
    nohup python steps/03_image_embeddings.py > logs/03_image_embeddings.out 2>&1 &
    tail -f logs/03_image_embeddings.out
    grep "FAIL\\|GAVE_UP\\|ERROR" logs/03_image_embeddings.out
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import time
import io
import numpy as np
import pandas as pd
import requests
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, SiglipVisionModel

import config
from utils.logger import get_logger

log = get_logger("03_image_embeddings")

# Updated dummy image size to match the new 384x384 resolution of the SO400M model
DUMMY_IMG = Image.new("RGB", (384, 384), (200, 200, 200))


# ── Image download with retry ─────────────────────────────────────────────────

def download_image(url: str, idx: int) -> tuple[Image.Image, str]:
    for attempt in range(1, config.IMG_MAX_RETRIES + 2):
        try:
            resp = requests.get(
                str(url),
                timeout=config.IMG_TIMEOUT_SECS,
                headers={"User-Agent": "Mozilla/5.0"},
                stream=True,
            )
            resp.raise_for_status()
            img_bytes   = resp.content
            img         = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            status      = "ok" if attempt == 1 else f"retry_ok(attempt={attempt})"
            content_len = int(resp.headers.get("Content-Length", 0))
            log.debug(f"[{idx:05d}] {status} | size={img.size} "
                      f"| bytes={len(img_bytes):,} | content_len={content_len:,} "
                      f"| url={str(url)[:80]}")
            return img, status

        except requests.exceptions.Timeout:
            log.warning(f"[{idx:05d}] TIMEOUT attempt={attempt} | url={str(url)[:80]}")
        except requests.exceptions.HTTPError as e:
            log.warning(f"[{idx:05d}] HTTP_ERROR {e.response.status_code} "
                        f"attempt={attempt} | url={str(url)[:80]}")
            break  # no point retrying 404
        except Exception as e:
            log.warning(f"[{idx:05d}] FAIL attempt={attempt} | "
                        f"{type(e).__name__}: {e} | url={str(url)[:80]}")

        if attempt <= config.IMG_MAX_RETRIES:
            time.sleep(0.5 * attempt)

    log.error(f"[{idx:05d}] GAVE_UP — using dummy image | url={str(url)[:80]}")
    return DUMMY_IMG, "failed"


# ── Embedding loop ────────────────────────────────────────────────────────────

def embed_images(urls, processor, model) -> np.ndarray:
    all_embeds = []
    stats      = {"ok": 0, "retry_ok": 0, "failed": 0}
    log_every  = max(1, len(urls) // 20)   # log summary every ~5%

    for idx, url in enumerate(tqdm(urls, desc="img-embed")):
        img, status = download_image(url, idx)

        if "failed" in status:
            stats["failed"] += 1
        elif "retry" in status:
            stats["retry_ok"] += 1
        else:
            stats["ok"] += 1

        with torch.no_grad():
            inputs = processor(images=img, return_tensors="pt")
            inputs = {k: v.to(config.DEVICE) for k, v in inputs.items()}  # ✅ fix
            out    = model(**inputs)
            # The larger model outputs a denser embedding vector
            embed  = out.pooler_output.cpu().numpy().flatten()

        all_embeds.append(embed)

        if (idx + 1) % log_every == 0 or (idx + 1) == len(urls):
            pct = (idx + 1) / len(urls) * 100
            log.info(
                f"Progress: {idx+1}/{len(urls)} ({pct:.1f}%) | "
                f"ok={stats['ok']} retry={stats['retry_ok']} fail={stats['failed']} | "
                f"remaining={len(urls) - (idx+1)}"
            )

    log.info(f"Download summary — ok={stats['ok']}, retry_ok={stats['retry_ok']}, "
             f"failed={stats['failed']}, total={len(urls)}")
    return np.array(all_embeds)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log.info(f"Using device: {config.DEVICE}")
    
    # BEST ARCHITECTURE UPGRADE: SigLIP 2 SO400M (High Resolution 384x384)
    MODEL_NAME = "google/siglip2-so400m-patch14-384"
    log.info(f"Vision model: {MODEL_NAME}")

    # ✅ FIXED
    for path in [config.TRAIN_CSV, config.TEST_CSV]:
        if not os.path.exists(path):
            log.error(f"Missing: {path}")
            sys.exit(1)

    log.info("Loading image URLs …")
    train = pd.read_csv(config.TRAIN_CSV)
    test  = pd.read_csv(config.TEST_CSV)
    urls  = train['image_link'].tolist() + test['image_link'].tolist()
    log.info(f"Total images: {len(urls)} (train={len(train)}, test={len(test)})")

    log.info(f"Loading model: {MODEL_NAME} …")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model     = SiglipVisionModel.from_pretrained(MODEL_NAME).to(config.DEVICE)
    model.eval()
    
    param_count = sum(p.numel() for p in model.parameters())
    log.info(f"Model loaded — {param_count/1e6:.0f}M parameters")

    log.info("Starting image download + embedding …")
    t0      = time.time()
    embeds  = embed_images(urls, processor, model)
    elapsed = time.time() - t0
    log.info(f"Done in {elapsed/60:.1f} min — shape={embeds.shape}")

    os.makedirs(config.EMBED_DIR, exist_ok=True)
    np.save(config.IMG_EMBED_FILE, embeds)
    log.info(f"Saved → {config.IMG_EMBED_FILE}")

if __name__ == "__main__":
    main()