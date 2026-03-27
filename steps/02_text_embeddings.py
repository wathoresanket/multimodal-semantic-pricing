"""
Step 2 — Text Embeddings (DeBERTa-v3-large)
    • Reads train and test data containing 'catalog_content'
    • Cleans the unstructured text by removing noisy structural labels
    • Generates CLS-token embeddings with microsoft/deberta-v3-large
    • Writes embeddings/text_embeddings.npy

WHY CATALOG_CONTENT INSTEAD OF ITEM_NAME:
    • Top-performing teams proved that 'catalog_content' contains critical pricing signals 
      (e.g., "Pack of 6", material types) that are missed if using only the item name.
    • We apply a regex cleaning step to remove boilerplate ("Bullet Point 1:", "Value:") 
      to maximize semantic fidelity without diluting the signal with useless structural words.
    • TEXT_MAX_LEN is increased to 512 to capture the full context of the product description.

UPGRADE PATH:
    • Update your config.py to point TRAIN_CSV and TEST_CSV to your 
      full datasets (csv or parquet) instead of the isolated item_name files.

Run:
    nohup python steps/02_text_embeddings.py > logs/02_text_embeddings.out 2>&1 &
    tail -f logs/02_text_embeddings.out
"""
import sys, os, re
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

import config
from utils.logger import get_logger

log = get_logger("02_text_embeddings")

def clean_catalog_content(text):
    """
    Cleans the raw catalog string by removing repetitive dataset boilerplate.
    """
    if pd.isna(text):
        return ""
    text = str(text)
    
    # Remove repetitive boilerplate labels to focus on pure semantic meaning
    text = re.sub(r'(Item Name:|Bullet Point \d+:|Product Description:|Value:|Unit:)', ' ', text, flags=re.IGNORECASE)
    
    # Normalize whitespace and remove extra newlines
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def embed_texts(texts: list, tokenizer, model, batch_size: int) -> np.ndarray:
    all_embeds = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    for i in tqdm(range(0, len(texts), batch_size), total=total_batches, desc="text-embed"):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,  # Increased to handle the full catalog context
            return_tensors="pt",
        )

        # Move tensors to device safely
        inputs = {k: v.to(config.DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            out = model(**inputs)

        # Extract CLS token embedding (first token)
        embeds = out.last_hidden_state[:, 0, :].cpu().numpy()  
        all_embeds.extend(embeds)

        if (i // batch_size) % 100 == 0 and i > 0:
            log.info(f"  Batch {i // batch_size}/{total_batches} | embedded {i + batch_size} texts so far")

    return np.array(all_embeds)

def main():
    log.info(f"Using device: {config.DEVICE}")

    # Note: Ensure your config.py has config.TRAIN_CSV and config.TEST_CSV
    for path in [config.TRAIN_CSV, config.TEST_CSV]:
        if not os.path.exists(path):
            log.error(f"Missing {path} — ensure data paths are correct in config!")
            sys.exit(1)

    log.info("Loading full catalog content …")
    
    # Dynamically read either parquet or csv based on your pipeline
    train_df = pd.read_parquet(config.TRAIN_CSV) if config.TRAIN_CSV.endswith('.parquet') else pd.read_csv(config.TRAIN_CSV)
    test_df = pd.read_parquet(config.TEST_CSV) if config.TEST_CSV.endswith('.parquet') else pd.read_csv(config.TEST_CSV)
    
    log.info("Cleaning noisy prefixes from catalog content...")
    train_texts = train_df['catalog_content'].apply(clean_catalog_content).tolist()
    test_texts  = test_df['catalog_content'].apply(clean_catalog_content).tolist()
    texts = train_texts + test_texts
    
    log.info(f"Total texts to embed: {len(texts)} (train={len(train_texts)}, test={len(test_texts)})")
    log.info(f"Sample cleaned text: {texts[:200]}...")

    # Using DeBERTa v3 Large as the optimal text encoder
    MODEL_NAME = "microsoft/deberta-v3-large"
    log.info(f"Loading model: {MODEL_NAME} …")
    
    # Note: DeBERTa requires sentencepiece. Ensure you run `pip install sentencepiece`
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(config.DEVICE)
    model.eval()
    
    param_count = sum(p.numel() for p in model.parameters())
    log.info(f"Model loaded — {param_count/1e6:.0f}M parameters")

    log.info("Generating embeddings …")
    embeds = embed_texts(texts, tokenizer, model, config.TEXT_BATCH_SIZE)

    os.makedirs(config.EMBED_DIR, exist_ok=True)
    np.save(config.TEXT_EMBED_FILE, embeds)
    log.info(f"Saved → {config.TEXT_EMBED_FILE} shape={embeds.shape} dtype={embeds.dtype}")

if __name__ == "__main__":
    main()