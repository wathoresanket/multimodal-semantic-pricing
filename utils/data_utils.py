import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

import config
from utils.logger import get_logger

log = get_logger("data_utils")

def load_meta_features(split: str) -> np.ndarray:
    if split == 'train':
        brand = pd.read_parquet(config.BRAND_FEATS_TRAIN)
        ipq   = pd.read_parquet(config.IPQ_TRAIN)
    elif split == 'test':
        brand = pd.read_parquet(config.BRAND_FEATS_TEST)
        ipq   = pd.read_parquet(config.IPQ_TEST)
    else:
        raise ValueError(f"Unknown split: {split}")

    brand_cols = ['brand_count', 'brand_freq_log', 'brand_smooth_mean',
                  'brand_std', 'brand_premium', 'is_known_brand']
    meta = np.column_stack([
        ipq['ipq_value'].values,
        brand[brand_cols].values,
    ]).astype(np.float32)
    return meta

def get_train_val_test_loaders(is_train=True):
    if is_train:
        log.info("Loading dataset and creating 70/15/15 splits (random_state=11) ...")
        train_df = pd.read_csv(config.TRAIN_CSV)
        n_total = len(train_df)
        
        n_train = int(0.70 * n_total)
        n_val   = int(0.15 * n_total)
        
        all_indices = np.arange(n_total)
        rng = np.random.RandomState(11)
        rng.shuffle(all_indices)
        
        train_idx = all_indices[:n_train]
        val_idx   = all_indices[n_train:n_train + n_val]
        test_idx  = all_indices[n_train + n_val:]

        os.makedirs(config.CKPT_DIR, exist_ok=True)
        split_indices = {'train': train_idx, 'val': val_idx, 'test': test_idx}
        with open(os.path.join(config.CKPT_DIR, "split_indices.pkl"), 'wb') as f:
            pickle.dump(split_indices, f)
    else:
        log.info("Loading saved split_indices.pkl ...")
        with open(os.path.join(config.CKPT_DIR, "split_indices.pkl"), 'rb') as f:
            split_indices = pickle.load(f)
        train_idx = split_indices['train']
        val_idx   = split_indices['val']
        test_idx  = split_indices['test']

    log.info(f"Split sizes -> Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
            
    text_embeds = np.load(config.TEXT_EMBED_FILE)
    img_embeds  = np.load(config.IMG_EMBED_FILE)
    log_prices  = pd.read_parquet(config.LOG_PRICE_TRAIN)['log_price'].values
    meta_all    = load_meta_features('train')
    
    scaler_path = os.path.join(config.CKPT_DIR, "meta_scaler.pkl")
    if is_train:
        scaler = StandardScaler()
        scaler.fit(meta_all[train_idx])
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
    else:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
            
    def create_loader(indices, shuffle):
        t_emb = torch.tensor(text_embeds[indices], dtype=torch.float32)
        i_emb = torch.tensor(img_embeds[indices], dtype=torch.float32)
        m     = torch.tensor(scaler.transform(meta_all[indices]), dtype=torch.float32)
        p     = torch.tensor(log_prices[indices], dtype=torch.float32)
        return DataLoader(
            TensorDataset(t_emb, i_emb, m, p),
            batch_size=config.TRAIN_BATCH_SIZE,
            shuffle=shuffle,
            pin_memory=(config.DEVICE.type == "cuda")
        )
        
    train_loader = create_loader(train_idx, shuffle=is_train)
    val_loader   = create_loader(val_idx, shuffle=False)
    test_loader  = create_loader(test_idx, shuffle=False)
    
    return train_loader, val_loader, test_loader, text_embeds.shape[1], img_embeds.shape[1]

def get_submission_loader():
    text_embeds = np.load(config.TEXT_EMBED_FILE)
    img_embeds  = np.load(config.IMG_EMBED_FILE)
    n_train = len(pd.read_parquet(config.LOG_PRICE_TRAIN))
    
    X_text_test = text_embeds[n_train:]
    X_img_test  = img_embeds[n_train:]
    meta_test   = load_meta_features('test')
    
    scaler_path = os.path.join(config.CKPT_DIR, "meta_scaler.pkl")
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
        
    meta_test_scaled = scaler.transform(meta_test).astype(np.float32)
    
    loader = DataLoader(
        TensorDataset(
            torch.tensor(X_text_test, dtype=torch.float32),
            torch.tensor(X_img_test, dtype=torch.float32),
            torch.tensor(meta_test_scaled, dtype=torch.float32)
        ),
        batch_size=256,
        shuffle=False
    )
    return loader
