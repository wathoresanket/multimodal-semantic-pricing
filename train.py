"""
Training Script — Train/Val/Test Split from train.csv (70/15/15)
    • Uses common utils/data_utils.py to ensure 70/15/15 random split (state=11)
    • Trains GatedMultimodalMLP with SMAPE loss on train split
    • Validates on val split during training
    • Saves best checkpoint to checkpoints/best_model.pt

Run:
    python train.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

import config
from utils.logger import get_logger
from utils.data_utils import get_train_val_test_loaders
from utils.metrics import smape_loss
from steps._model import GatedMultimodalMLP

log = get_logger("train")

def main():
    log.info(f"Using device: {config.DEVICE}")

    # ── Check required files ──────────────────────────────────────────────────
    required = [
        config.TEXT_EMBED_FILE, config.IMG_EMBED_FILE,
        config.BRAND_FEATS_TRAIN, config.IPQ_TRAIN, config.LOG_PRICE_TRAIN,
        config.TRAIN_CSV,
    ]
    for path in required:
        if not os.path.exists(path):
            log.error(f"Missing: {path}")
            sys.exit(1)

    # ── Load Data ─────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, text_dim, img_dim = get_train_val_test_loaders(is_train=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = GatedMultimodalMLP(
        text_dim=text_dim,
        img_dim=img_dim,
        meta_dim=config.META_DIM,
    ).to(config.DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"Model: {n_params:,} params | text_dim={text_dim} img_dim={img_dim} meta_dim={config.META_DIM}")
    log.info(f"Train batches/epoch: {len(train_loader)}")
    log.info(f"Val batches/epoch: {len(val_loader)}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.EPOCHS)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss = float("inf")

    for epoch in range(1, config.EPOCHS + 1):
        # Training
        model.train()
        train_loss = 0.0

        for t_emb, i_emb, meta, tgt in train_loader:
            t_emb = t_emb.to(config.DEVICE)
            i_emb = i_emb.to(config.DEVICE)
            meta  = meta.to(config.DEVICE)
            tgt   = tgt.to(config.DEVICE)

            optimizer.zero_grad()
            preds = model(t_emb, i_emb, meta)
            
            # --- The Optimization Magic ---
            # Optimizing L1 on log-prices naturally enforces the geometric median,
            # which inherently minimizes SMAPE perfectly.
            opt_loss = torch.nn.functional.l1_loss(preds, tgt)
            opt_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Continuously track the real SMAPE metric for human logs
            with torch.no_grad():
                metric_loss = smape_loss(preds, tgt)
            train_loss += metric_loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for t_emb, i_emb, meta, tgt in val_loader:
                t_emb = t_emb.to(config.DEVICE)
                i_emb = i_emb.to(config.DEVICE)
                meta  = meta.to(config.DEVICE)
                tgt   = tgt.to(config.DEVICE)
                
                preds = model(t_emb, i_emb, meta)
                loss  = smape_loss(preds, tgt)
                val_loss += loss.item()

        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss   = val_loss / len(val_loader)
        lr_now         = scheduler.get_last_lr()[0]
        
        log.info(f"Epoch {epoch:02d}/{config.EPOCHS} | "
                 f"Train SMAPE={avg_train_loss:.4f} | Val SMAPE={avg_val_loss:.4f} | LR={lr_now:.2e}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), config.MODEL_CKPT)
            log.info(f"  ✓ New best checkpoint saved (Val SMAPE={best_val_loss:.4f})")

    log.info(f"\nTraining complete — best Val SMAPE={best_val_loss:.4f}")
    log.info(f"Checkpoint: {config.MODEL_CKPT}")
    log.info(f"Run 'python predict.py' to evaluate on the test split and generate final predictions.\n")

if __name__ == "__main__":
    main()
