"""
Prediction & Evaluation Script
    • Loads best checkpoint from checkpoints/best_model.pt
    • Uses common utils/data_utils.py to ensure identical 70/15/15 random split
    • Evaluates metrics (R², MAE, RMSE, SMAPE) on the 15% test split
    • Runs inference on the true test set and generates final_submission.csv

Run:
    python predict.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch

import config
from utils.logger import get_logger
from utils.data_utils import get_train_val_test_loaders, get_submission_loader
from utils.metrics import compute_metrics
from steps._model import GatedMultimodalMLP

log = get_logger("predict")

def eval_split(model, loader, split_name):
    """Helper to evaluate on a split."""
    preds_log = []
    targets_log = []
    
    with torch.no_grad():
        for t_emb, i_emb, meta, tgt in loader:
            t_emb = t_emb.to(config.DEVICE)
            i_emb = i_emb.to(config.DEVICE)
            meta  = meta.to(config.DEVICE)
            
            preds = model(t_emb, i_emb, meta)
            preds_log.extend(preds.cpu().numpy())
            targets_log.extend(tgt.cpu().numpy())
    
    preds_log = np.array(preds_log)
    targets_log = np.array(targets_log)
    return compute_metrics(targets_log, preds_log, split_name=split_name)

def main():
    log.info(f"Using device: {config.DEVICE}")

    required = [config.MODEL_CKPT, config.CKPT_DIR + "/meta_scaler.pkl"]
    for path in required:
        if not os.path.exists(path):
            log.error(f"Missing: {path}")
            sys.exit(1)

    # ── Load Data ─────────────────────────────────────────────────────────────
    # Use is_train=False to load existing splits and scaler
    train_loader, val_loader, test_loader, text_dim, img_dim = get_train_val_test_loaders(is_train=False)

    # ── Load Model ────────────────────────────────────────────────────────────
    log.info(f"Loading best model from {config.MODEL_CKPT} …")
    model = GatedMultimodalMLP(
        text_dim=text_dim,
        img_dim=img_dim,
        meta_dim=config.META_DIM,
    ).to(config.DEVICE)
    
    model.load_state_dict(torch.load(config.MODEL_CKPT, map_location=config.DEVICE))
    model.eval()
    log.info("Model loaded ✓\n")

    # ── Evaluate on splits ────────────────────────────────────────────────────
    log.info("="*70)
    log.info("EVALUATING ON TRAINING SPLITS")
    log.info("="*70)
    
    train_metrics = eval_split(model, train_loader, "Train")
    val_metrics   = eval_split(model, val_loader, "Validation")
    test_metrics  = eval_split(model, test_loader, "Test")
    
    log.info("="*70)
    log.info("SUMMARY MAPEs/SMAPEs")
    log.info("="*70)
    log.info(f"Train SMAPE:{train_metrics['smape']:.4f}% | Val SMAPE:{val_metrics['smape']:.4f}% | Test SMAPE:{test_metrics['smape']:.4f}%")
    log.info("="*70 + "\n")
    
    # ── Final Submission ──────────────────────────────────────────────────────
    log.info("Generating predictions for final_submission.csv ...")
    submission_loader = get_submission_loader()
    
    preds_log = []
    with torch.no_grad():
        for t_emb, i_emb, meta in submission_loader:
            out = model(
                t_emb.to(config.DEVICE),
                i_emb.to(config.DEVICE),
                meta.to(config.DEVICE)
            )
            preds_log.extend(out.cpu().numpy())

    prices = np.expm1(preds_log).clip(min=0.1)
    
    test_ids = pd.read_parquet(
        os.path.join(config.DATA_DIR, "sample_id_test.parquet")
    )['sample_id'].values

    submission = pd.DataFrame({'sample_id': test_ids, 'price': prices})
    submission.to_csv(config.SUBMISSION_CSV, index=False)
    
    log.info(f"Submission saved → {config.SUBMISSION_CSV} ({len(submission)} rows)")
    log.info("Evaluation Complete ✓")


if __name__ == "__main__":
    main()
