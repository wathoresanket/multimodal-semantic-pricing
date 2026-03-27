import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import lightgbm as lgb
import xgboost as xgb
from tqdm import tqdm

from utils.data_utils import get_train_val_test_loaders
from utils.metrics import compute_metrics
from utils.logger import get_logger

log = get_logger("tree_compare")

def extract_arrays(loader, desc="Extracting"):
    X, y = [], []
    for t_emb, i_emb, meta, tgt in tqdm(loader, desc=desc):
        # Flatten textual, visual, and tabular meta-features
        batch_x = np.concatenate([t_emb.numpy(), i_emb.numpy(), meta.numpy()], axis=1)
        X.append(batch_x)
        y.append(tgt.numpy())
    return np.vstack(X), np.concatenate(y)

def main():
    log.info("Loading identical 70/15/15 dataset splits to ensure perfect architecture comparison...")
    # Using is_train=False strictly guarantees it loads the exact split_indices.pkl without re-shuffling!
    train_loader, val_loader, test_loader, _, _ = get_train_val_test_loaders(is_train=False)
    
    log.info("Extracting embeddings into CPU RAM (this may take up to 20 seconds)...")
    X_train, y_train = extract_arrays(train_loader, "Train Split")
    X_val, y_val     = extract_arrays(val_loader,   "Val Split  ")
    X_test, y_test   = extract_arrays(test_loader,  "Test Split ")
    
    log.info(f"Train Dataset Shape: {X_train.shape} (2240 pristine dimensions)")
    
    # ── LightGBM ─────────────────────────────────────────────────────────────
    log.info("\n--- Training LightGBM Architecture ---")
    lgbm_model = lgb.LGBMRegressor(
        n_estimators=200, 
        learning_rate=0.05, 
        random_state=11, 
        n_jobs=-1
    )
    lgbm_model.fit(X_train, y_train)
    
    lgbm_test_preds = lgbm_model.predict(X_test)
    log.info("LightGBM Test Metrics:")
    res_lgbm = compute_metrics(y_test, lgbm_test_preds, "LGBM Test")
    
    # ── XGBoost ─────────────────────────────────────────────────────────────
    log.info("\n--- Training XGBoost Architecture ---")
    xgb_model = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        random_state=11,
        n_jobs=-1,
        tree_method='hist' # Fast histogram method for huge datasets natively on CPU
    )
    xgb_model.fit(X_train, y_train)
    
    xgb_test_preds = xgb_model.predict(X_test)
    
    log.info("XGBoost Test Metrics:")
    res_xgb = compute_metrics(y_test, xgb_test_preds, "XGB Test")
    
    # ── Ridge Regression ─────────────────────────────────────────────────────────────
    from sklearn.linear_model import Ridge
    log.info("\n--- Training Ridge Regression ---")
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_train, y_train)
    
    ridge_test_preds = ridge_model.predict(X_test)
    log.info("Ridge Test Metrics:")
    res_ridge = compute_metrics(y_test, ridge_test_preds, "Ridge Test")
    
    # ── Summary ─────────────────────────────────────────────────────────────
    log.info("\n==================================================")
    log.info("FINAL ARCHITECTURE COMPARISON (TEST SMAPE)")
    log.info("==================================================")
    log.info(f"PyTorch Deep ResNet : ~ 48.77%")
    log.info(f"Ridge Regression    : {res_ridge['smape']:.4f}%")
    log.info(f"LightGBM            : {res_lgbm['smape']:.4f}%")
    log.info(f"XGBoost             : {res_xgb['smape']:.4f}%")
    log.info("==================================================")

if __name__ == "__main__":
    main()
