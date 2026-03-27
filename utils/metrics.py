import torch
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import config
from utils.logger import get_logger

log = get_logger("metrics")

def smape_loss(pred_log: torch.Tensor, true_log: torch.Tensor) -> torch.Tensor:
    """
    Differentiable SMAPE computed in original price space (after exp).
    Directly optimises the leaderboard metric on every weight update.
    """
    pred = torch.expm1(pred_log)
    true = torch.expm1(true_log)
    num  = torch.abs(pred - true)
    den  = (torch.abs(pred) + torch.abs(true)) / 2.0
    return torch.mean(100.0 * num / (den + config.SMAPE_EPS))

def compute_metrics(y_true_log, y_pred_log, split_name="Test"):
    """Compute all evaluation metrics given log-space predictions and targets."""
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log).clip(min=0.1)
    
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    smape = 100.0 * np.mean(2.0 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true) + 1e-8))
    mape = 100.0 * np.mean(np.abs((y_pred - y_true) / (y_true + 1e-8)))
    
    log.info(f"\n{'='*70}")
    log.info(f"{split_name} Set Evaluation Metrics:")
    log.info(f"{'='*70}")
    log.info(f"  R² Score:              {r2:.6f}")
    log.info(f"  MAE (Mean Abs Error):  ${mae:.2f}")
    log.info(f"  RMSE:                  ${rmse:.2f}")
    log.info(f"  SMAPE:                 {smape:.4f}%")
    log.info(f"  MAPE:                  {mape:.4f}%")
    log.info(f"  Min Price:             ${y_pred.min():.2f}")
    log.info(f"  Max Price:             ${y_pred.max():.2f}")
    log.info(f"  Mean Price:            ${y_pred.mean():.2f}")
    log.info(f"  Median Price:          ${np.median(y_pred):.2f}")
    log.info(f"{'='*70}\n")
    
    return {'r2': r2, 'mae': mae, 'rmse': rmse, 'smape': smape, 'mape': mape}
