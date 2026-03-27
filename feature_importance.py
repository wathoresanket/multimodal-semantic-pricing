"""
Permutation Feature Importance for GatedMultimodalMLP
This script checks how much the test SMAPE gets worse when a specific feature is randomly shuffled.
A higher error increase means the feature was extremely important for the model's performance!
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np

import config
from utils.data_utils import get_train_val_test_loaders
from utils.metrics import compute_metrics
from steps._model import GatedMultimodalMLP
from utils.logger import get_logger

log = get_logger("importance")

def evaluate_shuffled(model, loader, feature_to_shuffle):
    preds_log = []
    targets_log = []
    
    with torch.no_grad():
        for t_emb, i_emb, meta, tgt in loader:
            # Create a random permutation for the batch
            batch_size = t_emb.shape[0]
            perm = torch.randperm(batch_size).to(config.DEVICE)
            
            # Move to GPU
            t = t_emb.to(config.DEVICE)
            i = i_emb.to(config.DEVICE)
            m = meta.to(config.DEVICE)
            
            # Scramble just the requested feature
            if feature_to_shuffle == 'text':
                t = t[perm]
            elif feature_to_shuffle == 'image':
                i = i[perm]
            elif feature_to_shuffle == 'ipq':
                m[:, 0] = m[perm, 0]
            elif feature_to_shuffle == 'brand_count':
                m[:, 1] = m[perm, 1]
            elif feature_to_shuffle == 'brand_freq_log':
                m[:, 2] = m[perm, 2]
            elif feature_to_shuffle == 'brand_smooth_mean':
                m[:, 3] = m[perm, 3]
            elif feature_to_shuffle == 'brand_std':
                m[:, 4] = m[perm, 4]
            elif feature_to_shuffle == 'brand_premium':
                m[:, 5] = m[perm, 5]
            elif feature_to_shuffle == 'is_known_brand':
                m[:, 6] = m[perm, 6]
            elif feature_to_shuffle == 'all_meta':
                m = m[perm]
                
            preds = model(t, i, m)
            preds_log.extend(preds.cpu().numpy())
            targets_log.extend(tgt.numpy())
            
    # Calculate SMAPE for this messed up batch
    metrics = compute_metrics(np.array(targets_log), np.array(preds_log), split_name=f"Shuffle {feature_to_shuffle}")
    return metrics['smape']

def main():
    log.info(f"Using device: {config.DEVICE}")
    
    # Load identical Test loader (Using is_train=False ensures no data shuffling is ruined)
    _, _, test_loader, text_dim, img_dim = get_train_val_test_loaders(is_train=False)
    
    # Load Model
    model = GatedMultimodalMLP(text_dim=text_dim, img_dim=img_dim, meta_dim=config.META_DIM).to(config.DEVICE)
    model.load_state_dict(torch.load(config.MODEL_CKPT, map_location=config.DEVICE))
    model.eval()
    
    log.info("\n--- Calculating Baseline ---")
    baseline_smape = evaluate_shuffled(model, test_loader, 'none')
    log.info(f"BASELINE TEST SMAPE: {baseline_smape:.4f}%\n")
    
    features_to_test = [
        'text', 'image', 'all_meta', 
        'ipq', 'brand_count', 'brand_freq_log', 
        'brand_smooth_mean', 'brand_std', 'brand_premium', 'is_known_brand'
    ]
    
    log.info("--- Starting Permutation Importance ---")
    log.info("(Higher Error Increase = More Important Feature)\n")
    
    for f in features_to_test:
        # Run 3 times to smooth out random variance
        smapes = [evaluate_shuffled(model, test_loader, f) for _ in range(3)]
        avg_smape = np.mean(smapes)
        
        # How much did the error go up compared to baseline?
        error_increase = avg_smape - baseline_smape 
        
        log.info(f"Shuffled Feature : {f:20s} | New SMAPE: {avg_smape:.4f}% | Error Increase: +{error_increase:.4f}%")

if __name__ == "__main__":
    main()
