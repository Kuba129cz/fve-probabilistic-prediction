import os
import glob
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.preprocessing import Preprocessor
from src.dataset import Dataset
from train import parser, load_dataset, set_seed  # Recycling your functions
from src.models import model_attention

def load_preprocessor_and_data(args):
    """Loads data and initializes the preprocessor with correct scalers."""
    train_df, val_df, test_df = load_dataset(args.dataset_file, train_ratio=args.train_ratio, val_ratio=args.val_ratio)
    
    feature_cols = list(set(args.lookback_cols + args.horizon_cols))
    preprocessor = Preprocessor(feature_cols=feature_cols, target_col=args.target_col)
    
    _, _, test_scaled = preprocessor.process_data(train_df=train_df, val_df=val_df, test_df=test_df)
    return test_scaled, preprocessor

def run_inference(model, dataloader, preprocessor, args, device):
    """Runs through the test set and returns rescaled predictions and targets."""
    model.eval()
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for past, future, targets in dataloader:
            past, future, targets = past.to(device), future.to(device), targets.to(device)
            preds = model(past, future)
            
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())
            
    preds_tensor = torch.cat(all_preds, dim=0)
    targets_tensor = torch.cat(all_targets, dim=0)
    if targets_tensor.dim() == 2:
        targets_tensor = targets_tensor.unsqueeze(-1)
        
    # --- INVERSE TRANSFORMATION (RESCALING) ---
    original_preds_shape = preds_tensor.shape
    preds_flat = preds_tensor.numpy().reshape(-1, 1)
    preds_rescaled = preprocessor.target_scaler.inverse_transform(preds_flat).reshape(original_preds_shape)
    
    original_targets_shape = targets_tensor.shape
    targets_flat = targets_tensor.numpy().reshape(-1, 1)
    targets_rescaled = preprocessor.target_scaler.inverse_transform(targets_flat).reshape(original_targets_shape)
    
    return preds_rescaled, targets_rescaled

def plot_and_save_samples(preds, targets, quantiles, num_plots=100, output_dir="plots"):
    """Plots and saves 24-hour prediction samples sequentially matching t+24 steps."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Find indices for the target quantiles
    q_10_idx = quantiles.index(0.1)
    q_30_idx = quantiles.index(0.3)
    q_50_idx = quantiles.index(0.5)
    q_70_idx = quantiles.index(0.7)
    q_90_idx = quantiles.index(0.9)
    
    total_samples = preds.shape[0]
    sample_indices = list(range(0, total_samples, 24))[:num_plots]
    
    x_hours = np.arange(1, 25) # 1 to 24 hours horizon
    
    for idx in sample_indices:
        plt.figure(figsize=(12, 6))
        
        # Extract curves for the specific sample
        true_y = targets[idx, :, 0]
        pred_q10 = preds[idx, :, q_10_idx]
        pred_q30 = preds[idx, :, q_30_idx]
        pred_q50 = preds[idx, :, q_50_idx]
        pred_q70 = preds[idx, :, q_70_idx]
        pred_q90 = preds[idx, :, q_90_idx]
        

        plt.plot(x_hours, true_y, label="Actual (Real)", color="black", 
                 linewidth=2, marker='o', markersize=4, zorder=5)
        
        # Predicted median (ensemble average)
        plt.plot(x_hours, pred_q50, label="Predicted Median (q=0.5)", color="blue", 
                 linestyle="--", linewidth=2, marker='s', markersize=4)
        
        # 80% Confidence Interval
        plt.fill_between(x_hours, pred_q10, pred_q90, color="blue", alpha=0.15, label="80% Confidence Interval")
        
        # 40% Confidence Interval
        plt.fill_between(x_hours, pred_q30, pred_q70, color="blue", alpha=0.30, label="40% Confidence Interval")
        
        plt.title(f"Ensemble: Probabilistic PV Forecast for 24h Ahead (Sample Index: {idx})")
        plt.xlabel("Forecast Horizon (Hours Ahead)")
        plt.ylabel("PV Power Output (kW)")
        plt.xticks(x_hours) 
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(loc="upper left")
        
        # Save plot
        plot_path = os.path.join(output_dir, f"ensemble_prediction_{idx:04d}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Plot saved to: {plot_path}")

def main():
    args = parser.parse_args([])
    set_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on device: {device}")
    
    test_scaled, preprocessor = load_preprocessor_and_data(args)
    
    test_dataset = Dataset(
        data=test_scaled, 
        lookback=args.lookback, 
        horizon=args.horizon, 
        lookback_cols=args.lookback_cols, 
        horizon_cols=args.horizon_cols, 
        target_col=args.target_col
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    model = model_attention.Model(args=args).to(device)
    
    model_paths = glob.glob(os.path.join(args.save_dir, "best_model_*.pth"))
    
    if not model_paths:
        raise FileNotFoundError(f"No models found in the directory '{args.save_dir}'.")
        
    print(f"\nFound a total of {len(model_paths)} models for the Ensemble:")
    
    all_model_preds = []
    final_targets = None
    
    for idx, path in enumerate(model_paths, 1):
        print(f"  [{idx}/{len(model_paths)}] Loading and predicting: {os.path.basename(path)}")
        model.load_state_dict(torch.load(path, map_location=device))
        
        preds, targets = run_inference(model, test_loader, preprocessor, args, device)
        all_model_preds.append(preds)
        
        if final_targets is None:
            final_targets = targets
            
    ensemble_preds = np.mean(all_model_preds, axis=0)
    print("\nEnsemble aggregation completed.")
    
    plot_and_save_samples(ensemble_preds, final_targets, args.quantiles, num_plots=100, output_dir="plots")
    print("\nAll Ensemble plots have been successfully generated in the 'plots/' directory.")

if __name__ == "__main__":
    main()