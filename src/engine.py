import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.metrics import compute_all_metrics

def train_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    """It runs one epoch over the training data."""
    model.train()
    epoch_loss = 0.0
    
    for past, future, targets in dataloader:
        past, future, targets = past.to(device), future.to(device), targets.to(device)
        
        optimizer.zero_grad()
        preds = model(past, future)         
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(dataloader)


def evaluate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, preprocessor, quantiles: list[float], device: torch.device) -> tuple[float, dict[str, float]]:
    """It runs through validation data and calculates real energy metrics."""
    model.eval()
    epoch_loss = 0.0
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for past, future, targets in dataloader:
            past, future, targets = past.to(device), future.to(device), targets.to(device)
            
            preds = model(past, future)
            loss = criterion(preds, targets)
            epoch_loss += loss.item()
            
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())
            
    avg_loss = epoch_loss / len(dataloader)
    
    preds_tensor = torch.cat(all_preds, dim=0)
    targets_tensor = torch.cat(all_targets, dim=0)

    if targets_tensor.dim() == 2:
        targets_tensor = targets_tensor.unsqueeze(-1)

    original_preds_shape = preds_tensor.shape # [6674, 24, 5]
    preds_flat = preds_tensor.numpy().reshape(-1, 1)
    preds_rescaled = preprocessor.target_scaler.inverse_transform(preds_flat).reshape(original_preds_shape)

    original_targets_shape = targets_tensor.shape # [6674, 24, 1]
    targets_flat = targets_tensor.numpy().reshape(-1, 1)
    targets_rescaled = preprocessor.target_scaler.inverse_transform(targets_flat).reshape(original_targets_shape)

    preds_rescaled = torch.tensor(preds_rescaled)    # [6674, 24, 5]
    targets_rescaled = torch.tensor(targets_rescaled)  # [6674, 24, 1]

    metrics = compute_all_metrics(preds_rescaled, targets_rescaled, quantiles)

    return avg_loss, metrics


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    preprocessor,
    writer,
    device: torch.device,
    args
) -> None:
    """The main control function of the entire training process with Early Stopping."""
    best_val_loss = float('inf')
    patience_counter = 0 
    os.makedirs(args.save_dir, exist_ok=True)
    
    print(f"\nLaunching training on {args.epochs} epoch (Patience: {args.patience})...")
    
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, preprocessor, args.quantiles, device)
        
        scheduler.step()
        
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Validation", val_loss, epoch)
        writer.add_scalar("Train/Learning_Rate", optimizer.param_groups[0]['lr'], epoch)
        
        for metric_name, value in val_metrics.items():
            writer.add_scalar(f"Metrics/{metric_name}", value, epoch)
            
        print(f"Epoch {epoch:03d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val MAE: {val_metrics.get('mae', 0.0):.2f} kW | "
              f"Val MAE (Day): {val_metrics.get('mae_day', 0.0):.2f} kW")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0  
            checkpoint_path = os.path.join(args.save_dir, f"best_model_{args.exp_name}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  --> New best model Loss saved with Val: {val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"  [Early Stopping] with no improvement {patience_counter}/{args.patience}")
            
            if patience_counter >= args.patience:
                print(f"\n[STOP] Early Stopping activated! Training interrupted at epoch {epoch}.")
                break
                
    print("\nTraining completed.")