import torch

def compute_mae(pred_median: torch.Tensor, targets: torch.Tensor) -> float:
    """Computes Mean Absolute Error for the median prediction."""
    return torch.abs(targets - pred_median).mean().item()


def compute_masked_mae(pred_median: torch.Tensor, targets: torch.Tensor, threshold: float = 0.01) -> float:
    """
    Computes Mean Absolute Error only for time steps where target power is strictly positive (daytime).
    """
    # Vytvoříme masku pro hodnoty nad thresholdem
    day_mask = targets > threshold
    
    # Pokud v celém batchi/horizontu není žádné denní světlo (např. specifický testovací segment)
    if not day_mask.any():
        return 0.0
        
    absolute_errors = torch.abs(targets - pred_median)
    
    # Vybereme pouze prvky, které odpovídají masce a spočítáme z nich průměr
    return absolute_errors[day_mask].mean().item()


def compute_rmse(pred_median: torch.Tensor, targets: torch.Tensor) -> float:
    """Computes Root Mean Squared Error for the median prediction."""
    return torch.sqrt(torch.mean((targets - pred_median) ** 2)).item()


def compute_picp(pred_low: torch.Tensor, pred_high: torch.Tensor, targets: torch.Tensor) -> float:
    """Computes Prediction Interval Coverage Probability."""
    inside_interval = (targets >= pred_low) & (targets <= pred_high)
    return inside_interval.float().mean().item()


def compute_ace(picp: float, nominal_coverage: float = 0.80) -> float:
    """Computes Average Coverage Error."""
    return picp - nominal_coverage


def compute_sharpness(pred_low: torch.Tensor, pred_high: torch.Tensor) -> float:
    """Computes Sharpness (average interval width)."""
    return (pred_high - pred_low).mean().item()


def compute_all_metrics(
    preds: torch.Tensor, 
    targets: torch.Tensor, 
    quantiles: list[float]
) -> dict[str, float]:
    """
    Orchestrates the computation of all individual metrics and returns them as a dictionary.
    """
    metrics = {}
    
    if targets.dim() == 2:
        targets = targets.unsqueeze(-1)
    
    with torch.no_grad():
        # --- Deterministic Metrics (q=0.5) ---
        if 0.5 in quantiles:
            median_idx = quantiles.index(0.5)
            pred_median = preds[:, :, median_idx : median_idx + 1]
            
            metrics["mae"] = compute_mae(pred_median, targets)
            metrics["mae_day"] = compute_masked_mae(pred_median, targets, threshold=0.01)  # ZMĚNA
            metrics["rmse"] = compute_rmse(pred_median, targets)

        # --- Probabilistic Metrics (80% Interval: q=0.1 to q=0.9) ---
        if 0.1 in quantiles and 0.9 in quantiles:
            low_idx = quantiles.index(0.1)
            high_idx = quantiles.index(0.9)
            
            pred_low = preds[:, :, low_idx : low_idx + 1]
            pred_high = preds[:, :, high_idx : high_idx + 1]
            
            picp = compute_picp(pred_low, pred_high, targets)
            
            metrics["picp_80"] = picp
            metrics["ace_80"] = compute_ace(picp, nominal_coverage=0.80)
            metrics["sharpness_80"] = compute_sharpness(pred_low, pred_high)
            
    return metrics