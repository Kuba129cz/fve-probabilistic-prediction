import torch

class MultiQuantileLoss(torch.nn.Module):
    """
    Computes the Mean Pinball Loss across multiple quantiles.
    """
    def __init__(self, quantiles: list[float]):
        super().__init__()
        self.quantiles = quantiles
    
    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        preds:   Tensor of shape [batch_size, horizon, num_quantiles]
        targets: Tensor of shape [batch_size, horizon, 1]
        """
        total_loss = 0.0

        for i, q in enumerate(self.quantiles):
            preds_q = preds[:, :, i: i+1]
            error = targets - preds_q
            quantile_loss = torch.max(q*error, (q - 1 * error))
            total_loss += quantile_loss.mean()
        
        return total_loss / len(self.quantiles)