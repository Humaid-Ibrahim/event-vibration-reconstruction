"""
Unified trainer for all model types.
"""

import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from fastdtw import fastdtw
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from .losses import spectral_loss, envelope_loss, ssim_loss
from .soft_dtw import soft_dtw_loss


def compute_dtw_distance(pred, target, radius=10):
    """
    Compute average FastDTW distance between predicted and target signals.

    Args:
        pred: Predictions array (N, T)
        target: Targets array (N, T)
        radius: Constraint on warping path (smaller = faster but less accurate)

    Returns:
        Average DTW distance (lower is better)
    """
    distances = []
    n_samples = min(pred.shape[0], 50)  # Limit samples for speed

    def scalar_dist(a, b):
        return abs(a - b)

    for i in range(n_samples):
        p = pred[i][::4]  # Subsample by 4x for speed
        t = target[i][::4]
        distance, _ = fastdtw(p, t, radius=radius, dist=scalar_dist)
        distances.append(distance / len(p))

    return float(np.mean(distances))


class Trainer:
    """
    Unified trainer for event-to-vibration models.

    Handles:
        - Scaling (input/output normalization)
        - Training loop with combined loss functions
        - Validation with metrics computation
        - Model checkpointing
    """

    def __init__(
        self,
        model,
        device=None,
        use_scaler=True,
        mse_lambda=1.0,
        smooth_lambda=0.0,
        spectral_lambda=0.0,
        envelope_lambda=0.0,
        ssim_lambda=0.0,
        sdtw_lambda=0.0,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.use_scaler = use_scaler

        # Loss weights
        self.mse_lambda = mse_lambda
        self.smooth_lambda = smooth_lambda
        self.spectral_lambda = spectral_lambda
        self.envelope_lambda = envelope_lambda
        self.ssim_lambda = ssim_lambda
        self.sdtw_lambda = sdtw_lambda

        self.train_losses = []
        self.val_losses = []
        self.val_r2 = []

        self.x_scaler = None
        self.y_scaler = None
        self.criterion = nn.MSELoss()

    def _compute_loss(self, y_pred, y_target):
        """Compute combined loss using all configured loss functions."""
        loss = self.mse_lambda * self.criterion(y_pred, y_target)

        if self.spectral_lambda > 0:
            loss = loss + self.spectral_lambda * spectral_loss(y_pred, y_target)

        if self.envelope_lambda > 0:
            loss = loss + self.envelope_lambda * envelope_loss(y_pred, y_target)

        if self.ssim_lambda > 0:
            loss = loss + self.ssim_lambda * ssim_loss(y_pred, y_target)

        if self.sdtw_lambda > 0:
            loss = loss + self.sdtw_lambda * soft_dtw_loss(y_pred, y_target, gamma=0.1, normalize=True)

        if self.smooth_lambda > 0:
            smoothness = (y_pred[:, 1:] - y_pred[:, :-1]).pow(2).mean()
            loss = loss + self.smooth_lambda * smoothness

        return loss

    def fit_scalers(self, train_loader, scaler_path="results/scalers.pkl"):
        """Fit StandardScalers on training data.

        Note: For high-dimensional inputs like image_sequence (T, C, H, W),
        StandardScaler would require billions of features. Instead, we skip
        x scaling for such cases and only scale the output y.
        """
        if not self.use_scaler:
            return

        xs, ys = [], []

        with torch.no_grad():
            for batch in tqdm(train_loader, desc="Fitting scalers"):
                x, y = batch[0], batch[1]
                xs.append(x.view(x.size(0), -1).cpu().numpy())
                # Handle both (B, T) and (B, T, 2) vibration formats
                if y.dim() == 3:
                    y = y[:, :, 1]  # Take velocity channel
                ys.append(y.cpu().numpy())

        X = np.vstack(xs)
        Y = np.vstack(ys)

        # Check if input dimensionality is reasonable for StandardScaler
        # For image_sequence with shape (T, C, H, W), flattened size can be millions
        # StandardScaler would need to store mean/std for each feature
        # 150k allows 5x5 grid with polarity (102,400) and 6x6 grid (147,456)
        n_features = X.shape[1]
        if n_features > 150000:  # More than 150k features - skip x scaling
            print(f"WARNING: Input has {n_features:,} features - too large for StandardScaler")
            print("         Skipping input (x) scaling, only scaling output (y)")
            self.x_scaler = None
        else:
            self.x_scaler = StandardScaler().fit(X)

        self.y_scaler = StandardScaler().fit(Y)

        Path(scaler_path).parent.mkdir(exist_ok=True, parents=True)
        with open(scaler_path, "wb") as f:
            pickle.dump({"x_scaler": self.x_scaler, "y_scaler": self.y_scaler}, f)

    def _scale_x(self, x):
        if not self.use_scaler or self.x_scaler is None:
            return x
        B = x.size(0)
        flat = x.view(B, -1).cpu().numpy()
        scaled = self.x_scaler.transform(flat)
        return torch.from_numpy(scaled).to(x.device, dtype=x.dtype).view_as(x)

    def _scale_y(self, y):
        if not self.use_scaler or self.y_scaler is None:
            return y
        y_np = y.cpu().numpy()
        scaled = self.y_scaler.transform(y_np)
        return torch.from_numpy(scaled).to(y.device, dtype=y.dtype)

    def _inv_scale_y(self, y):
        if not self.use_scaler or self.y_scaler is None:
            return y
        y_np = y.detach().cpu().numpy()
        inv = self.y_scaler.inverse_transform(y_np)
        return torch.from_numpy(inv).to(y.device, dtype=y.dtype)

    def _extract_y(self, y):
        """Extract vibration signal from batch (handles different formats)."""
        if y.dim() == 3:
            return y[:, :, 1]  # (B, T, 2) -> (B, T) velocity channel
        return y  # (B, T) already

    def train_epoch(self, loader, optimizer):
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0

        for batch in tqdm(loader, desc="Training", leave=False):
            x, y = batch[0], batch[1]
            x = x.to(self.device, non_blocking=True).float()
            y = self._extract_y(y).to(self.device, non_blocking=True).float()

            x = self._scale_x(x)
            y_scaled = self._scale_y(y)

            optimizer.zero_grad()
            y_pred = self.model(x)
            loss = self._compute_loss(y_pred, y_scaled)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        return total_loss / max(len(loader), 1)

    def evaluate(self, loader):
        """Evaluate model and return metrics + predictions."""
        self.model.eval()

        total_loss = 0.0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for batch in tqdm(loader, desc="Evaluating", leave=False):
                x, y = batch[0], batch[1]
                x = x.to(self.device, non_blocking=True).float()
                y = self._extract_y(y).to(self.device, non_blocking=True).float()

                x = self._scale_x(x)
                y_scaled = self._scale_y(y)

                y_pred = self.model(x)
                loss = self._compute_loss(y_pred, y_scaled)
                total_loss += loss.item()

                y_pred_inv = self._inv_scale_y(y_pred)

                all_preds.append(y_pred_inv.cpu().numpy())
                all_targets.append(y.cpu().numpy())

        preds = np.concatenate(all_preds, axis=0)
        targets = np.concatenate(all_targets, axis=0)

        metrics = {
            "val_loss": total_loss / max(len(loader), 1),
            "r2": r2_score(targets.flatten(), preds.flatten()),
            "mse": mean_squared_error(targets.flatten(), preds.flatten()),
            "mae": mean_absolute_error(targets.flatten(), preds.flatten()),
            "mape": mean_absolute_percentage_error(targets.flatten(), preds.flatten()),
            "dtw": compute_dtw_distance(preds, targets),
        }

        return metrics, preds, targets

    def train(
        self,
        train_loader,
        val_loader,
        epochs=100,
        lr=1e-3,
        weight_decay=1e-3,
        checkpoint_path="results/best_model.pth",
        scaler_path="results/scalers.pkl",
        early_stopping_patience=15,
    ):
        """
        Full training loop with early stopping.

        Loss weights are configured in __init__.

        Args:
            epochs: Number of training epochs
            lr: Learning rate
            weight_decay: AdamW weight decay
            checkpoint_path: Path to save best model
            scaler_path: Path to save input/output scalers
            early_stopping_patience: Stop if no improvement for this many epochs (0 to disable)

        Returns:
            best_val_loss: Best validation loss achieved
        """
        Path(checkpoint_path).parent.mkdir(exist_ok=True, parents=True)

        if self.use_scaler:
            self.fit_scalers(train_loader, scaler_path)

        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_val_loss = float("inf")
        best_epoch = 0
        epochs_without_improvement = 0

        for epoch in range(epochs):
            epoch_start = time.time()

            train_loss = self.train_epoch(train_loader, optimizer)
            self.train_losses.append(train_loss)

            metrics, _, _ = self.evaluate(val_loader)
            self.val_losses.append(metrics["val_loss"])
            self.val_r2.append(metrics["r2"])

            scheduler.step()

            elapsed = time.time() - epoch_start

            print(
                f"Epoch {epoch+1:03d}/{epochs} [{elapsed:.1f}s] | "
                f"train_loss={train_loss:.6f} val_loss={metrics['val_loss']:.6f} "
                f"R2={metrics['r2']:.4f} MSE={metrics['mse']:.6f} DTW={metrics['dtw']:.4f}"
            )

            if metrics["val_loss"] < best_val_loss:
                best_val_loss = metrics["val_loss"]
                best_epoch = epoch + 1
                epochs_without_improvement = 0
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "metrics": metrics,
                }, checkpoint_path)
                print(f"  -> Saved best model (val_loss={best_val_loss:.6f})")
            else:
                epochs_without_improvement += 1

            if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
                print(f"\nEarly stopping: No improvement for {early_stopping_patience} epochs")
                print(f"Best model was at epoch {best_epoch} with val_loss={best_val_loss:.6f}")
                break

        return best_val_loss

    def load_checkpoint(self, checkpoint_path):
        """Load model from checkpoint."""
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        # Handle both old and new checkpoint formats
        if "model_state_dict" in ckpt:
            self.model.load_state_dict(ckpt["model_state_dict"])
        elif "model" in ckpt:
            self.model.load_state_dict(ckpt["model"])
        self.model.to(self.device)
        return ckpt.get("metrics", {})
