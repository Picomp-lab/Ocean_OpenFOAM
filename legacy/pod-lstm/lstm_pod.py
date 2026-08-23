"""
LSTM for POD Coefficient Prediction
=====================================
Loads POD results, builds sliding-window sequences,
trains LSTM to predict next-step POD coefficients.

Usage:
    python lstm_pod.py --pod_dir $OCEAN_DATA/pod_results --output ./lstm_results

Modes:
    - Single-step (teacher forcing): always uses ground truth as input
    - Autoregressive: feeds own predictions back as input

Pipeline:
    1. Load POD coeffs for alpha.water, p_rgh, Ux, Uz
    2. Truncate to 90% energy modes per variable
    3. Discard first 400 steps (0-20s transient), use steps 400-1000
    4. StandardScaler normalization
    5. Sliding window (W=20) -> (input, target) pairs
    6. Train/test split: first 480 train, last 120 test
    7. LSTM training with early stopping
    8. Evaluate: single-step + autoregressive
    9. Reconstruct flow fields and compute errors
"""

import os
import argparse
import time
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import joblib
import wandb


# ============================================================
# Configuration
# ============================================================

# 90% energy mode counts from POD results
MODE_COUNTS = {
    'alpha_water': 51,
    # 'p_rgh': ,
    'Ux': 67,
    'Uz': 100,
}
TOTAL_MODES = sum(MODE_COUNTS.values())  # 85

# VARIABLES = ['alpha_water', 'p_rgh', 'Ux', 'Uz']
VARIABLES = ['alpha_water', 'Ux', 'Uz']

# Data params
TRANSIENT_STEPS = 200     # discard first 200 steps (0-10s)
WINDOW_SIZE = 20          # input window length
TRAIN_STEPS = 700         # training steps (10-45s)
TEST_STEPS = 100          # test steps (45-50s)

# LSTM params
HIDDEN_SIZE = 128
NUM_LAYERS = 3
DROPOUT = 0.3

# Training params
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
MAX_EPOCHS = 300
PATIENCE = 30             # early stopping patience


# ============================================================
# Dataset
# ============================================================

class WindowDataset(Dataset):
    """Sliding window dataset for time series prediction."""

    def __init__(self, data, window_size):
        """
        Args:
            data: (T, D) array of normalized POD coefficients
            window_size: number of past steps as input
        """
        self.data = torch.FloatTensor(data)
        self.window_size = window_size

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.window_size]       # (W, D)
        y = self.data[idx + self.window_size]              # (D,)
        return x, y


# ============================================================
# Model
# ============================================================

class PODLSTM(nn.Module):
    """LSTM for predicting next-step POD coefficients."""

    def __init__(self, input_dim, hidden_size, num_layers, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, input_dim)

    def forward(self, x):
        # x: (batch, window, input_dim)
        lstm_out, _ = self.lstm(x)           # (batch, window, hidden)
        last_hidden = lstm_out[:, -1, :]     # (batch, hidden)
        delta = self.fc(last_hidden)           # (batch, input_dim)
        out = x[:, -1, :] + delta
        return out


# ============================================================
# Data preparation
# ============================================================

def load_and_prepare_data(pod_dir):
    """Load POD coefficients, truncate, combine, normalize."""

    print("Loading POD coefficients...")
    all_coeffs = []
    var_info = {}

    for var in VARIABLES:
        coeffs_path = os.path.join(pod_dir, f'{var}_coeffs.npy')
        coeffs = np.load(coeffs_path)  # (1000, n_components)
        n_modes = MODE_COUNTS[var]
        coeffs_truncated = coeffs[:, :n_modes]  # (1000, n_modes)
        var_info[var] = {
            'n_modes': n_modes,
            'start_col': sum(MODE_COUNTS[v] for v in VARIABLES[:VARIABLES.index(var)]),
        }
        all_coeffs.append(coeffs_truncated)
        print(f"  {var}: loaded {coeffs.shape[1]} modes, truncated to {n_modes}")

    # Concatenate: (1000, 85)
    coeffs_combined = np.hstack(all_coeffs)
    print(f"  Combined shape: {coeffs_combined.shape}")

    # Discard transient (first 400 steps)
    coeffs_steady = coeffs_combined[TRANSIENT_STEPS:]  # (600, 85)
    print(f"  After discarding transient ({TRANSIENT_STEPS} steps): {coeffs_steady.shape}")

    # Normalize
    scaler = StandardScaler()
    coeffs_normalized = scaler.fit_transform(coeffs_steady)  # (600, 85)

    # Split train/test (before windowing)
    train_data = coeffs_normalized[:TRAIN_STEPS]    # (480, 85)
    test_data = coeffs_normalized[TRAIN_STEPS:]     # (120, 85)

    print(f"  Train: {train_data.shape}, Test: {test_data.shape}")

    return train_data, test_data, scaler, var_info


def create_dataloaders(train_data, test_data, window_size, batch_size):
    """Create PyTorch DataLoaders."""
    train_dataset = WindowDataset(train_data, window_size)
    test_dataset = WindowDataset(test_data, window_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, drop_last=False)

    print(f"  Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    return train_loader, test_loader


# ============================================================
# Training
# ============================================================

def train_model(model, train_loader, test_loader, device, output_dir,
                lr=1e-3, max_epochs=300, patience=30):
    """Train LSTM with early stopping."""

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode='min', factor=0.5, patience=10
    # )
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, T_max=max_epochs, eta_min=1e-6
    # )
    warmup_epochs = 10
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs - warmup_epochs, eta_min=1e-6
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_epoch = 0
    train_losses = []
    val_losses = []
    no_improve = 0

    print(f"\nTraining started (max {max_epochs} epochs, patience {patience})...")
    t0 = time.time()

    for epoch in range(max_epochs):
        # --- Train ---
        model.train()
        epoch_train_loss = 0.0
        n_train_batches = 0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_train_loss += loss.item()
            n_train_batches += 1

        avg_train_loss = epoch_train_loss / n_train_batches
        train_losses.append(avg_train_loss)

        # --- Validate ---
        model.eval()
        epoch_val_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                pred = model(x_batch)
                loss = criterion(pred, y_batch)
                epoch_val_loss += loss.item()
                n_val_batches += 1

        avg_val_loss = epoch_val_loss / n_val_batches
        val_losses.append(avg_val_loss)

        # scheduler.step(avg_val_loss)
        scheduler.step()

        # Log to wandb
        current_lr = optimizer.param_groups[0]['lr']
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'best_val_loss': best_val_loss,
            'learning_rate': current_lr,
        })

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(),
                       os.path.join(output_dir, 'best_model.pt'))
        else:
            no_improve += 1

        if (epoch + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch+1:4d} | "
                  f"Train: {avg_train_loss:.6f} | "
                  f"Val: {avg_val_loss:.6f} | "
                  f"Best: {best_val_loss:.6f} (ep {best_epoch+1}) | "
                  f"Time: {elapsed:.1f}s")

        if no_improve >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    total_time = time.time() - t0
    print(f"Training complete in {total_time:.1f}s, best epoch: {best_epoch+1}")

    # Load best model
    model.load_state_dict(
        torch.load(os.path.join(output_dir, 'best_model.pt'),
                    weights_only=True)
    )

    return train_losses, val_losses, best_epoch


# ============================================================
# Evaluation
# ============================================================

def evaluate_single_step(model, test_data, window_size, device):
    """Single-step prediction using ground truth inputs."""
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for i in range(len(test_data) - window_size):
            x = torch.FloatTensor(test_data[i:i+window_size]).unsqueeze(0).to(device)
            pred = model(x).cpu().numpy()[0]
            predictions.append(pred)
            targets.append(test_data[i + window_size])

    predictions = np.array(predictions)  # (N_test - W, 85)
    targets = np.array(targets)
    return predictions, targets


def evaluate_autoregressive(model, test_data, window_size, device, n_steps=None):
    """Autoregressive prediction: feed own predictions back as input."""
    model.eval()
    if n_steps is None:
        n_steps = len(test_data) - window_size

    # Start with first window of ground truth
    buffer = list(test_data[:window_size])
    predictions = []
    targets = []

    with torch.no_grad():
        for i in range(n_steps):
            window = np.array(buffer[-window_size:])
            x = torch.FloatTensor(window).unsqueeze(0).to(device)
            pred = model(x).cpu().numpy()[0]
            predictions.append(pred)
            buffer.append(pred)
            if i + window_size < len(test_data):
                targets.append(test_data[i + window_size])

    predictions = np.array(predictions)
    targets = np.array(targets[:len(predictions)])
    return predictions, targets


def compute_errors(predictions, targets, scaler, var_info):
    """Compute per-variable RMSE in original scale."""
    # Inverse transform to original scale
    pred_orig = scaler.inverse_transform(predictions)
    tgt_orig = scaler.inverse_transform(targets)

    errors = {}
    for var in VARIABLES:
        start = var_info[var]['start_col']
        n = var_info[var]['n_modes']
        pred_var = pred_orig[:, start:start+n]
        tgt_var = tgt_orig[:, start:start+n]
        rmse = np.sqrt(np.mean((pred_var - tgt_var) ** 2))
        rel_err = rmse / (np.std(tgt_var) + 1e-10)
        errors[var] = {'rmse': float(rmse), 'relative': float(rel_err)}

    # Overall
    overall_rmse = np.sqrt(np.mean((pred_orig - tgt_orig) ** 2))
    errors['overall'] = {'rmse': float(overall_rmse)}

    return errors


# ============================================================
# Plotting
# ============================================================

def plot_training_curves(train_losses, val_losses, best_epoch, output_dir):
    """Plot training and validation loss curves."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.semilogy(train_losses, label='Train', alpha=0.8)
    ax.semilogy(val_losses, label='Validation', alpha=0.8)
    ax.axvline(best_epoch, color='r', linestyle='--', alpha=0.5,
               label=f'Best (ep {best_epoch+1})')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Training Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Training curves saved: {path}")


def plot_coefficient_predictions(predictions, targets, scaler, var_info,
                                 output_dir, prefix='singlestep'):
    """Plot predicted vs true POD coefficients for first few modes of each variable."""
    pred_orig = scaler.inverse_transform(predictions)
    tgt_orig = scaler.inverse_transform(targets)

    n_vars = len(VARIABLES)
    fig, axes = plt.subplots(n_vars, 3, figsize=(15, 3 * n_vars))

    for row, var in enumerate(VARIABLES):
        start = var_info[var]['start_col']
        n_modes = min(var_info[var]['n_modes'], 3)  # plot first 3 modes

        for col in range(3):
            ax = axes[row, col]
            if col < n_modes:
                mode_idx = start + col
                ax.plot(tgt_orig[:, mode_idx], 'b-', label='True', alpha=0.7, linewidth=1)
                ax.plot(pred_orig[:, mode_idx], 'r--', label='Pred', alpha=0.7, linewidth=1)
                ax.set_title(f'{var} mode {col+1}')
                ax.legend(fontsize=7)
            else:
                ax.axis('off')
            ax.grid(True, alpha=0.3)

    plt.suptitle(f'{prefix} prediction', fontsize=14)
    plt.tight_layout()
    path = os.path.join(output_dir, f'{prefix}_coefficients.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Coefficient plot saved: {path}")


def plot_autoregressive_comparison(pred_ss, pred_ar, targets, scaler,
                                   var_info, output_dir):
    """Compare single-step vs autoregressive for first mode of each variable."""
    tgt_orig = scaler.inverse_transform(targets)
    pred_ss_orig = scaler.inverse_transform(pred_ss[:len(targets)])
    pred_ar_orig = scaler.inverse_transform(pred_ar[:len(targets)])

    n_vars = len(VARIABLES)
    fig, axes = plt.subplots(n_vars, 1, figsize=(12, 3 * n_vars))

    for row, var in enumerate(VARIABLES):
        ax = axes[row]
        start = var_info[var]['start_col']
        ax.plot(tgt_orig[:, start], 'b-', label='True', alpha=0.8, linewidth=1)
        ax.plot(pred_ss_orig[:, start], 'g--', label='Single-step', alpha=0.7, linewidth=1)
        ax.plot(pred_ar_orig[:, start], 'r--', label='Autoregressive', alpha=0.7, linewidth=1)
        ax.set_title(f'{var} mode 1')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Single-step vs Autoregressive', fontsize=14)
    plt.tight_layout()
    path = os.path.join(output_dir, 'ss_vs_ar_comparison.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Comparison plot saved: {path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='LSTM for POD coefficient prediction')
    parser.add_argument('--pod_dir', type=str, required=True,
                        help='Directory containing POD .npy results')
    parser.add_argument('--output', type=str, default='./lstm_results',
                        help='Output directory')
    parser.add_argument('--window', type=int, default=WINDOW_SIZE,
                        help=f'Input window size (default: {WINDOW_SIZE})')
    parser.add_argument('--hidden', type=int, default=HIDDEN_SIZE,
                        help=f'LSTM hidden size (default: {HIDDEN_SIZE})')
    parser.add_argument('--epochs', type=int, default=MAX_EPOCHS,
                        help=f'Max training epochs (default: {MAX_EPOCHS})')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE,
                        help=f'Learning rate (default: {LEARNING_RATE})')
    parser.add_argument('--batch', type=int, default=BATCH_SIZE,
                        help=f'Batch size (default: {BATCH_SIZE})')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("=" * 60)
    print("LSTM for POD Coefficient Prediction")
    print("=" * 60)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Initialize wandb
    wandb.init(
        project='pod-lstm',
        config={
            'window_size': args.window,
            'hidden_size': args.hidden,
            'num_layers': NUM_LAYERS,
            'dropout': DROPOUT,
            'batch_size': args.batch,
            'learning_rate': args.lr,
            'max_epochs': args.epochs,
            'patience': PATIENCE,
            'total_modes': TOTAL_MODES,
            'mode_counts': MODE_COUNTS,
            'transient_steps': TRANSIENT_STEPS,
            'train_steps': TRAIN_STEPS,
            'test_steps': TEST_STEPS,
            'optimizer': 'AdamW',
            'weight_decay': 1e-4,
            'device': str(device),
        }
    )

    # 1. Load and prepare data
    train_data, test_data, scaler, var_info = load_and_prepare_data(args.pod_dir)

    # Save scaler for later use
    joblib.dump(scaler, os.path.join(args.output, 'scaler.pkl'))
    with open(os.path.join(args.output, 'var_info.json'), 'w') as f:
        json.dump(var_info, f, indent=2)

    # 2. Create dataloaders
    train_loader, test_loader = create_dataloaders(
        train_data, test_data, args.window, args.batch
    )

    # 3. Build model
    model = PODLSTM(
        input_dim=TOTAL_MODES,
        hidden_size=args.hidden,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {NUM_LAYERS}-layer LSTM, hidden={args.hidden}, "
          f"input/output={TOTAL_MODES}, params={n_params:,}")

    # 4. Train
    train_losses, val_losses, best_epoch = train_model(
        model, train_loader, test_loader, device, args.output,
        lr=args.lr, max_epochs=args.epochs, patience=PATIENCE
    )

    # 5. Plot training curves
    plot_training_curves(train_losses, val_losses, best_epoch, args.output)

    # 6. Single-step evaluation
    print("\n--- Single-step evaluation ---")
    pred_ss, tgt_ss = evaluate_single_step(model, test_data, args.window, device)
    errors_ss = compute_errors(pred_ss, tgt_ss, scaler, var_info)
    print("Single-step errors:")
    for var, err in errors_ss.items():
        if var == 'overall':
            print(f"  {var}: RMSE = {err['rmse']:.6f}")
        else:
            print(f"  {var}: RMSE = {err['rmse']:.6f}, relative = {err['relative']:.4f}")

    plot_coefficient_predictions(pred_ss, tgt_ss, scaler, var_info,
                                 args.output, prefix='singlestep')

    # 7. Autoregressive evaluation
    print("\n--- Autoregressive evaluation ---")
    pred_ar, tgt_ar = evaluate_autoregressive(model, test_data, args.window, device)
    errors_ar = compute_errors(pred_ar, tgt_ar, scaler, var_info)
    print("Autoregressive errors:")
    for var, err in errors_ar.items():
        if var == 'overall':
            print(f"  {var}: RMSE = {err['rmse']:.6f}")
        else:
            print(f"  {var}: RMSE = {err['rmse']:.6f}, relative = {err['relative']:.4f}")

    plot_coefficient_predictions(pred_ar, tgt_ar, scaler, var_info,
                                 args.output, prefix='autoregressive')

    # 8. Comparison plot
    min_len = min(len(pred_ss), len(pred_ar), len(tgt_ss))
    plot_autoregressive_comparison(
        pred_ss[:min_len], pred_ar[:min_len], tgt_ss[:min_len],
        scaler, var_info, args.output
    )

    # 9. Save all results
    np.save(os.path.join(args.output, 'pred_singlestep.npy'), pred_ss)
    np.save(os.path.join(args.output, 'pred_autoregressive.npy'), pred_ar)
    np.save(os.path.join(args.output, 'targets.npy'), tgt_ss)

    results_summary = {
        'config': {
            'window_size': args.window,
            'hidden_size': args.hidden,
            'num_layers': NUM_LAYERS,
            'total_modes': TOTAL_MODES,
            'mode_counts': MODE_COUNTS,
            'transient_steps': TRANSIENT_STEPS,
            'train_steps': TRAIN_STEPS,
            'test_steps': TEST_STEPS,
            'best_epoch': best_epoch + 1,
            'n_params': n_params,
            'device': str(device),
        },
        'errors_single_step': errors_ss,
        'errors_autoregressive': errors_ar,
    }
    with open(os.path.join(args.output, 'results_summary.json'), 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nResults summary saved: {os.path.join(args.output, 'results_summary.json')}")

    # Log final results and plots to wandb
    wandb.log({
        'ss_overall_rmse': errors_ss['overall']['rmse'],
        'ar_overall_rmse': errors_ar['overall']['rmse'],
    })
    for var in VARIABLES:
        wandb.log({
            f'ss_{var}_rmse': errors_ss[var]['rmse'],
            f'ss_{var}_relative': errors_ss[var]['relative'],
            f'ar_{var}_rmse': errors_ar[var]['rmse'],
            f'ar_{var}_relative': errors_ar[var]['relative'],
        })

    # Upload plots to wandb
    for img_name in ['training_curves.png', 'singlestep_coefficients.png',
                     'autoregressive_coefficients.png', 'ss_vs_ar_comparison.png']:
        img_path = os.path.join(args.output, img_name)
        if os.path.exists(img_path):
            wandb.log({img_name.replace('.png', ''): wandb.Image(img_path)})

    wandb.finish()

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == '__main__':
    main()