"""
Training Script for Transolver++ Wave Prediction (Hydra)
=========================================================
Usage:
    python train.py                                    # default config
    python train.py model.n_hidden=256 training.lr=5e-4  # override
    python train.py --multirun model.n_layers=4,6,8       # sweep
"""

import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import hydra
from omegaconf import DictConfig, OmegaConf

from transolver_pp import TransolverPP, count_parameters


# ============================================================
# Normalizer
# ============================================================
class Normalizer:
    """Mean/std normalizer. Computes stats per feature (last dim)."""
    def __init__(self, data, eps=1e-6):
        self.mean = data.reshape(-1, data.shape[-1]).mean(dim=0)
        self.std = data.reshape(-1, data.shape[-1]).std(dim=0) + eps

    def encode(self, x):
        return (x - self.mean) / self.std

    def decode(self, x):
        return x * self.std + self.mean

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self


# ============================================================
# Dataset
# ============================================================
class WaveWindowDataset(Dataset):
    def __init__(self, fields, window=5, out_dim=4):
        self.window = window
        self.fields = fields
        self.out_dim = out_dim

    def __len__(self):
        return len(self.fields) - self.window

    def __getitem__(self, idx):
        inp = self.fields[idx:idx+self.window]              # (W, N, C)
        tgt = self.fields[idx+self.window, :, :self.out_dim] # (N, out_dim)
        return inp, tgt


# ============================================================
# Chunked forward pass
# ============================================================
@torch.no_grad()
def chunked_forward(model, x_full, fx_full, chunk_size=30000):
    B, N, _ = x_full.shape
    if N <= chunk_size:
        return model(x_full, fx_full)

    outputs = []
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        out_chunk = model(x_full[:, start:end, :], fx_full[:, start:end, :])
        outputs.append(out_chunk)
    return torch.cat(outputs, dim=1)


# ============================================================
# Training
# ============================================================
def train_one_epoch(model, loader, coords, optimizer, scheduler, criterion,
                    device, input_normalizer, target_normalizer, coord_normalizer,
                    max_grad_norm=1.0, subset_size=None):
    model.train()
    total_loss = 0.0

    x_norm = coord_normalizer.encode(coords).unsqueeze(0)

    for batch_idx, (inp, tgt) in enumerate(loader):
        inp = inp.to(device)
        tgt = tgt.to(device)
        B = inp.shape[0]

        inp_norm = input_normalizer.encode(inp)
        tgt_norm = target_normalizer.encode(tgt)
        x_batch = x_norm.expand(B, -1, -1)

        if subset_size is not None and subset_size < coords.shape[0]:
            idx = torch.randperm(coords.shape[0], device=device)[:subset_size]
            x_batch = x_batch[:, idx, :]
            inp_norm = inp_norm[:, idx, :]
            tgt_norm = tgt_norm[:, idx, :]

        optimizer.zero_grad()
        out = model(x_batch, inp_norm)
        loss = criterion(out, tgt_norm)
        loss.backward()

        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, coords, device, input_normalizer, target_normalizer,
             coord_normalizer, out_dim, chunk_size=30000):
    model.eval()
    criterion = nn.MSELoss(reduction='none')

    x_norm = coord_normalizer.encode(coords).unsqueeze(0)
    total_loss = 0.0
    per_field = np.zeros(out_dim)
    count = 0

    for inp, tgt in loader:
        inp = inp.to(device)
        tgt = tgt.to(device)
        B = inp.shape[0]

        inp_norm = input_normalizer.encode(inp)
        tgt_norm = target_normalizer.encode(tgt)
        x_batch = x_norm.expand(B, -1, -1)

        out = chunked_forward(model, x_batch, inp_norm, chunk_size)
        loss = criterion(out, tgt_norm)

        total_loss += loss.mean().item() * B
        per_field += loss.mean(dim=(0, 1)).cpu().numpy() * B
        count += B

    return total_loss / count, per_field / count


@torch.no_grad()
def autoregressive_rollout(model, coords, init_field, n_steps, device,
                           coord_normalizer, input_normalizer, target_normalizer,
                           in_dim, out_dim, chunk_size=30000):
    """
    Autoregressive rollout.
    init_field: (N, in_dim) un-normalized
    
    At each step:
      - model predicts out_dim channels (alpha, Ux, Uz, p_rgh)
      - nut is recomputed from predicted U for next input
    """
    model.eval()
    x_norm = coord_normalizer.encode(coords).unsqueeze(0)

    rollout = [init_field[:, :out_dim].cpu().numpy()]  # store only output channels
    current = init_field.unsqueeze(0).to(device)       # (1, N, in_dim)

    for step in range(n_steps):
        current_norm = input_normalizer.encode(current)
        out_norm = chunked_forward(model, x_norm, current_norm, chunk_size)
        out = target_normalizer.decode(out_norm)  # (1, N, out_dim)

        # Clamp alpha to [0, 1]
        out[:, :, 0] = out[:, :, 0].clamp(0.0, 1.0)
        rollout.append(out[0].cpu().numpy())

        # Build next input: predicted fields + recompute nut from predicted U
        # For now, carry forward nut from current step
        # TODO: compute nut = (Cs*delta)^2 * |S| from predicted U
        if in_dim > out_dim:
            next_input = torch.cat([out, current[:, :, out_dim:]], dim=-1)
        else:
            next_input = out
        current = next_input

        if (step + 1) % 20 == 0:
            print(f"    Rollout step {step+1}/{n_steps}")

    return np.stack(rollout, axis=0)


# ============================================================
# Main
# ============================================================
@hydra.main(config_path="configs", config_name="default", version_base="1.3")
def main(cfg: DictConfig):
    print("-" * 20 + " Configuration " + "-" * 20)
    print(OmegaConf.to_yaml(cfg))
    print("-" * 55)

    # Resolve output dir (Hydra changes cwd, so use absolute path)
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    print(f"Output: {output_dir}")

    # Device
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device not found. This script requires a GPU.")
    device = torch.device('cuda')
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------
    print("\nLoading data...")
    data_dir = cfg.data.data_dir
    coords = np.load(os.path.join(data_dir, 'coords_2d.npy'))
    fields = np.load(os.path.join(data_dir, 'fields.npy'))
    times = np.load(os.path.join(data_dir, 'times.npy'))

    N = coords.shape[0]
    print(f"  Points: {N}, Timesteps: {len(times)}, Channels: {fields.shape[-1]}")
    print(f"  Time range: [{times[0]:.2f}, {times[-1]:.2f}]s")

    # Split by time
    train_mask = (times >= cfg.data.train_start) & (times < cfg.data.train_end)
    test_mask = (times >= cfg.data.test_start) & (times <= cfg.data.test_end)
    train_indices = np.where(train_mask)[0]
    test_indices = np.where(test_mask)[0]

    print(f"  Train: {len(train_indices)} steps ({cfg.data.train_start}-{cfg.data.train_end}s)")
    print(f"  Test:  {len(test_indices)} steps ({cfg.data.test_start}-{cfg.data.test_end}s)")

    # To tensors
    coords_tensor = torch.tensor(coords, dtype=torch.float32).to(device)
    train_fields = torch.tensor(fields[train_indices], dtype=torch.float32)
    test_fields = torch.tensor(fields[test_indices], dtype=torch.float32)

    # Normalizers
    # Input normalizer: all in_dim channels
    input_normalizer = Normalizer(train_fields).to(device)
    # Target normalizer: only first out_dim channels
    target_normalizer = Normalizer(train_fields[:, :, :cfg.model.out_dim]).to(device)
    coord_normalizer = Normalizer(coords_tensor).to(device)

    print(f"\n  Coord normalizer:")
    coord_names = ['x', 'z']
    for j, name in enumerate(coord_names):
        print(f"    {name}: mean={coord_normalizer.mean[j]:.4f}, "
              f"std={coord_normalizer.std[j]:.4f}")

    print(f"  Input normalizer ({cfg.model.in_dim} channels):")
    for j, name in enumerate(cfg.input_fields):
        print(f"    {name}: mean={input_normalizer.mean[j]:.4f}, "
              f"std={input_normalizer.std[j]:.4f}")

    print(f"  Target normalizer ({cfg.model.out_dim} channels):")
    for j, name in enumerate(cfg.output_fields):
        print(f"    {name}: mean={target_normalizer.mean[j]:.4f}, "
              f"std={target_normalizer.std[j]:.4f}")

    # Datasets & loaders
    train_dataset = WaveStepDataset(train_fields, out_dim=cfg.model.out_dim)
    test_dataset = WaveStepDataset(test_fields, out_dim=cfg.model.out_dim)
    print(f"\n  Train pairs: {len(train_dataset)}")
    print(f"  Test pairs:  {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size,
                              shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.training.batch_size,
                             shuffle=False, num_workers=0, pin_memory=True)

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------
    in_dim = cfg.model.window * cfg.model.channels

    model = TransolverPP(
        space_dim=cfg.model.space_dim,
        window=cfg.model.window,
        channels=cfg.model.channels,
        n_layers=cfg.model.n_layers,
        n_hidden=cfg.model.n_hidden,
        dropout=cfg.model.dropout,
        n_head=cfg.model.n_heads,
        in_dim=cfg.model.in_dim,
        out_dim=cfg.model.out_dim,
        slice_num=cfg.model.slice_num,
        mlp_ratio=cfg.model.mlp_ratio,
        use_checkpoint=cfg.model.use_checkpoint,
    ).to(device)

    n_params = count_parameters(model)
    print(f"\nModel: TransolverPP")
    print(f"  Input:  {cfg.model.in_dim} channels → Output: {cfg.model.out_dim} channels")
    print(f"  Layers={cfg.model.n_layers}, Hidden={cfg.model.n_hidden}, "
          f"Heads={cfg.model.n_heads}, Slices={cfg.model.slice_num}")
    print(f"  Parameters: {n_params:,}")

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr,
                                  weight_decay=cfg.training.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.training.lr,
        epochs=cfg.training.epochs, steps_per_epoch=len(train_loader),
        final_div_factor=1000.
    )
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []

    print(f"\nTraining for {cfg.training.epochs} epochs...")
    print(f"  Batch size: {cfg.training.batch_size}, LR: {cfg.training.lr}")
    if cfg.training.subset_size:
        print(f"  Subset: {cfg.training.subset_size} points/step")
    print(f"  Eval chunk: {cfg.training.chunk_size} points")
    t_start = time.time()

    for epoch in range(cfg.training.epochs):
        t_ep = time.time()

        train_loss = train_one_epoch(
            model, train_loader, coords_tensor, optimizer, scheduler,
            criterion, device, input_normalizer, target_normalizer,
            coord_normalizer,
            max_grad_norm=cfg.training.max_grad_norm,
            subset_size=cfg.training.subset_size
        )
        val_loss, per_field = evaluate(
            model, test_loader, coords_tensor, device,
            input_normalizer, target_normalizer, coord_normalizer,
            out_dim=cfg.model.out_dim,
            chunk_size=cfg.training.chunk_size
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': OmegaConf.to_container(cfg, resolve=True),
                'input_normalizer': {
                    'mean': input_normalizer.mean.cpu(),
                    'std': input_normalizer.std.cpu()
                },
                'target_normalizer': {
                    'mean': target_normalizer.mean.cpu(),
                    'std': target_normalizer.std.cpu()
                },
                'coord_normalizer': {
                    'mean': coord_normalizer.mean.cpu(),
                    'std': coord_normalizer.std.cpu()
                },
            }, os.path.join(output_dir, 'best_model.pt'))
        else:
            patience_counter += 1

        dt = time.time() - t_ep
        field_str = " ".join(
            f"{name}={per_field[j]:.6f}"
            for j, name in enumerate(cfg.output_fields)
        )
        print(f"  Epoch {epoch:3d}/{cfg.training.epochs} | "
              f"train={train_loss:.6f} val={val_loss:.6f} | "
              f"{field_str} | {dt:.1f}s"
              + (" *" if patience_counter == 0 else ""))

        if patience_counter >= cfg.training.patience:
            print(f"\n  Early stopping at epoch {epoch} (patience={cfg.training.patience})")
            break

    total_time = time.time() - t_start
    print(f"\nTraining done in {total_time:.0f}s. Best val: {best_val_loss:.6f}")

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)

    ckpt = torch.load(os.path.join(output_dir, 'best_model.pt'),
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])

    val_loss, per_field = evaluate(
        model, test_loader, coords_tensor, device,
        input_normalizer, target_normalizer, coord_normalizer,
        out_dim=cfg.model.out_dim,
        chunk_size=cfg.training.chunk_size
    )
    print(f"\nSingle-step test loss (normalized MSE): {val_loss:.6f}")
    for j, name in enumerate(cfg.output_fields):
        print(f"  {name}: {per_field[j]:.6f}")

    # Relative L2 error
    print(f"\nSingle-step relative L2 error:")
    model.eval()
    x_norm = coord_normalizer.encode(coords_tensor).unsqueeze(0)
    rel_errors = {n: [] for n in cfg.output_fields}
    with torch.no_grad():
        for inp, tgt in test_loader:
            inp, tgt = inp.to(device), tgt.to(device)
            B = inp.shape[0]
            inp_norm = input_normalizer.encode(inp)
            out_norm = chunked_forward(model, x_norm.expand(B, -1, -1),
                                       inp_norm, cfg.training.chunk_size)
            out = target_normalizer.decode(out_norm)
            for j, name in enumerate(cfg.output_fields):
                err = torch.norm(out[:, :, j] - tgt[:, :, j]) / \
                      (torch.norm(tgt[:, :, j]) + 1e-8)
                rel_errors[name].append(err.item())
    for name in cfg.output_fields:
        mean_err = np.mean(rel_errors[name])
        print(f"  {name}: {mean_err:.4f} ({mean_err*100:.2f}%)")

    # Autoregressive rollout
    print(f"\nAutoregressive rollout ({cfg.eval.ar_steps} steps)...")
    init_field = test_fields[0].to(device)
    rollout = autoregressive_rollout(
        model, coords_tensor, init_field, cfg.eval.ar_steps, device,
        coord_normalizer, input_normalizer, target_normalizer,
        in_dim=cfg.model.in_dim, out_dim=cfg.model.out_dim,
        chunk_size=cfg.training.chunk_size
    )
    np.save(os.path.join(output_dir, 'rollout_pred.npy'), rollout)

    gt_end = min(test_indices[0] + cfg.eval.ar_steps + 1, len(fields))
    gt_fields = fields[test_indices[0]:gt_end, :, :cfg.model.out_dim]
    np.save(os.path.join(output_dir, 'rollout_gt.npy'), gt_fields)

    # Rollout errors
    print(f"\nAutoregressive relative L2 error:")
    n_eval = min(cfg.eval.ar_steps, len(gt_fields) - 1)
    rollout_errors = []
    for step in range(1, n_eval + 1):
        err = np.linalg.norm(rollout[step] - gt_fields[step]) / \
              (np.linalg.norm(gt_fields[step]) + 1e-8)
        rollout_errors.append(err)

    for s in [1, 5, 10, 20, 50, 100]:
        if s <= len(rollout_errors):
            print(f"  Step {s:3d}: {rollout_errors[s-1]:.6f} "
                  f"({rollout_errors[s-1]*100:.2f}%)")

    # Save history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': float(best_val_loss),
        'best_epoch': int(ckpt['epoch']),
        'per_field_loss': per_field.tolist(),
        'rel_errors': {k: float(np.mean(v)) for k, v in rel_errors.items()},
        'rollout_errors': [float(e) for e in rollout_errors],
        'n_params': n_params,
        'total_time_s': total_time,
        'config': OmegaConf.to_container(cfg, resolve=True),
    }
    with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nAll results saved to {output_dir}/")


if __name__ == '__main__':
    main()
