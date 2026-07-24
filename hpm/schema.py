"""
Channel schema — single source of truth for channel configuration.

config.yaml declares `data.channels` (list of {name, update_rule, loss_weight,
alpha_weighted}); everything else — field_dim, out_dim, delta indices, alpha
index, disk column selection, stats-file signature — is DERIVED here.
No other file may hardcode channel counts or channel indices.

This module also owns `advance_window()`: the single shared rollout closure
(window shift + scatter of partial delta) used by BOTH training and inference,
so the two code paths can never drift apart.

update_rule semantics:
    delta    — channel is predicted by the delta head (residual learning)
    frozen   — channel is input-only; carried unchanged through rollout
               (Method B for the nut ablation). Receives no loss.
    flux_div — RESERVED for the E2 flux head (alpha update via face-flux
               divergence). Validated as a legal enum value but raises
               NotImplementedError until the flux head lands.
"""

from dataclasses import dataclass

import torch

# Fixed column order of chunk_*_data.npy files on disk. Data files are NEVER
# regenerated for ablations; channels are selected BY NAME at load time
# against this anchor.
DISK_CHANNELS = ["alpha", "Ux", "Uy", "Uz", "p_rgh", "nut"]

UPDATE_RULES = ("delta", "frozen", "flux_div")


@dataclass(frozen=True)
class ChannelSchema:
    names: tuple            # channel names, order == tensor channel order
    update_rules: tuple     # per-channel update rule (see module docstring)
    loss_weights: tuple     # per-channel loss weight (ignored for non-delta)
    alpha_weighted: tuple   # per-channel bool: multiply by alpha in physical space

    # ---------------- derived quantities (never declared, always computed) ----

    @property
    def field_dim(self):
        return len(self.names)

    @property
    def delta_indices(self):
        """Channel indices predicted by the delta head."""
        return [i for i, r in enumerate(self.update_rules) if r == "delta"]

    @property
    def out_dim(self):
        """Model head width = number of delta channels (fork-1b)."""
        return len(self.delta_indices)

    @property
    def alpha_idx(self):
        """Index of the alpha channel — looked up by name (fork-2)."""
        return self.names.index("alpha")

    @property
    def disk_indices(self):
        """Column indices into the on-disk 6-channel layout."""
        return [DISK_CHANNELS.index(n) for n in self.names]

    def delta_loss_weights(self):
        """Loss weights restricted to delta channels (loss covers only these)."""
        return [self.loss_weights[i] for i in self.delta_indices]

    def display_names(self):
        """Human-readable names; alpha-weighted channels get an 'α' prefix."""
        return [("α" + n) if w else n
                for n, w in zip(self.names, self.alpha_weighted)]

    def signature(self):
        """Channel signature for stats filenames. Alpha-weighted channels are
        marked with a 'w' suffix, e.g.  alpha.Uxw.Uyw.Uzw.p_rgh
        Guarantees different channel sets / weightings never collide on disk."""
        parts = [(n + "w") if w else n
                 for n, w in zip(self.names, self.alpha_weighted)]
        return ".".join(parts)

    def is_legacy_layout(self):
        """True iff this schema is exactly the historical 6-channel all-delta
        layout (used only to fall back to old stats_*_u{0|1}_nut{0|1}.npy)."""
        return (list(self.names) == DISK_CHANNELS
                and all(r == "delta" for r in self.update_rules))

    def describe(self):
        lines = ["ChannelSchema:"]
        for i, (n, r, lw, aw) in enumerate(zip(
                self.names, self.update_rules, self.loss_weights,
                self.alpha_weighted)):
            lines.append(f"  [{i}] {n:8s} rule={r:8s} loss_w={lw:<4g} "
                         f"alpha_weighted={aw}")
        lines.append(f"  field_dim={self.field_dim}  out_dim={self.out_dim}  "
                     f"delta_indices={self.delta_indices}  "
                     f"alpha_idx={self.alpha_idx}")
        return "\n".join(lines)

    # ---------------- construction + validation ----------------

    @classmethod
    def from_cfg(cls, cfg, verbose=True):
        """Build + validate from a Hydra/OmegaConf config.

        New configs:    cfg.data.channels list (single source of truth).
        Legacy configs: no 'channels' key (old checkpoints' .hydra/config.yaml)
                        -> historical 6-channel all-delta fallback honoring
                        weight_u_by_alpha / weight_nut_by_alpha.

        verbose=False silences info prints (used by the autoname resolver,
        which may be invoked multiple times during config resolution).
        Validation asserts are NEVER silenced.
        """
        data_cfg = cfg.data
        if "channels" not in data_cfg or data_cfg.channels is None:
            return cls._legacy_default(data_cfg, verbose=verbose)

        names, rules, weights, aw = [], [], [], []
        all_names, disabled = [], []
        for ch in data_cfg.channels:
            name = str(ch["name"])
            all_names.append(name)
            if not bool(ch.get("enabled", True)):
                # enabled: false == the row does not exist (Method C):
                # not an input, no head, no stats, no field_dim contribution.
                disabled.append(name)
                continue
            names.append(name)
            rules.append(str(ch.get("update_rule", "delta")))
            weights.append(float(ch.get("loss_weight", 1.0)))
            aw.append(bool(ch.get("alpha_weighted", False)))

        # ---- explicit domain-level validation (single-layer, no ConfigStore) --
        # Validate ALL names incl. disabled ones — a typo in a disabled row
        # must still fail loud, not silently "disable" a nonexistent channel.
        unknown = [n for n in all_names if n not in DISK_CHANNELS]
        assert not unknown, (
            f"channel names not in DISK_CHANNELS {DISK_CHANNELS}: {unknown}")
        assert len(set(all_names)) == len(all_names), \
            f"duplicate channel names: {all_names}"
        assert "alpha" not in disabled, \
            "channel 'alpha' cannot be disabled — it is the VOF core field"
        if disabled and verbose:
            print(f"[schema] disabled channels (excluded entirely): {disabled}")
        assert len(names) > 0, "data.channels has no enabled channels"
        bad = [r for r in rules if r not in UPDATE_RULES]
        assert not bad, f"invalid update_rule {bad}; allowed: {UPDATE_RULES}"
        assert "alpha" in names, \
            "channel 'alpha' (alpha.water) must be present — it is the VOF core field"
        if "flux_div" in rules:
            raise NotImplementedError(
                "update_rule 'flux_div' is reserved for the E2 flux head and "
                "not implemented yet")
        for n, r, lw in zip(names, rules, weights):
            if r != "delta" and lw != 0.0 and verbose:
                print(f"[schema] WARNING: channel '{n}' has rule '{r}' but "
                      f"loss_weight={lw}; non-delta channels receive no loss "
                      f"— weight ignored.")

        schema = cls(tuple(names), tuple(rules), tuple(weights), tuple(aw))
        assert schema.out_dim > 0, "no 'delta' channels — nothing to predict"
        return schema

    @classmethod
    def _legacy_default(cls, data_cfg, verbose=True):
        wu = bool(data_cfg.get("weight_u_by_alpha", True))
        wn = bool(data_cfg.get("weight_nut_by_alpha", False))
        if verbose:
            print(f"[schema] no data.channels in config — legacy 6-channel "
                  f"fallback (weight_u={wu}, weight_nut={wn})")
        return cls(
            names=tuple(DISK_CHANNELS),
            update_rules=("delta",) * len(DISK_CHANNELS),
            loss_weights=(1.0, 1.0, 1.0, 1.0, 0.1, 0.1),
            alpha_weighted=(False, wu, wu, wu, False, wn),
        )


# ====================================================================
# Shared rollout closure — the ONLY place window shifting happens.
# Training (train.py) and inference (vis.py / vis_u.py) both call this.
# Future flux head: flux_div dispatch will be added HERE and nowhere else.
# ====================================================================

def advance_window(window, delta_partial, schema):
    """Advance the temporal window by one predicted step.

    Args:
        window:        (B, N, W*F) — normalized field window, flattened
        delta_partial: (B, N, out_dim) — model output, DELTA channels only
        schema:        ChannelSchema

    Returns:
        pred_frame: (B, N, F) — next frame. Delta channels updated by
                    prediction; frozen channels carried unchanged (delta=0).
        new_window: (B, N, W*F) — window shifted by one, pred appended.

    Gradients flow through delta_partial (index_copy is differentiable);
    full-BPTT rollout behavior is unchanged from the original inline code.
    """
    F = schema.field_dim
    current = window[..., -F:]                                  # (B, N, F)

    if schema.out_dim == F:
        delta_full = delta_partial                              # fast path
    else:
        # Scatter partial delta into full channel layout (fork-1b).
        idx = torch.as_tensor(schema.delta_indices,
                              device=delta_partial.device)
        delta_full = delta_partial.new_zeros(
            *delta_partial.shape[:-1], F).index_copy(-1, idx, delta_partial)

    pred_frame = current + delta_full                           # (B, N, F)
    new_window = torch.cat([window[..., F:], pred_frame], dim=-1)
    return pred_frame, new_window


# ====================================================================
# Auto run naming — the experiment name is itself a derived quantity.
#
# Name encodes the DIFF from the full 6-channel all-delta baseline:
#   hpm_bl_h128            E0 (no diffs)
#   hpm_no-nut_h128        nut enabled: false   (Method C)
#   hpm_frz-nut_h128       nut update_rule: frozen (Method B)
#   hpm_flux-alpha_no-nut_h128   future E2
#   ..._aU_...             any alpha_weighted channel present
# Optional manual tag via wandb.name_suffix (appended at the end).
#
# Wired into config.yaml as  wandb.name: ${autoname:}  — hydra run dir and
# vis.sh FEATURE already follow wandb.name, so the whole chain is automatic.
# CLI override still works:  python train.py wandb.name=custom
# ====================================================================

def auto_run_name(cfg, prefix="hpm"):
    """Derive a run name from the channel schema + key hyperparameters."""
    schema = ChannelSchema.from_cfg(cfg, verbose=False)
    parts = []
    for n in DISK_CHANNELS:
        if n not in schema.names:
            parts.append(f"no-{n}")
        else:
            rule = schema.update_rules[schema.names.index(n)]
            if rule == "frozen":
                parts.append(f"frz-{n}")
            elif rule == "flux_div":
                parts.append(f"flux-{n}")
    if any(schema.alpha_weighted):
        parts.append("aU")
    if not parts:
        parts = ["bl"]

    name = "_".join([prefix, *parts, f"h{cfg.model.n_hidden}"])
    suffix = str(cfg.wandb.get("name_suffix", "") or "")
    if suffix:
        name += f"_{suffix}"
    return name


def register_autoname_resolver():
    """Register the ${autoname:} OmegaConf resolver. Must run BEFORE Hydra
    resolves hydra.run.dir — i.e. at module import time in train.py."""
    from omegaconf import OmegaConf
    if not OmegaConf.has_resolver("autoname"):
        OmegaConf.register_new_resolver(
            "autoname", lambda *, _root_: auto_run_name(_root_))
