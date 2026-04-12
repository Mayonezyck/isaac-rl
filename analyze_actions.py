#!/usr/bin/env python3
"""Analyze action distribution from a trained SceneFactory policy checkpoint.

Reconstructs the LateFusion actor offline (no Isaac Sim needed),
feeds it observations drawn from the running normalizer stats stored in the checkpoint,
and plots throttle / steering / brake histograms.
"""
import sys, json
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── config ──────────────────────────────────────────────────────────────
CKPT = Path("logs/rsl_rl/scene_factory_goal_reaching_roads/"
            "2026-04-06_12-46-16_scene_factory_256scene_curated_0326_train_fastgoal_v2_anticrawl/"
            "model_1050.pt")
N_SAMPLES  = 50_000          # synthetic observations to draw
SEED       = 42
OUT_DIR    = Path("writeup/action_analysis")
# architecture constants (must match training config)
EGO_DIM        = 11
ROAD_POINT_DIM = 5
ROAD_POINT_K   = 350
VEHICLE_DIM    = 7
VEHICLE_K      = 24
EGO_LAYERS     = [64, 64]
ROAD_LAYERS    = [96, 96]
VEHICLE_LAYERS = [96, 96]
SHARED_LAYERS  = [128, 64]
LAST_PI = 64
LAST_VF = 64
NUM_ACTIONS = 3
ACTIVATION  = "relu"
DROPOUT     = 0.0
POOL        = "max"

# ── helpers ─────────────────────────────────────────────────────────────
def _act(name):
    return {"relu": nn.ReLU, "tanh": nn.Tanh, "elu": nn.ELU, "silu": nn.SiLU}[name]()

def _mlp(in_d, layers, act, drop):
    mods = []
    d = in_d
    for w in layers:
        mods += [nn.Linear(d, w), nn.LayerNorm(w)]
        if drop > 0: mods.append(nn.Dropout(drop))
        mods.append(_act(act) if isinstance(act, str) else act)
        d = w
    return nn.Sequential(*mods)

class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.ego_dim = EGO_DIM
        self.road_point_dim = ROAD_POINT_DIM
        self.road_point_k = ROAD_POINT_K
        self.vehicle_dim = VEHICLE_DIM
        self.vehicle_k = VEHICLE_K
        self.latent_dim_pi = LAST_PI
        self.latent_dim_vf = LAST_VF
        self.pool = POOL

        self.ego_out = EGO_LAYERS[-1]
        self.road_out = ROAD_LAYERS[-1]
        self.vehicle_out = VEHICLE_LAYERS[-1]

        self.ego_net_actor = _mlp(EGO_DIM, EGO_LAYERS, ACTIVATION, DROPOUT)
        self.ego_net_critic = _mlp(EGO_DIM, EGO_LAYERS, ACTIVATION, DROPOUT)
        self.road_net_actor = _mlp(ROAD_POINT_DIM, ROAD_LAYERS, ACTIVATION, DROPOUT)
        self.road_net_critic = _mlp(ROAD_POINT_DIM, ROAD_LAYERS, ACTIVATION, DROPOUT)
        self.vehicle_net_actor = _mlp(VEHICLE_DIM, VEHICLE_LAYERS, ACTIVATION, DROPOUT)
        self.vehicle_net_critic = _mlp(VEHICLE_DIM, VEHICLE_LAYERS, ACTIVATION, DROPOUT)

        shared_in = self.ego_out + self.road_out + self.vehicle_out
        self.actor_shared = _mlp(shared_in, SHARED_LAYERS, ACTIVATION, DROPOUT)
        self.critic_shared = _mlp(shared_in, SHARED_LAYERS, ACTIVATION, DROPOUT)
        self.actor_last = nn.Linear(SHARED_LAYERS[-1], LAST_PI)
        self.critic_last = nn.Linear(SHARED_LAYERS[-1], LAST_VF)

    def _pool_entities(self, raw, emb):
        if raw.numel() == 0: return emb.new_zeros(emb.shape[0], emb.shape[-1])
        valid = (raw.abs() > 1e-6).any(-1)
        if self.pool == "max":
            m = emb.masked_fill(~valid.unsqueeze(-1), float("-inf"))
            p = m.max(1).values
            empty = ~valid.any(1)
            if empty.any(): p[empty] = 0.0
            return p
        w = valid.float().unsqueeze(-1)
        return (emb * w).sum(1) / w.sum(1).clamp(min=1)

    def _encode(self, raw, net, out):
        B = raw.shape[0]
        if raw.numel() == 0: return raw.new_zeros(B, out)
        flat = raw.reshape(-1, raw.shape[-1])
        e = net(flat).reshape(B, raw.shape[1], -1)
        return self._pool_entities(raw, e)

    def forward_actor(self, obs):
        ego = obs[:, :EGO_DIM]
        off = EGO_DIM
        road = obs[:, off:off+ROAD_POINT_K*ROAD_POINT_DIM].reshape(-1, ROAD_POINT_K, ROAD_POINT_DIM)
        off += ROAD_POINT_K*ROAD_POINT_DIM
        veh  = obs[:, off:off+VEHICLE_K*VEHICLE_DIM].reshape(-1, VEHICLE_K, VEHICLE_DIM)
        ego_e = self.ego_net_actor(ego)
        road_e = self._encode(road, self.road_net_actor, self.road_out)
        veh_e  = self._encode(veh,  self.vehicle_net_actor, self.vehicle_out)
        fused = torch.cat([ego_e, road_e, veh_e], 1)
        return self.actor_last(self.actor_shared(fused))

class ActorHead(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(LAST_PI, NUM_ACTIONS)
    def forward(self, obs):
        return self.head(self.backbone.forward_actor(obs))

# ── main ────────────────────────────────────────────────────────────────
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = "cpu"

    print(f"Loading checkpoint: {CKPT}")
    ckpt = torch.load(str(CKPT), map_location=device, weights_only=False)
    sd = ckpt["model_state_dict"]

    # build model
    backbone = Backbone()
    actor = ActorHead(backbone)
    action_std = nn.Parameter(torch.ones(NUM_ACTIONS))

    # load weights (actor.* keys)
    actor_sd = {}
    for k, v in sd.items():
        if k.startswith("actor."):
            actor_sd[k[len("actor."):]] = v
    missing, unexpected = actor.load_state_dict(actor_sd, strict=False)
    if missing:
        print(f"WARNING missing keys: {missing}")
    action_std.data = sd["std"].clone()

    # load normalizer stats
    obs_mean = sd["actor_obs_normalizer._mean"].squeeze().to(device)   # [obs_dim]
    obs_var  = sd["actor_obs_normalizer._var"].squeeze().to(device)
    obs_std  = sd["actor_obs_normalizer._std"].squeeze().to(device)
    obs_dim  = obs_mean.shape[0]
    print(f"Obs dim: {obs_dim}")
    print(f"Action std (learned): {action_std.data.tolist()}")
    print(f"Actor head bias: {actor.head.bias.data.tolist()}")

    actor.eval()

    # Sample observations from learned normalizer distribution (Gaussian)
    # The normalizer normalises: obs_norm = (obs - mean) / std
    # So real obs ~ N(mean, var). We sample from that, then normalise like the env does.
    raw_obs = obs_mean.unsqueeze(0) + obs_std.unsqueeze(0) * torch.randn(N_SAMPLES, obs_dim)
    normed_obs = (raw_obs - obs_mean.unsqueeze(0)) / obs_std.clamp(min=1e-8).unsqueeze(0)

    with torch.no_grad():
        raw_actions = actor(normed_obs)  # [N, 3]

    # Apply same clamping as the env
    raw_actions_np = raw_actions.numpy()
    throttle_raw = raw_actions_np[:, 0]
    steer_raw    = raw_actions_np[:, 1]
    brake_raw    = raw_actions_np[:, 2]

    # Semantic actions (post-clamp)
    throttle = np.clip(throttle_raw, 0.0, 1.0)
    steer    = np.clip(steer_raw, -1.0, 1.0)
    brake    = np.clip(brake_raw, 0.0, 1.0)

    # Also show what happens with exploration noise
    throttle_noisy_raw = throttle_raw + np.random.randn(N_SAMPLES) * action_std.data[0].item()
    steer_noisy_raw    = steer_raw    + np.random.randn(N_SAMPLES) * action_std.data[1].item()
    brake_noisy_raw    = brake_raw    + np.random.randn(N_SAMPLES) * action_std.data[2].item()
    throttle_noisy = np.clip(throttle_noisy_raw, 0.0, 1.0)
    steer_noisy    = np.clip(steer_noisy_raw, -1.0, 1.0)
    brake_noisy    = np.clip(brake_noisy_raw, 0.0, 1.0)

    # Net drive = throttle - brake
    net_drive = throttle - brake
    net_drive_noisy = throttle_noisy - brake_noisy

    # ── statistics ──
    print("\n" + "="*60)
    print("ACTION STATISTICS (deterministic / mean policy)")
    print("="*60)
    for name, arr in [("throttle_raw", throttle_raw), ("steer_raw", steer_raw), ("brake_raw", brake_raw),
                      ("throttle", throttle), ("steer", steer), ("brake", brake), ("net_drive", net_drive)]:
        print(f"  {name:18s}: mean={arr.mean():.4f}  std={arr.std():.4f}  "
              f"min={arr.min():.4f}  p10={np.percentile(arr,10):.4f}  "
              f"p50={np.percentile(arr,50):.4f}  p90={np.percentile(arr,90):.4f}  max={arr.max():.4f}")

    print(f"\n  Fraction throttle < 0.1: {(throttle < 0.1).mean()*100:.1f}%")
    print(f"  Fraction throttle > 0.5: {(throttle > 0.5).mean()*100:.1f}%")
    print(f"  Fraction brake    > 0.1: {(brake > 0.1).mean()*100:.1f}%")
    print(f"  Fraction brake    > 0.5: {(brake > 0.5).mean()*100:.1f}%")
    print(f"  Fraction net_drive < 0 : {(net_drive < 0).mean()*100:.1f}%")
    print(f"  Fraction |steer|  > 0.3: {(np.abs(steer) > 0.3).mean()*100:.1f}%")
    thr_brake_conflict = throttle * brake
    print(f"  Throttle*Brake conflict mean: {thr_brake_conflict.mean():.4f}")

    # ── plot ──
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Action Distribution — anticrawl model_1050 (deterministic policy, synthetic obs)", fontsize=14)

    # Row 1: raw actions (before clamp)
    for ax, data, label, color in zip(
        axes[0],
        [throttle_raw, steer_raw, brake_raw],
        ["Throttle (raw)", "Steering (raw)", "Brake (raw)"],
        ["#e74c3c", "#3498db", "#f39c12"]
    ):
        ax.hist(data, bins=100, color=color, alpha=0.7, edgecolor="black", linewidth=0.3)
        ax.axvline(data.mean(), color="black", linestyle="--", linewidth=1.5, label=f"mean={data.mean():.3f}")
        ax.set_title(label, fontsize=12)
        ax.set_xlabel("Raw action value")
        ax.set_ylabel("Count")
        ax.legend(fontsize=10)

    # Row 2: semantic actions (after clamp)
    for ax, data, label, color in zip(
        axes[1],
        [throttle, steer, brake],
        ["Throttle [0,1]", "Steering [-1,1]", "Brake [0,1]"],
        ["#e74c3c", "#3498db", "#f39c12"]
    ):
        ax.hist(data, bins=100, color=color, alpha=0.7, edgecolor="black", linewidth=0.3)
        ax.axvline(data.mean(), color="black", linestyle="--", linewidth=1.5, label=f"mean={data.mean():.3f}")
        ax.set_title(label, fontsize=12)
        ax.set_xlabel("Semantic action value")
        ax.set_ylabel("Count")
        ax.legend(fontsize=10)

    plt.tight_layout()
    p = OUT_DIR / "action_distribution_model1050.png"
    fig.savefig(str(p), dpi=150)
    print(f"\nSaved: {p}")
    plt.close(fig)

    # Additional plot: net drive and throttle-brake scatter
    fig2, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    fig2.suptitle("Throttle vs Brake Analysis — anticrawl model_1050", fontsize=14)

    ax1.hist(net_drive, bins=100, color="#2ecc71", alpha=0.7, edgecolor="black", linewidth=0.3)
    ax1.axvline(net_drive.mean(), color="black", linestyle="--", linewidth=1.5, label=f"mean={net_drive.mean():.3f}")
    ax1.set_title("Net Drive (throttle − brake)")
    ax1.set_xlabel("Net drive")
    ax1.legend()

    # Scatter: throttle vs brake
    idx = np.random.choice(N_SAMPLES, min(5000, N_SAMPLES), replace=False)
    ax2.scatter(throttle[idx], brake[idx], alpha=0.1, s=5, c="#8e44ad")
    ax2.set_xlabel("Throttle")
    ax2.set_ylabel("Brake")
    ax2.set_title("Throttle vs Brake (semantic)")
    ax2.set_xlim(-0.05, 1.05)
    ax2.set_ylim(-0.05, 1.05)
    ax2.plot([0,1],[0,1], "k--", alpha=0.3, label="throttle=brake")
    ax2.legend()

    # Speed proxy: how many are in each action regime
    regimes = {
        "Full throttle\n(thr>0.8, brk<0.1)": ((throttle > 0.8) & (brake < 0.1)).mean()*100,
        "Moderate throttle\n(0.3<thr<0.8)": ((throttle > 0.3) & (throttle <= 0.8) & (brake < 0.1)).mean()*100,
        "Light throttle\n(thr<0.3, brk<0.1)": ((throttle <= 0.3) & (brake < 0.1)).mean()*100,
        "Braking\n(brk>0.1)": (brake > 0.1).mean()*100,
        "Conflict\n(thr>0.3 & brk>0.1)": ((throttle > 0.3) & (brake > 0.1)).mean()*100,
    }
    bars = ax3.barh(list(regimes.keys()), list(regimes.values()), color=["#27ae60","#f1c40f","#e67e22","#e74c3c","#8e44ad"])
    ax3.set_xlabel("Fraction of agents (%)")
    ax3.set_title("Action Regime Breakdown")
    for bar, val in zip(bars, regimes.values()):
        ax3.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, f"{val:.1f}%", va="center", fontsize=10)
    ax3.set_xlim(0, max(regimes.values()) * 1.3)

    plt.tight_layout()
    p2 = OUT_DIR / "action_regime_model1050.png"
    fig2.savefig(str(p2), dpi=150)
    print(f"Saved: {p2}")
    plt.close(fig2)

    print("\nDone.")

if __name__ == "__main__":
    main()
