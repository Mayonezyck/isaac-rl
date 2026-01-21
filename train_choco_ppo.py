# train_choco_ppo.py
from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.chocolate_env import ChocolateEnv


# -------------------------
# PPO policy/value network
# -------------------------
class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int = 7, act_dim: int = 2, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.mu = nn.Linear(hidden, act_dim)
        self.v  = nn.Linear(hidden, 1)

        # log std as parameters (global per action dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs: torch.Tensor):
        h = self.net(obs)
        mu = self.mu(h)
        v = self.v(h).squeeze(-1)
        std = torch.exp(self.log_std).clamp(1e-4, 10.0)
        return mu, std, v


def tanh_squash_action(u: torch.Tensor):
    # squashed action in (-1,1)
    return torch.tanh(u)


def action_postprocess(a_tanh: torch.Tensor):
    """
    a_tanh shape (..., 2):
      a_long in [-1,1]  (positive=throttle, negative=brake)
      steer  in [-1,1]
    Return env action (..., 3): [thr, steer, brake], with thr*brake never both >0.
    """
    a_long = a_tanh[..., 0]
    steer  = a_tanh[..., 1]

    thr   = torch.clamp(a_long, min=0.0, max=1.0)
    brake = torch.clamp(-a_long, min=0.0, max=1.0)
    return torch.stack([thr, steer, brake], dim=-1)



def logprob_gaussian(u: torch.Tensor, mu: torch.Tensor, std: torch.Tensor):
    # log N(u|mu,std) per-dim summed
    var = std ** 2
    return (-0.5 * (((u - mu) ** 2) / var + 2.0 * torch.log(std) + math.log(2.0 * math.pi))).sum(dim=-1)


# -------------------------
# PPO trainer
# -------------------------
@torch.no_grad()
def rollout(env: ChocolateEnv, model: ActorCritic, T: int, device: str):
    print('enter rollout')
    obs_np, mask_np, keys = env.reset()
    N = len(keys)
    obs_dim = obs_np.shape[1]
    obs = torch.tensor(obs_np, dtype=torch.float32, device=device)           # (N,obs)
    mask = torch.tensor(mask_np, dtype=torch.bool, device=device)            # (N,)
    act_dim = model.log_std.numel()  # should be 2 after you change the model

    # buffers [T,N,...]
    B_obs   = torch.zeros((T, N, obs_dim), dtype=torch.float32, device=device)
    B_u     = torch.zeros((T, N, act_dim), dtype=torch.float32, device=device)     # pre-tanh gaussian sample
    B_a     = torch.zeros((T, N, 3), dtype=torch.float32, device=device)     # postprocessed env action
    B_logp  = torch.zeros((T, N), dtype=torch.float32, device=device)
    B_val   = torch.zeros((T, N), dtype=torch.float32, device=device)
    B_rew   = torch.zeros((T, N), dtype=torch.float32, device=device)
    B_done  = torch.zeros((T, N), dtype=torch.bool, device=device)
    B_mask  = torch.zeros((T, N), dtype=torch.bool, device=device)
    print('in rollout')
    for t in range(T):
        #print(f'Stepped {t} times')
        mu, std, v = model(obs)
        u = mu + std * torch.randn_like(mu)
        a_tanh = tanh_squash_action(u)
        a_env = action_postprocess(a_tanh)

        # store
        B_obs[t]  = obs
        B_u[t]    = u
        B_a[t]    = a_env
        B_val[t]  = v
        B_mask[t] = mask

        # compute logp on u (pre-tanh) — not exact tanh correction, but works as a practical baseline
        B_logp[t] = logprob_gaussian(u, mu, std)

        # step env (only act on active rows)
        U_np = np.zeros((N, act_dim), np.float32)  # (N,2)
        if mask.any():
            # send 2D action (a_long, steer) to env; env will convert to (thr, steer, brake)
            U_np[mask.cpu().numpy()] = a_tanh[mask].detach().cpu().numpy()

        obs_np, rew_np, done_np, info = env.step(U_np)

        print('obs', obs_np)
        print('rew', rew_np)
        print('done', done_np)
        print('info', info)
        # optional: immediately reset done agents so batch stays “alive”
        if done_np.any():
            env.reset_done(done_np)

            # rebuild obs after reset so next step sees fresh starts
            obs_np, mask_np, keys = env._build_obs()  # uses obs_builder + ctrl
        else:
            mask_np = info.mask

        # record transition results
        B_rew[t]  = torch.tensor(rew_np, dtype=torch.float32, device=device)
        B_done[t] = torch.tensor(done_np, dtype=torch.bool, device=device)

        obs = torch.tensor(obs_np, dtype=torch.float32, device=device)
        mask = torch.tensor(mask_np, dtype=torch.bool, device=device)

    # force-reset all agents at rollout boundary
    env.reset_done(np.ones((N,), dtype=bool))
    obs_np, mask_np, keys = env._build_obs()
    obs = torch.tensor(obs_np, dtype=torch.float32, device=device)
    mask = torch.tensor(mask_np, dtype=torch.bool, device=device)

    # bootstrap value
    mu, std, v_last = model(obs)  # v(s_T)
    return B_obs, B_u, B_a, B_logp, B_val, B_rew, B_done, B_mask, v_last


def compute_gae(B_rew, B_done, B_val, v_last, gamma=0.99, lam=0.95):
    """
    GAE computed per-agent across time (the N dimension).
    B_* shapes: [T,N]
    """
    T, N = B_rew.shape
    adv = torch.zeros((T, N), dtype=torch.float32, device=B_rew.device)
    last_gae = torch.zeros((N,), dtype=torch.float32, device=B_rew.device)

    for t in reversed(range(T)):
        nonterminal = (~B_done[t]).float()
        v_next = v_last if t == T - 1 else B_val[t + 1]
        delta = B_rew[t] + gamma * v_next * nonterminal - B_val[t]
        last_gae = delta + gamma * lam * nonterminal * last_gae
        adv[t] = last_gae

    ret = adv + B_val
    return adv, ret


def ppo_update(model, opt, B_obs, B_u, B_logp_old, B_adv, B_ret, B_mask,
               clip=0.2, vf_coef=0.5, ent_coef=0.0, epochs=5, minibatch=8192):
    device = B_obs.device
    T, N, obs_dim = B_obs.shape

    # flatten (T,N) -> (T*N)
    obs = B_obs.reshape(T * N, obs_dim)
    act_dim = B_u.shape[-1]
    u = B_u.reshape(T * N, act_dim)

    logp_old = B_logp_old.reshape(T * N)
    adv = B_adv.reshape(T * N)
    ret = B_ret.reshape(T * N)
    mask = B_mask.reshape(T * N)

    # Only train on active rows from rollout
    idx = torch.where(mask)[0]
    if idx.numel() == 0:
        return

    obs = obs[idx]
    u = u[idx]
    logp_old = logp_old[idx]
    adv = adv[idx]
    ret = ret[idx]

    # normalize adv
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    M = obs.shape[0]
    perm = torch.randperm(M, device=device)

    for _ in range(epochs):
        for start in range(0, M, minibatch):
            mb = perm[start:start + minibatch]
            obs_mb = obs[mb]
            u_mb = u[mb]
            logp_old_mb = logp_old[mb]
            adv_mb = adv[mb]
            ret_mb = ret[mb]

            mu, std, v = model(obs_mb)
            logp = logprob_gaussian(u_mb, mu, std)
            ratio = torch.exp(logp - logp_old_mb)

            # clipped policy objective
            surr1 = ratio * adv_mb
            surr2 = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * adv_mb
            pi_loss = -torch.min(surr1, surr2).mean()

            # value loss
            vf_loss = 0.5 * (ret_mb - v).pow(2).mean()

            # entropy (gaussian)
            ent = (0.5 + 0.5 * math.log(2.0 * math.pi) + torch.log(std)).sum(dim=-1).mean()
            loss = pi_loss + vf_coef * vf_loss - ent_coef * ent

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()


def train_shared_policy(env: ChocolateEnv,
                        total_updates=2000,
                        rollout_T=64,
                        lr=3e-4,
                        gamma=0.99,
                        lam=0.95,
                        #device="cuda" if torch.cuda.is_available() else "cpu"):
                        device="cpu"):
    model = ActorCritic(obs_dim=7, act_dim=2, hidden=256).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)

    for upd in range(1, total_updates + 1):
        print('here right?')
        B_obs, B_u, B_a, B_logp, B_val, B_rew, B_done, B_mask, v_last = rollout(env, model, rollout_T, device)
        adv, ret = compute_gae(B_rew, B_done, B_val, v_last, gamma=gamma, lam=lam)

        ppo_update(model, opt, B_obs, B_u, B_logp, adv, ret, B_mask,
                   clip=0.2, vf_coef=0.5, ent_coef=0.0, epochs=5, minibatch=8192)

        # lightweight logging (distance meters)
        with torch.no_grad():
            active = B_mask[-1]
            avg_rew = float(B_rew.mean().cpu().item())
            done_rate = float(B_done.float().mean().cpu().item())
        if (upd % 10) == 0:
            print(f"[upd {upd:04d}] avg_rew={avg_rew:.4f} done_rate={done_rate:.3f}")

    return model
