from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import torch
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from torch import nn


def _make_mlp(input_dim: int, layers: List[int], act_fn: nn.Module, dropout: float) -> nn.Sequential:
    seq = []
    last_dim = int(input_dim)
    for layer_dim in layers:
        seq.append(nn.Linear(last_dim, int(layer_dim)))
        if dropout > 0:
            seq.append(nn.Dropout(float(dropout)))
        seq.append(act_fn)
        last_dim = int(layer_dim)
    return nn.Sequential(*seq)


class LateFusionNet(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        *,
        ego_dim: int = 11,
        point_dim: int = 3,
        point_k: Optional[int] = None,
        ego_layers: Optional[List[int]] = None,
        point_layers: Optional[List[int]] = None,
        shared_layers: Optional[List[int]] = None,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
        act: str = "relu",
        dropout: float = 0.0,
        pool: str = "max",
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.ego_dim = int(ego_dim)
        self.point_dim = int(point_dim)

        remaining = max(0, self.obs_dim - self.ego_dim)
        inferred_k = remaining // max(1, self.point_dim)
        if point_k is None:
            self.point_k = int(inferred_k)
        else:
            self.point_k = min(int(point_k), int(inferred_k))

        ego_layers = ego_layers or [64, 64]
        point_layers = point_layers or [64, 64]
        shared_layers = shared_layers or [64]

        if act == "tanh":
            act_fn = nn.Tanh()
        else:
            act_fn = nn.ReLU()

        self.pool = pool
        self.latent_dim_pi = int(last_layer_dim_pi)
        self.latent_dim_vf = int(last_layer_dim_vf)

        self.ego_out_dim = int(ego_layers[-1]) if ego_layers else self.ego_dim
        self.point_out_dim = int(point_layers[-1]) if point_layers else self.point_dim
        self.ego_net_actor = _make_mlp(self.ego_dim, ego_layers, act_fn, dropout)
        self.ego_net_critic = _make_mlp(self.ego_dim, ego_layers, act_fn, dropout)
        self.point_net_actor = _make_mlp(self.point_dim, point_layers, act_fn, dropout)
        self.point_net_critic = _make_mlp(self.point_dim, point_layers, act_fn, dropout)

        shared_input_dim = self.ego_out_dim + self.point_out_dim
        self.actor_out = _make_mlp(shared_input_dim, shared_layers, act_fn, dropout)
        self.critic_out = _make_mlp(shared_input_dim, shared_layers, act_fn, dropout)

        self.actor_last = nn.Linear(shared_layers[-1] if shared_layers else shared_input_dim, self.latent_dim_pi)
        self.critic_last = nn.Linear(shared_layers[-1] if shared_layers else shared_input_dim, self.latent_dim_vf)

    def _split_obs(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ego = features[:, : self.ego_dim]
        if self.point_k <= 0:
            points = features.new_zeros((features.shape[0], 0, self.point_dim))
        else:
            start = self.ego_dim
            end = self.ego_dim + self.point_k * self.point_dim
            points_flat = features[:, start:end]
            points = points_flat.reshape(-1, self.point_k, self.point_dim)
        return ego, points

    def _pool_points(self, point_feats: torch.Tensor) -> torch.Tensor:
        if point_feats.numel() == 0:
            return point_feats.new_zeros((point_feats.shape[0], point_feats.shape[-1]))
        if self.pool == "mean":
            return point_feats.mean(dim=1)
        return point_feats.max(dim=1).values

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        ego, points = self._split_obs(features)
        ego_emb = self.ego_net_actor(ego)
        if points.numel() == 0:
            points_emb = ego_emb.new_zeros((ego_emb.shape[0], self.point_out_dim))
        else:
            p = points.reshape(-1, self.point_dim)
            p_emb = self.point_net_actor(p).reshape(points.shape[0], points.shape[1], -1)
            points_emb = self._pool_points(p_emb)
        x = torch.cat([ego_emb, points_emb], dim=1)
        x = self.actor_out(x)
        return self.actor_last(x)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        ego, points = self._split_obs(features)
        ego_emb = self.ego_net_critic(ego)
        if points.numel() == 0:
            points_emb = ego_emb.new_zeros((ego_emb.shape[0], self.point_out_dim))
        else:
            p = points.reshape(-1, self.point_dim)
            p_emb = self.point_net_critic(p).reshape(points.shape[0], points.shape[1], -1)
            points_emb = self._pool_points(p_emb)
        x = torch.cat([ego_emb, points_emb], dim=1)
        x = self.critic_out(x)
        return self.critic_last(x)


class LateFusionPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        ego_dim: int = 11,
        point_dim: int = 3,
        point_k: Optional[int] = None,
        ego_layers: Optional[List[int]] = None,
        point_layers: Optional[List[int]] = None,
        shared_layers: Optional[List[int]] = None,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
        act: str = "relu",
        dropout: float = 0.0,
        pool: str = "max",
        *args,
        **kwargs,
    ):
        self._lf_kwargs = dict(
            ego_dim=ego_dim,
            point_dim=point_dim,
            point_k=point_k,
            ego_layers=ego_layers,
            point_layers=point_layers,
            shared_layers=shared_layers,
            last_layer_dim_pi=last_layer_dim_pi,
            last_layer_dim_vf=last_layer_dim_vf,
            act=act,
            dropout=dropout,
            pool=pool,
        )
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        obs_dim = int(self.observation_space.shape[0])
        self.mlp_extractor = LateFusionNet(obs_dim, **self._lf_kwargs)
