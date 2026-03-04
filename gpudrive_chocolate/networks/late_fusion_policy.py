from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import torch
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from torch import nn


def _make_mlp(
    input_dim: int,
    layers: List[int],
    act_fn: nn.Module,
    dropout: float,
) -> nn.Sequential:
    seq: List[nn.Module] = []
    last_dim = int(input_dim)
    for layer_dim in layers:
        seq.append(nn.Linear(last_dim, int(layer_dim)))
        seq.append(nn.LayerNorm(int(layer_dim)))
        if dropout > 0:
            seq.append(nn.Dropout(float(dropout)))
        seq.append(act_fn)
        last_dim = int(layer_dim)
    return nn.Sequential(*seq)


def _branch_out_dim(input_dim: int, layers: List[int]) -> int:
    if input_dim <= 0:
        return 0
    return int(layers[-1]) if layers else int(input_dim)


class LegacyLateFusionNet(nn.Module):
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

        act_fn = nn.Tanh() if act == "tanh" else nn.ReLU()
        self.pool = str(pool)
        self.latent_dim_pi = int(last_layer_dim_pi)
        self.latent_dim_vf = int(last_layer_dim_vf)

        self.ego_out_dim = _branch_out_dim(self.ego_dim, ego_layers)
        self.point_out_dim = _branch_out_dim(self.point_dim, point_layers)
        self.ego_net_actor = _make_mlp(self.ego_dim, ego_layers, act_fn, dropout)
        self.ego_net_critic = _make_mlp(self.ego_dim, ego_layers, act_fn, dropout)
        self.point_net_actor = _make_mlp(self.point_dim, point_layers, act_fn, dropout)
        self.point_net_critic = _make_mlp(self.point_dim, point_layers, act_fn, dropout)

        shared_input_dim = self.ego_out_dim + self.point_out_dim
        self.actor_out = _make_mlp(shared_input_dim, shared_layers, act_fn, dropout)
        self.critic_out = _make_mlp(shared_input_dim, shared_layers, act_fn, dropout)

        actor_in = shared_layers[-1] if shared_layers else shared_input_dim
        critic_in = shared_layers[-1] if shared_layers else shared_input_dim
        self.actor_last = nn.Linear(int(actor_in), self.latent_dim_pi)
        self.critic_last = nn.Linear(int(critic_in), self.latent_dim_vf)

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


class StructuredLateFusionNet(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        *,
        ego_dim: int = 11,
        road_point_dim: int = 5,
        road_point_k: int = 0,
        vehicle_dim: int = 6,
        vehicle_k: int = 0,
        ego_layers: Optional[List[int]] = None,
        road_layers: Optional[List[int]] = None,
        vehicle_layers: Optional[List[int]] = None,
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
        self.road_point_dim = max(0, int(road_point_dim))
        self.road_point_k = max(0, int(road_point_k))
        self.vehicle_dim = max(0, int(vehicle_dim))
        self.vehicle_k = max(0, int(vehicle_k))

        ego_layers = ego_layers or [64, 64]
        road_layers = road_layers or [64, 64]
        vehicle_layers = vehicle_layers or [64, 64]
        shared_layers = shared_layers or [128, 64]

        act_fn = nn.Tanh() if act == "tanh" else nn.ReLU()
        self.pool = str(pool)
        self.latent_dim_pi = int(last_layer_dim_pi)
        self.latent_dim_vf = int(last_layer_dim_vf)

        self.ego_out_dim = _branch_out_dim(self.ego_dim, ego_layers)
        self.road_out_dim = _branch_out_dim(self.road_point_dim, road_layers)
        self.vehicle_out_dim = _branch_out_dim(self.vehicle_dim, vehicle_layers)

        self.ego_net_actor = _make_mlp(self.ego_dim, ego_layers, act_fn, dropout)
        self.ego_net_critic = _make_mlp(self.ego_dim, ego_layers, act_fn, dropout)

        self.road_net_actor = (
            _make_mlp(self.road_point_dim, road_layers, act_fn, dropout)
            if self.road_point_dim > 0
            else None
        )
        self.road_net_critic = (
            _make_mlp(self.road_point_dim, road_layers, act_fn, dropout)
            if self.road_point_dim > 0
            else None
        )

        self.vehicle_net_actor = (
            _make_mlp(self.vehicle_dim, vehicle_layers, act_fn, dropout)
            if self.vehicle_dim > 0
            else None
        )
        self.vehicle_net_critic = (
            _make_mlp(self.vehicle_dim, vehicle_layers, act_fn, dropout)
            if self.vehicle_dim > 0
            else None
        )

        shared_input_dim = self.ego_out_dim + self.road_out_dim + self.vehicle_out_dim
        self.actor_out = _make_mlp(shared_input_dim, shared_layers, act_fn, dropout)
        self.critic_out = _make_mlp(shared_input_dim, shared_layers, act_fn, dropout)

        actor_in = shared_layers[-1] if shared_layers else shared_input_dim
        critic_in = shared_layers[-1] if shared_layers else shared_input_dim
        self.actor_last = nn.Linear(int(actor_in), self.latent_dim_pi)
        self.critic_last = nn.Linear(int(critic_in), self.latent_dim_vf)

    @staticmethod
    def _slice_with_padding(features: torch.Tensor, start: int, length: int) -> torch.Tensor:
        if length <= 0:
            return features.new_zeros((features.shape[0], 0))
        end = min(features.shape[1], start + length)
        chunk = features[:, start:end]
        if chunk.shape[1] == length:
            return chunk
        pad = features.new_zeros((features.shape[0], length - chunk.shape[1]))
        return torch.cat([chunk, pad], dim=1)

    def _split_obs(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ego = self._slice_with_padding(features, 0, self.ego_dim)
        offset = self.ego_dim

        road_len = self.road_point_k * self.road_point_dim
        road_flat = self._slice_with_padding(features, offset, road_len)
        offset += road_len

        vehicle_len = self.vehicle_k * self.vehicle_dim
        vehicle_flat = self._slice_with_padding(features, offset, vehicle_len)

        if self.road_point_k > 0 and self.road_point_dim > 0:
            road_points = road_flat.reshape(-1, self.road_point_k, self.road_point_dim)
        else:
            road_points = features.new_zeros((features.shape[0], 0, max(1, self.road_point_dim)))

        if self.vehicle_k > 0 and self.vehicle_dim > 0:
            vehicles = vehicle_flat.reshape(-1, self.vehicle_k, self.vehicle_dim)
        else:
            vehicles = features.new_zeros((features.shape[0], 0, max(1, self.vehicle_dim)))

        return ego, road_points, vehicles

    def _pool_entities(self, entity_inputs: torch.Tensor, entity_emb: torch.Tensor) -> torch.Tensor:
        if entity_inputs.numel() == 0 or entity_emb.numel() == 0:
            return entity_emb.new_zeros((entity_emb.shape[0], entity_emb.shape[-1]))
        valid = torch.any(torch.abs(entity_inputs) > 1e-6, dim=-1)
        if self.pool == "mean":
            weights = valid.to(entity_emb.dtype).unsqueeze(-1)
            denom = torch.clamp(weights.sum(dim=1), min=1.0)
            return (entity_emb * weights).sum(dim=1) / denom

        masked = entity_emb.masked_fill(~valid.unsqueeze(-1), float("-inf"))
        pooled = masked.max(dim=1).values
        empty = ~valid.any(dim=1)
        if empty.any():
            pooled[empty] = 0.0
        return pooled

    def _encode_branch(
        self,
        inputs: torch.Tensor,
        net: Optional[nn.Module],
        out_dim: int,
    ) -> torch.Tensor:
        if out_dim <= 0 or net is None or inputs.numel() == 0 or inputs.shape[1] == 0:
            return inputs.new_zeros((inputs.shape[0], out_dim))
        flat = inputs.reshape(-1, inputs.shape[-1])
        emb = net(flat).reshape(inputs.shape[0], inputs.shape[1], -1)
        return self._pool_entities(inputs, emb)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        ego, road_points, vehicles = self._split_obs(features)
        ego_emb = self.ego_net_actor(ego)
        road_emb = self._encode_branch(road_points, self.road_net_actor, self.road_out_dim)
        vehicle_emb = self._encode_branch(vehicles, self.vehicle_net_actor, self.vehicle_out_dim)
        fused = torch.cat([ego_emb, road_emb, vehicle_emb], dim=1)
        return self.actor_last(self.actor_out(fused))

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        ego, road_points, vehicles = self._split_obs(features)
        ego_emb = self.ego_net_critic(ego)
        road_emb = self._encode_branch(road_points, self.road_net_critic, self.road_out_dim)
        vehicle_emb = self._encode_branch(vehicles, self.vehicle_net_critic, self.vehicle_out_dim)
        fused = torch.cat([ego_emb, road_emb, vehicle_emb], dim=1)
        return self.critic_last(self.critic_out(fused))


class LateFusionNet(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        *,
        ego_dim: int = 11,
        point_dim: int = 3,
        point_k: Optional[int] = None,
        road_point_dim: Optional[int] = None,
        road_point_k: Optional[int] = None,
        vehicle_dim: Optional[int] = None,
        vehicle_k: Optional[int] = None,
        ego_layers: Optional[List[int]] = None,
        point_layers: Optional[List[int]] = None,
        road_layers: Optional[List[int]] = None,
        vehicle_layers: Optional[List[int]] = None,
        shared_layers: Optional[List[int]] = None,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
        act: str = "relu",
        dropout: float = 0.0,
        pool: str = "max",
    ):
        super().__init__()
        use_structured = road_point_k is not None or vehicle_k is not None
        if use_structured:
            self.impl = StructuredLateFusionNet(
                obs_dim=obs_dim,
                ego_dim=ego_dim,
                road_point_dim=int(road_point_dim or 0),
                road_point_k=int(road_point_k or 0),
                vehicle_dim=int(vehicle_dim or 0),
                vehicle_k=int(vehicle_k or 0),
                ego_layers=ego_layers,
                road_layers=road_layers or point_layers,
                vehicle_layers=vehicle_layers or point_layers,
                shared_layers=shared_layers,
                last_layer_dim_pi=last_layer_dim_pi,
                last_layer_dim_vf=last_layer_dim_vf,
                act=act,
                dropout=dropout,
                pool=pool,
            )
        else:
            self.impl = LegacyLateFusionNet(
                obs_dim=obs_dim,
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
        self.latent_dim_pi = self.impl.latent_dim_pi
        self.latent_dim_vf = self.impl.latent_dim_vf

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.impl(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.impl.forward_actor(features)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.impl.forward_critic(features)


class LateFusionPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        ego_dim: int = 11,
        point_dim: int = 3,
        point_k: Optional[int] = None,
        road_point_dim: Optional[int] = None,
        road_point_k: Optional[int] = None,
        vehicle_dim: Optional[int] = None,
        vehicle_k: Optional[int] = None,
        ego_layers: Optional[List[int]] = None,
        point_layers: Optional[List[int]] = None,
        road_layers: Optional[List[int]] = None,
        vehicle_layers: Optional[List[int]] = None,
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
            road_point_dim=road_point_dim,
            road_point_k=road_point_k,
            vehicle_dim=vehicle_dim,
            vehicle_k=vehicle_k,
            ego_layers=ego_layers,
            point_layers=point_layers,
            road_layers=road_layers,
            vehicle_layers=vehicle_layers,
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
