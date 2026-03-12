from __future__ import annotations

from typing import Generator, Optional

import numpy as np
import torch as th
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.type_aliases import RolloutBufferSamples
from stable_baselines3.common.vec_env import VecNormalize


class MaskedRolloutBuffer(RolloutBuffer):
    """RolloutBuffer that masks invalid samples marked via NaN rewards.

    This matches the GPUDDrive training pattern where dead-agent timesteps are
    kept in the env stream but excluded from PPO loss computation.
    """

    valid_samples_mask: np.ndarray

    def compute_returns_and_advantage(self, last_values: th.Tensor, dones: np.ndarray) -> None:
        # Convert to numpy and treat NaN dones as terminal.
        last_values_np = last_values.clone().cpu().numpy().flatten()
        dones_np = np.nan_to_num(np.asarray(dones, dtype=np.float32), nan=1.0)

        last_gae_lam = 0.0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones_np
                next_values = last_values_np
            else:
                episode_starts = np.nan_to_num(self.episode_starts[step + 1], nan=1.0)
                next_non_terminal = 1.0 - episode_starts
                next_values = self.values[step + 1]

            rewards = np.nan_to_num(self.rewards[step], nan=0.0)
            values = np.nan_to_num(self.values[step], nan=0.0)
            next_values = np.nan_to_num(next_values, nan=0.0)

            delta = rewards + self.gamma * next_values * next_non_terminal - values
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam

        self.returns = self.advantages + self.values

    def get(self, batch_size: Optional[int] = None) -> Generator[RolloutBufferSamples, None, None]:
        assert self.full, ""

        if not self.generator_ready:
            tensor_names = [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "rewards",
            ]

            flat_rewards = self.swap_and_flatten(self.rewards)
            # SB3 swap_and_flatten may keep a trailing singleton dim.
            # Use a 1D mask over sample rows.
            self.valid_samples_mask = ~np.isnan(np.asarray(flat_rewards).reshape(-1))

            for tensor in tensor_names:
                flat_tensor = np.asarray(self.swap_and_flatten(self.__dict__[tensor]))
                if flat_tensor.ndim == 1:
                    masked_tensor = flat_tensor[self.valid_samples_mask]
                elif flat_tensor.ndim == 2:
                    masked_tensor = flat_tensor[self.valid_samples_mask, :]
                    if tensor not in {"observations", "actions"}:
                        masked_tensor = masked_tensor.reshape(-1)
                else:
                    raise RuntimeError(
                        f"Unexpected tensor rank for {tensor}: shape={flat_tensor.shape}"
                    )
                self.__dict__[tensor] = masked_tensor

                if np.isnan(self.__dict__[tensor]).any():
                    raise RuntimeError(f"{tensor} tensor contains NaN values after masking")

            self.generator_ready = True

        total_num_samples = int(self.valid_samples_mask.sum())
        if total_num_samples <= 0:
            raise RuntimeError(
                "MaskedRolloutBuffer has no valid samples. "
                "Check env masking and done/reset configuration."
            )

        indices = np.random.permutation(total_num_samples)

        if batch_size is None:
            batch_size = total_num_samples

        start_idx = 0
        while start_idx < total_num_samples:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> RolloutBufferSamples:
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds].astype(np.float32, copy=False),
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
        )
        return RolloutBufferSamples(*tuple(map(self.to_torch, data)))
