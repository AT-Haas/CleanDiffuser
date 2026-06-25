"""Maze2D goal-reaching dataset for the few-step planner (D4RL ``maze2d-*-v1``).

Adds ``D4RLMaze2DDataset`` to CleanDiffuser: it chunks the offline transition stream
into horizon-length windows (not crossing episode boundaries) and exposes the
per-episode goal so the planner can be conditioned by goal-inpainting rather than
return-CFG. See the class docstring for the item layout and a usage example.
"""

from typing import Dict

import numpy as np
import torch

from cleandiffuser.dataset.base_dataset import BaseDataset
from cleandiffuser.utils import GaussianNormalizer


class D4RLMaze2DDataset(BaseDataset):
    """**D4RL-Maze2D Sequential Dataset** (goal-reaching navigation).

    Wrapper for the ``maze2d-*-v1`` datasets. Observations are 4-D
    ``[pos_x, pos_y, vel_x, vel_y]`` and actions are 2-D. Chunks the contiguous
    stream into windows of length ``horizon`` that do not cross episode
    boundaries (defined by ``timeouts``/``terminals``); observations are
    Gaussian-normalised. The per-episode goal (``infos/goal``, a 2-D target
    position) is exposed so the planner can be conditioned by **goal-inpainting**
    (pin the start at ``[0]`` and the goal position at ``[-1, :2]`` via the
    diffusion ``fix_mask``/``prior``) rather than return-CFG.

    Each item:
      - ``obs["state"]``: (horizon, 4)  normalised observations
      - ``act``:          (horizon, 2)
      - ``rew``:          (horizon, 1)  sparse goal reward
      - ``goal``:         (2,)          raw (unnormalised) target position

    Args:
        dataset: ``env.get_dataset()`` for a maze2d env.
        horizon: window length. Default 1.
        normalize_goal_with_state: if True, ``goal`` is returned normalised with
            the position dims of the state normaliser (handy for inpainting).

    Examples:
        >>> import gym, d4rl
        >>> ds = D4RLMaze2DDataset(gym.make("maze2d-large-v1").get_dataset(), horizon=128)
        >>> b = ds[0]; b["obs"]["state"].shape, b["goal"].shape
        (torch.Size([128, 4]), torch.Size([2]))
    """

    def __init__(
        self,
        dataset: Dict[str, np.ndarray],
        horizon: int = 1,
        normalize_goal_with_state: bool = False,
    ):
        super().__init__()
        observations = dataset["observations"].astype(np.float32)
        actions = dataset["actions"].astype(np.float32)
        rewards = dataset["rewards"].astype(np.float32)
        terminals = np.asarray(dataset["terminals"]).astype(bool)
        timeouts = np.asarray(dataset.get("timeouts", np.zeros_like(terminals))).astype(bool)
        N = observations.shape[0]
        goals = dataset.get("infos/goal", None)
        if goals is None:
            goals = np.zeros((N, 2), dtype=np.float32)
        goals = np.asarray(goals).astype(np.float32)

        self.normalizers = {"state": GaussianNormalizer(observations)}
        normed_obs = self.normalizers["state"].normalize(observations)

        self.horizon = horizon
        self.obs_dim, self.act_dim = observations.shape[-1], actions.shape[-1]

        self.seq_obs = torch.tensor(normed_obs, dtype=torch.float32)
        self.seq_act = torch.tensor(actions, dtype=torch.float32)
        self.seq_rew = torch.tensor(rewards, dtype=torch.float32)[:, None]
        if normalize_goal_with_state:
            mean = self.normalizers["state"].mean[:2]
            std = self.normalizers["state"].std[:2]
            goals = (goals - mean) / std
        self.goal = torch.tensor(goals, dtype=torch.float32)

        # Build window start indices that stay within a single episode.
        done = np.logical_or(terminals, timeouts)
        self.indices = []
        ep_start = 0
        for i in range(N):
            if done[i] or i == N - 1:
                ep_end = i + 1  # exclusive
                last_start = ep_end - horizon
                if last_start >= ep_start:
                    self.indices.extend(range(ep_start, last_start + 1))
                ep_start = ep_end
        self.indices = np.asarray(self.indices, dtype=np.int64)

    def get_normalizer(self):
        """Return the fitted observation normalizer (a ``GaussianNormalizer`` over the
        state dims), so the planner and inverse-dynamics model share the same scaling."""
        return self.normalizers["state"]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        s = int(self.indices[idx])
        e = s + self.horizon
        return {
            "obs": {"state": self.seq_obs[s:e]},
            "act": self.seq_act[s:e],
            "rew": self.seq_rew[s:e],
            "goal": self.goal[s],
        }
