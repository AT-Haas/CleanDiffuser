"""Shortcut Flow planner on D4RL MuJoCo locomotion.

Pipeline:
    Planner:   ContinuousShortcutFlow on obs-only horizon-H trajectories
    Condition: MLPCondition — CFG on discounted cumulative return
    InvDyn:    MlpInvDynamic — (obs_t, obs_{t+1}) → act_t

Run modes (selected by ``mode`` in the Hydra config):
    training   — train both planner and invdyn from scratch, save checkpoints
    inference  — load checkpoints, evaluate on a vectorised gym env

Reference: Frans et al., "One Step Diffusion via Shortcut Models", arXiv:2410.12557.
"""

import os
os.environ.setdefault("D4RL_SUPPRESS_IMPORT_ERROR", "1")

from pathlib import Path

import d4rl  # noqa: F401
import gym
import hydra
import numpy as np
import pytorch_lightning as L
import torch
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset

from cleandiffuser.dataset.d4rl_mujoco_dataset import D4RLMuJoCoDataset
from cleandiffuser.diffusion import ContinuousShortcutFlow
from cleandiffuser.invdynamic import MlpInvDynamic
from cleandiffuser.nn_condition import MLPCondition
from cleandiffuser.nn_diffusion import DiT1dShortcut
from cleandiffuser.utils import DD_RETURN_SCALE, set_seed


# ----------------------------- Dataset wrappers ---------------------------- #

class PlannerDataset(Dataset):
    """Yields ``{"x0": (H, obs_dim), "condition_cfg": (1,)}`` for the planner."""

    def __init__(self, ds: D4RLMuJoCoDataset, scale: float):
        self.ds = ds
        self.scale = scale

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        path_idx, start, end = self.ds.indices[idx]
        obs = self.ds.seq_obs[path_idx, start:end]
        ret = float(self.ds.seq_val[path_idx, start]) / self.scale
        return {
            "x0":            torch.tensor(obs, dtype=torch.float32),
            "condition_cfg": torch.tensor([ret], dtype=torch.float32),
        }


class FlatInvDynDataset(Dataset):
    """Yields ``{"obs", "act", "next_obs"}`` flattened across trajectories."""

    def __init__(self, ds: D4RLMuJoCoDataset):
        seq_obs = ds.seq_obs.astype(np.float32)
        seq_act = ds.seq_act.astype(np.float32)
        self.obs_arr  = seq_obs[:, :-1].reshape(-1, seq_obs.shape[-1])
        self.act_arr  = seq_act[:, :-1].reshape(-1, seq_act.shape[-1])
        self.nobs_arr = seq_obs[:, 1:].reshape(-1, seq_obs.shape[-1])

    def __len__(self):
        return len(self.obs_arr)

    def __getitem__(self, idx):
        return {
            "obs":      torch.tensor(self.obs_arr[idx]),
            "act":      torch.tensor(self.act_arr[idx]),
            "next_obs": torch.tensor(self.nobs_arr[idx]),
        }


# ----------------------------- Model factories ----------------------------- #

def build_planner(cfg, obs_dim: int) -> ContinuousShortcutFlow:
    nn_planner = DiT1dShortcut(
        x_dim=obs_dim,
        x_seq_len=cfg.horizon,
        emb_dim=cfg.emb_dim,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        depth=cfg.depth,
        timestep_emb_type="untrainable_fourier",
        timestep_emb_params={"scale": 0.02},
    )
    nn_cond = MLPCondition(
        in_dim=1, out_dim=cfg.emb_dim, hidden_dims=cfg.emb_dim, dropout=cfg.cfg_dropout,
    )

    fix_mask = torch.zeros((cfg.horizon, obs_dim))
    fix_mask[0] = 1.0
    loss_weight = torch.ones((cfg.horizon, obs_dim))
    loss_weight[1] = cfg.next_obs_loss_weight

    return ContinuousShortcutFlow(
        nn_diffusion=nn_planner,
        nn_condition=nn_cond,
        fix_mask=fix_mask,
        loss_weight=loss_weight,
        ema_rate=cfg.ema_rate,
        K_max=cfg.K_max,
        fm_consistency_ratio=cfg.fm_consistency_ratio,
        optimizer_params={"lr": cfg.lr},  # wrapped to {"diffusion": {"optimizer": "adam", "lr": ...}}
    )


def build_invdyn(cfg, obs_dim: int, act_dim: int) -> MlpInvDynamic:
    return MlpInvDynamic(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_dim=cfg.invdyn_hidden_dim,
    )


# ----------------------------- Evaluation ---------------------------------- #

def evaluate(cfg, flow: ContinuousShortcutFlow, invdyn: MlpInvDynamic,
             dataset: D4RLMuJoCoDataset, env_name: str, device: torch.device):
    """Run a vectorised rollout and report mean normalised D4RL score."""
    flow.eval()
    invdyn.eval()

    env_eval = gym.vector.make(env_name, cfg.num_envs)
    env_norm = gym.make(env_name)  # only for env.get_normalized_score
    normalizer = dataset.get_normalizer()

    obs_dim = dataset.obs_dim
    target_return = torch.full((cfg.num_envs, 1), cfg.task.target_return, device=device)

    all_episode_returns = []
    for ep in range(cfg.num_episodes):
        obs, ep_reward, cum_done, t = env_eval.reset(), 0.0, np.zeros(cfg.num_envs, dtype=bool), 0
        while not np.all(cum_done) and t < 1000 + 1:
            obs_n = torch.tensor(normalizer.normalize(obs), dtype=torch.float32, device=device)
            prior = torch.zeros((cfg.num_envs, cfg.horizon, obs_dim), device=device)
            prior[:, 0] = obs_n
            with torch.no_grad():
                obs_traj, _ = flow.sample(
                    prior=prior,
                    solver="euler_shortcut",
                    sample_steps=cfg.sampling_steps,
                    condition_cfg=target_return,
                    w_cfg=cfg.w_cfg,
                    use_ema=cfg.use_ema,
                )
                act = invdyn.predict(obs_traj[:, 0], obs_traj[:, 1]).cpu().numpy()
            obs, rew, done, info = env_eval.step(act)
            t += 1
            cum_done = np.logical_or(cum_done, done)
            ep_reward = ep_reward + rew * (~cum_done)
        norm = np.array([env_norm.get_normalized_score(r) for r in ep_reward])
        all_episode_returns.append(norm)
        print(f"[eval episode {ep}] normalised score "
              f"mean={norm.mean()*100:.2f}  std={norm.std()*100:.2f}")

    env_eval.close()
    env_norm.close()
    all_returns = np.stack(all_episode_returns) * 100
    print(f"\n[eval summary]  mean={all_returns.mean():.2f}  "
          f"std={all_returns.std():.2f}  over {cfg.num_episodes} episodes × "
          f"{cfg.num_envs} envs, sampling_steps={cfg.sampling_steps}")
    return float(all_returns.mean()), float(all_returns.std())


# ----------------------------- Main entry ---------------------------------- #

@hydra.main(config_path="../../configs/shortcut/mujoco", config_name="mujoco", version_base=None)
def pipeline(cfg):
    set_seed(cfg.seed)
    print(OmegaConf.to_yaml(cfg))

    save_dir = Path(cfg.save_dir) / cfg.pipeline_name / cfg.task.env_name / f"seed_{cfg.seed}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- Dataset ----------------
    env = gym.make(cfg.task.env_name)
    raw = env.get_dataset()
    dataset = D4RLMuJoCoDataset(raw, horizon=cfg.horizon, discount=cfg.discount)
    obs_dim, act_dim = dataset.obs_dim, dataset.act_dim
    scale = DD_RETURN_SCALE.get(cfg.task.env_name, cfg.task.get("return_scale", 1.0))

    # ---------------- Models ----------------
    flow = build_planner(cfg, obs_dim)
    invdyn = build_invdyn(cfg, obs_dim, act_dim)

    device = torch.device("cuda" if torch.cuda.is_available() and cfg.devices > 0 else "cpu")
    flow.to(device)
    invdyn.to(device)

    planner_ckpt = save_dir / "planner.ckpt"
    invdyn_ckpt = save_dir / "invdyn.ckpt"

    # ---------------- Training ----------------
    if cfg.mode == "training":
        planner_loader = DataLoader(
            PlannerDataset(dataset, scale),
            batch_size=cfg.batch_size, shuffle=True,
            num_workers=cfg.num_workers, pin_memory=True, persistent_workers=cfg.num_workers > 0,
        )
        invdyn_loader = DataLoader(
            FlatInvDynDataset(dataset),
            batch_size=cfg.invdyn_batch_size, shuffle=True,
            num_workers=cfg.num_workers, pin_memory=True, persistent_workers=cfg.num_workers > 0,
        )

        # Loggers
        loggers = []
        if cfg.use_wandb:
            loggers.append(WandbLogger(
                project=cfg.wandb_project,
                name=f"{cfg.task.env_name}-seed{cfg.seed}",
                save_dir=str(save_dir),
            ))

        # ---- Train planner ----
        print(f"\n=== Training planner for {cfg.diffusion_gradient_steps:,} steps "
              f"on {cfg.task.env_name} ===")
        planner_cb = ModelCheckpoint(
            dirpath=str(save_dir), filename="planner_step{step}",
            every_n_train_steps=cfg.save_interval, save_top_k=-1,
        )
        trainer_p = L.Trainer(
            accelerator="gpu" if device.type == "cuda" else "cpu",
            devices=cfg.devices if device.type == "cuda" else 1,
            max_steps=cfg.diffusion_gradient_steps,
            log_every_n_steps=cfg.log_interval,
            enable_progress_bar=cfg.enable_progress_bar,
            callbacks=[planner_cb],
            logger=loggers or False,
            precision=cfg.precision,
        )
        trainer_p.fit(flow, planner_loader)
        trainer_p.save_checkpoint(str(planner_ckpt))
        print(f"Planner saved → {planner_ckpt}")

        # ---- Train invdyn ----
        print(f"\n=== Training invdyn for {cfg.invdyn_gradient_steps:,} steps ===")
        invdyn_cb = ModelCheckpoint(
            dirpath=str(save_dir), filename="invdyn_step{step}",
            every_n_train_steps=cfg.save_interval, save_top_k=-1,
        )
        trainer_i = L.Trainer(
            accelerator="gpu" if device.type == "cuda" else "cpu",
            devices=cfg.devices if device.type == "cuda" else 1,
            max_steps=cfg.invdyn_gradient_steps,
            log_every_n_steps=cfg.log_interval,
            enable_progress_bar=cfg.enable_progress_bar,
            callbacks=[invdyn_cb],
            logger=loggers or False,
            precision=cfg.precision,
        )
        trainer_i.fit(invdyn, invdyn_loader)
        trainer_i.save_checkpoint(str(invdyn_ckpt))
        print(f"InvDyn saved → {invdyn_ckpt}")

        # ---- Quick post-training eval ----
        if cfg.eval_after_training:
            evaluate(cfg, flow, invdyn, dataset, cfg.task.env_name, device)

    # ---------------- Inference ----------------
    elif cfg.mode == "inference":
        planner_path = Path(cfg.load_planner_ckpt) if cfg.load_planner_ckpt else planner_ckpt
        invdyn_path = Path(cfg.load_invdyn_ckpt) if cfg.load_invdyn_ckpt else invdyn_ckpt
        assert planner_path.exists(), f"Planner checkpoint not found: {planner_path}"
        assert invdyn_path.exists(),  f"InvDyn checkpoint not found:  {invdyn_path}"

        flow_state = torch.load(planner_path, map_location=device)
        flow.load_state_dict(flow_state["state_dict"])
        invdyn_state = torch.load(invdyn_path, map_location=device)
        invdyn.load_state_dict(invdyn_state["state_dict"])

        evaluate(cfg, flow, invdyn, dataset, cfg.task.env_name, device)

    else:
        raise ValueError(f"Unknown mode: {cfg.mode}. Use 'training' or 'inference'.")


if __name__ == "__main__":
    pipeline()
