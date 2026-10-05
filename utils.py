"""
Shared utilities for the LunarLander-v3 RL project (Project 2).

Contains:
  - RewardLoggerCallback: records per-episode reward/length during training
    so we can plot learning curves without relying on TensorBoard.
  - save_curve_csv / load_curve_csv: persist reward curves to disk.
  - plot_training_curve: single-algorithm training curve plot.
  - plot_comparison_curve: overlay DQN vs PPO training curves (Task 3).
  - moving_average: simple smoothing for noisy episode-reward curves.
"""

import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback


class RewardLoggerCallback(BaseCallback):
    """
    Logs (timestep, episode_reward, episode_length) every time an episode
    finishes. Works with SB3's built-in Monitor wrapper, which puts episode
    stats into `infos[i]["episode"]` when an episode ends.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.timesteps = []
        self.rewards = []
        self.lengths = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            ep_info = info.get("episode")
            if ep_info is not None:
                self.timesteps.append(self.num_timesteps)
                self.rewards.append(ep_info["r"])
                self.lengths.append(ep_info["l"])
        return True


def moving_average(values, window=20):
    values = np.asarray(values, dtype=np.float64)
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def save_curve_csv(path, timesteps, rewards, lengths):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestep", "episode_reward", "episode_length"])
        for t, r, l in zip(timesteps, rewards, lengths):
            writer.writerow([t, r, l])


def load_curve_csv(path):
    timesteps, rewards, lengths = [], [], []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            timesteps.append(int(row["timestep"]))
            rewards.append(float(row["episode_reward"]))
            lengths.append(int(row["episode_length"]))
    return timesteps, rewards, lengths


def plot_training_curve(timesteps, rewards, title, out_path, window=20, label="Episode Reward"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(timesteps, rewards, alpha=0.25, color="tab:blue", label=f"{label} (raw)")
    if len(rewards) >= window:
        smoothed = moving_average(rewards, window)
        smoothed_x = timesteps[window - 1:]
        ax.plot(smoothed_x, smoothed, color="tab:blue", linewidth=2,
                 label=f"{label} (smoothed, w={window})")
    ax.axhline(200, color="green", linestyle="--", linewidth=1, label="Solved threshold (200)")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Episode Reward")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_comparison_curve(curve_a, curve_b, label_a, label_b, title, out_path, window=20):
    """
    curve_a / curve_b: tuples of (timesteps, rewards)
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))

    for (timesteps, rewards), label, color in [
        (curve_a, label_a, "tab:blue"),
        (curve_b, label_b, "tab:orange"),
    ]:
        ax.plot(timesteps, rewards, alpha=0.15, color=color)
        if len(rewards) >= window:
            smoothed = moving_average(rewards, window)
            smoothed_x = timesteps[window - 1:]
            ax.plot(smoothed_x, smoothed, color=color, linewidth=2, label=label)

    ax.axhline(200, color="green", linestyle="--", linewidth=1, label="Solved threshold (200)")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Episode Reward")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
