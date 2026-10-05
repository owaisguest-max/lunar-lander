"""
Task 2: Baseline RL Model Training (DQN) — Project 2 (LunarLander-v3)

- Trains a DQN agent with Stable-Baselines3.
- Logs episode reward vs. timesteps (via Monitor + RewardLoggerCallback).
- Saves: models/dqn_lunarlander.zip
- Saves: results/dqn_training_curve.csv
- Saves: plots/dqn_training_curve.png

Usage:
    python task2_train_dqn.py --timesteps 200000
"""

import os
import argparse
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

from utils import RewardLoggerCallback, save_curve_csv, plot_training_curve

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(OUT_DIR, "models")
RESULTS_DIR = os.path.join(OUT_DIR, "results")
PLOTS_DIR = os.path.join(OUT_DIR, "plots")
LOG_DIR = os.path.join(OUT_DIR, "tb_logs")

for d in (MODELS_DIR, RESULTS_DIR, PLOTS_DIR, LOG_DIR):
    os.makedirs(d, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=200_000,
                         help="Total training timesteps (paper default: 200,000)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = gym.make("LunarLander-v3")
    env = Monitor(env)  # required so `infos["episode"]` gets populated

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        buffer_size=100_000,
        learning_starts=1_000,
        batch_size=64,
        gamma=0.99,
        train_freq=4,
        target_update_interval=1_000,
        exploration_fraction=0.2,
        exploration_final_eps=0.02,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        seed=args.seed,
        tensorboard_log=LOG_DIR,
    )

    callback = RewardLoggerCallback()
    print(f"Training DQN for {args.timesteps} timesteps...")
    model.learn(total_timesteps=args.timesteps, callback=callback,
                tb_log_name="dqn", progress_bar=False)

    model_path = os.path.join(MODELS_DIR, "dqn_lunarlander.zip")
    model.save(model_path)
    print(f"Saved model to {model_path}")

    csv_path = os.path.join(RESULTS_DIR, "dqn_training_curve.csv")
    save_curve_csv(csv_path, callback.timesteps, callback.rewards, callback.lengths)
    print(f"Saved training curve data to {csv_path}")

    plot_path = os.path.join(PLOTS_DIR, "dqn_training_curve.png")
    plot_training_curve(callback.timesteps, callback.rewards,
                         title="DQN Training Curve — LunarLander-v3",
                         out_path=plot_path)
    print(f"Saved training curve plot to {plot_path}")

    env.close()


if __name__ == "__main__":
    main()
