"""
Task 3: Algorithm Comparison (Make It Better) — Project 2 (LunarLander-v3)

- Trains a PPO agent for the same number of timesteps as the DQN run (Task 2).
- Saves: models/ppo_lunarlander.zip
- Saves: results/ppo_training_curve.csv
- Saves: plots/ppo_training_curve.png
- Saves: plots/dqn_vs_ppo_comparison.png (requires dqn_training_curve.csv from Task 2)

Usage:
    python task3_train_ppo_compare.py --timesteps 200000
"""

import os
import argparse
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from utils import (RewardLoggerCallback, save_curve_csv, load_curve_csv,
                    plot_training_curve, plot_comparison_curve)

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
                         help="Total training timesteps (must match DQN run for a fair comparison)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = gym.make("LunarLander-v3")
    env = Monitor(env)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.999,
        gae_lambda=0.98,
        ent_coef=0.01,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        seed=args.seed,
        tensorboard_log=LOG_DIR,
    )

    callback = RewardLoggerCallback()
    print(f"Training PPO for {args.timesteps} timesteps...")
    model.learn(total_timesteps=args.timesteps, callback=callback,
                tb_log_name="ppo", progress_bar=False)

    model_path = os.path.join(MODELS_DIR, "ppo_lunarlander.zip")
    model.save(model_path)
    print(f"Saved model to {model_path}")

    csv_path = os.path.join(RESULTS_DIR, "ppo_training_curve.csv")
    save_curve_csv(csv_path, callback.timesteps, callback.rewards, callback.lengths)
    print(f"Saved training curve data to {csv_path}")

    plot_path = os.path.join(PLOTS_DIR, "ppo_training_curve.png")
    plot_training_curve(callback.timesteps, callback.rewards,
                         title="PPO Training Curve — LunarLander-v3",
                         out_path=plot_path)
    print(f"Saved training curve plot to {plot_path}")

    env.close()

    # --- Comparison plot against DQN, if Task 2 has already been run ---
    dqn_csv = os.path.join(RESULTS_DIR, "dqn_training_curve.csv")
    if os.path.exists(dqn_csv):
        dqn_t, dqn_r, _ = load_curve_csv(dqn_csv)
        comp_path = os.path.join(PLOTS_DIR, "dqn_vs_ppo_comparison.png")
        plot_comparison_curve(
            curve_a=(dqn_t, dqn_r),
            curve_b=(callback.timesteps, callback.rewards),
            label_a="DQN",
            label_b="PPO",
            title="DQN vs PPO — Training Reward on LunarLander-v3",
            out_path=comp_path,
        )
        print(f"Saved DQN vs PPO comparison plot to {comp_path}")
    else:
        print(f"[info] {dqn_csv} not found — run task2_train_dqn.py first "
              f"to generate the DQN-vs-PPO comparison plot.")


if __name__ == "__main__":
    main()
