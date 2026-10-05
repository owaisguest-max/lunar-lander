"""
Task 4: Formal Evaluation and Visual Testing — Project 2 (LunarLander-v3)

- Loads a trained model (DQN or PPO).
- Runs 100 independent evaluation episodes with the deterministic (greedy) policy.
- Reports: average reward, std dev of reward, landing success rate (%).
- Renders 2-3 successful-landing episodes to video for the report.

Usage:
    python task4_evaluate.py --algo ppo --model models/ppo_lunarlander.zip
    python task4_evaluate.py --algo dqn --model models/dqn_lunarlander.zip
"""

import os
import csv
import argparse
import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN, PPO

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(OUT_DIR, "results")
VIDEOS_DIR = os.path.join(OUT_DIR, "videos")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(VIDEOS_DIR, exist_ok=True)

ALGO_MAP = {"dqn": DQN, "ppo": PPO}


def evaluate(model, n_episodes=100, seed=1000):
    env = gym.make("LunarLander-v3")
    rewards, lengths, landed_flags = [], [], []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        terminated = truncated = False
        total_reward = 0.0
        steps = 0
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
        rewards.append(total_reward)
        lengths.append(steps)
        landed_flags.append(total_reward >= 200)

    env.close()
    return rewards, lengths, landed_flags


def record_episodes(model, algo_name, n_videos=3, max_attempts=20, seed=2000):
    """Try to capture n_videos successful landings (fall back to best available)."""
    import imageio
    env = gym.make("LunarLander-v3", render_mode="rgb_array")

    saved = 0
    attempt = 0
    fallback_best = None  # (reward, frames) in case we never land n_videos times

    while saved < n_videos and attempt < max_attempts:
        obs, info = env.reset(seed=seed + attempt)
        frames = []
        terminated = truncated = False
        total_reward = 0.0
        while not (terminated or truncated):
            frames.append(env.render())
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
        attempt += 1

        if fallback_best is None or total_reward > fallback_best[0]:
            fallback_best = (total_reward, frames)

        if total_reward >= 200:
            path = os.path.join(VIDEOS_DIR, f"{algo_name}_landing_{saved + 1}.mp4")
            imageio.mimsave(path, frames, fps=30)
            print(f"Saved successful-landing video ({total_reward:.1f} reward) to {path}")
            saved += 1

    if saved == 0 and fallback_best is not None:
        path = os.path.join(VIDEOS_DIR, f"{algo_name}_best_attempt.mp4")
        imageio.mimsave(path, fallback_best[1], fps=30)
        print(f"[info] No episode reached the 200+ landing threshold in {max_attempts} "
              f"attempts; saved best attempt ({fallback_best[0]:.1f} reward) to {path}")

    env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["dqn", "ppo"], required=True)
    parser.add_argument("--model", required=True, help="Path to the saved .zip model")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--videos", type=int, default=3)
    args = parser.parse_args()

    ModelClass = ALGO_MAP[args.algo]
    model = ModelClass.load(args.model)

    print(f"Evaluating {args.algo.upper()} model over {args.episodes} episodes "
          f"(deterministic policy)...")
    rewards, lengths, landed_flags = evaluate(model, n_episodes=args.episodes)

    avg_reward = float(np.mean(rewards))
    std_reward = float(np.std(rewards))
    success_rate = 100.0 * sum(landed_flags) / len(landed_flags)

    summary_lines = [
        f"=== Formal Evaluation Summary ({args.algo.upper()}, LunarLander-v3) ===",
        f"Episodes evaluated:      {args.episodes}",
        f"Average reward:          {avg_reward:.2f}",
        f"Std dev of reward:       {std_reward:.2f}",
        f"Min / Max reward:        {min(rewards):.2f} / {max(rewards):.2f}",
        f"Average episode length:  {np.mean(lengths):.1f} steps",
        f"Landing success rate:    {success_rate:.1f}%",
    ]
    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)

    summary_path = os.path.join(RESULTS_DIR, f"{args.algo}_eval_summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary_text + "\n")
    print(f"\nSaved summary to {summary_path}")

    csv_path = os.path.join(RESULTS_DIR, f"{args.algo}_eval_episodes.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward", "length", "landed_success"])
        for i, (r, l, s) in enumerate(zip(rewards, lengths, landed_flags)):
            writer.writerow([i, r, l, s])
    print(f"Saved per-episode eval results to {csv_path}")

    print(f"\nRecording up to {args.videos} successful-landing videos...")
    record_episodes(model, args.algo, n_videos=args.videos)


if __name__ == "__main__":
    main()
