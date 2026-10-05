"""
Task 1: Environment Setup and Random Baseline Exploration (Project 2 - LunarLander)

- Instantiates LunarLander-v3
- Runs a random agent for N episodes (default 100)
- Records reward per episode, crash vs. successful-landing counts
- Saves: results/random_baseline_results.csv, results/random_baseline_summary.txt
- Saves: plots/random_baseline_rewards.png
- Saves: videos/random_agent_episode.mp4  (one episode rendered)

LunarLander scoring convention (per the Gymnasium docs):
  - Episode reward >= 200 is generally considered a successful landing.
  - A large negative terminal reward (-100) is applied on crash.
We use "reward >= 200" as the success criterion and "reward <= -100"
region as a proxy flag for a crash, and cross-check with termination info.
"""

import os
import csv
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

N_EPISODES = 100
SEED = 42
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(OUT_DIR, "results")
PLOTS_DIR = os.path.join(OUT_DIR, "plots")
VIDEOS_DIR = os.path.join(OUT_DIR, "videos")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(VIDEOS_DIR, exist_ok=True)


def run_random_agent(n_episodes=N_EPISODES, seed=SEED):
    env = gym.make("LunarLander-v3")
    rng = np.random.default_rng(seed)

    rewards, lengths, crashed_flags, landed_flags = [], [], [], []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=int(rng.integers(0, 1_000_000)))
        total_reward = 0.0
        steps = 0
        terminated = truncated = False
        crashed = False

        while not (terminated or truncated):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            # A crash ends the episode with a large one-step penalty (-100)
            if terminated and reward <= -90:
                crashed = True

        landed = total_reward >= 200
        rewards.append(total_reward)
        lengths.append(steps)
        crashed_flags.append(crashed)
        landed_flags.append(landed)

    env.close()
    return rewards, lengths, crashed_flags, landed_flags


def record_one_episode_video(path, seed=SEED):
    """Render a single random-agent episode to an mp4 for the report."""
    try:
        env = gym.make("LunarLander-v3", render_mode="rgb_array")
        frames = []
        obs, info = env.reset(seed=seed)
        terminated = truncated = False
        while not (terminated or truncated):
            frames.append(env.render())
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
        env.close()

        import imageio
        imageio.mimsave(path, frames, fps=30)
        print(f"Saved random-agent video to {path}")
    except Exception as e:
        print(f"[warn] Could not render/save video ({e}). "
              f"This needs a working Box2D + ffmpeg install; skipping video.")


def main():
    print(f"Environment: LunarLander-v3")
    dummy_env = gym.make("LunarLander-v3")
    print(f"Observation space: {dummy_env.observation_space}")
    print(f"Action space: {dummy_env.action_space}")
    print("Reward range: approx [-100 (crash), +100 (per-leg contact bonuses), "
          "200+ = successful landing]")
    dummy_env.close()

    print(f"\nRunning random agent for {N_EPISODES} episodes...")
    rewards, lengths, crashed_flags, landed_flags = run_random_agent()

    # --- Save per-episode results ---
    csv_path = os.path.join(RESULTS_DIR, "random_baseline_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward", "length", "crashed", "landed_success"])
        for i, (r, l, c, s) in enumerate(zip(rewards, lengths, crashed_flags, landed_flags)):
            writer.writerow([i, r, l, c, s])
    print(f"Saved per-episode results to {csv_path}")

    # --- Summary metrics ---
    avg_reward = float(np.mean(rewards))
    std_reward = float(np.std(rewards))
    crash_rate = 100.0 * sum(crashed_flags) / len(crashed_flags)
    win_rate = 100.0 * sum(landed_flags) / len(landed_flags)

    summary_lines = [
        "=== Random Agent Baseline Summary (LunarLander-v3) ===",
        f"Episodes run:            {N_EPISODES}",
        f"Average reward:          {avg_reward:.2f}",
        f"Std dev of reward:       {std_reward:.2f}",
        f"Min / Max reward:        {min(rewards):.2f} / {max(rewards):.2f}",
        f"Average episode length:  {np.mean(lengths):.1f} steps",
        f"Crash rate:              {crash_rate:.1f}%",
        f"Successful landing rate: {win_rate:.1f}%",
    ]
    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)

    summary_path = os.path.join(RESULTS_DIR, "random_baseline_summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary_text + "\n")
    print(f"\nSaved summary to {summary_path}")

    # --- Plot reward distribution over episodes ---
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(rewards, marker="o", markersize=3, linewidth=1, alpha=0.7, label="Episode reward")
    ax.axhline(avg_reward, color="red", linestyle="--", label=f"Mean = {avg_reward:.1f}")
    ax.axhline(200, color="green", linestyle=":", label="Solved threshold (200)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("Random Agent — Reward per Episode (LunarLander-v3)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, "random_baseline_rewards.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved reward plot to {plot_path}")

    # --- One rendered episode (for "1 visual frame or short gif/video") ---
    video_path = os.path.join(VIDEOS_DIR, "random_agent_episode.mp4")
    record_one_episode_video(video_path)


if __name__ == "__main__":
    main()
