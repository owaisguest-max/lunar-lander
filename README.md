# Project 2 — Deep RL Lunar Lander (RL Theory Course)

Trains and compares DQN and PPO agents on Gymnasium's `LunarLander-v3`, following
Tasks 1–4 of the project brief. Task 5 (report/discussion) is written analysis on
top of the outputs these scripts generate — see "Writing Task 5" below.

## Setup

```bash
pip install -r requirements.txt
```

This installs `gymnasium[box2d]` (needed for the Lunar Lander physics/rendering),
`stable-baselines3`, `matplotlib`, `imageio` (+ ffmpeg backend for video), and
`tensorboard` (optional, for live training curves).

## Run order

```bash
# Task 1: random baseline (fast, ~1 minute)
python task1_random_baseline.py

# Task 2: train DQN (slow — 200k timesteps takes roughly 10-30 min on CPU)
python task2_train_dqn.py --timesteps 200000

# Task 3: train PPO for the same timestep budget + auto-generates the
# DQN vs PPO comparison plot (needs Task 2's output to exist first)
python task3_train_ppo_compare.py --timesteps 200000

# Task 4: formal evaluation (100 episodes) + a few landing videos, for each model
python task4_evaluate.py --algo dqn --model models/dqn_lunarlander.zip
python task4_evaluate.py --algo ppo --model models/ppo_lunarlander.zip
```

Increase `--timesteps` (e.g. to 500,000) if either agent hasn't converged —
check `plots/dqn_training_curve.png` / `plots/ppo_training_curve.png` for a
plateau near or above the "solved" line (reward = 200).

## What each script produces

| Script | Outputs |
|---|---|
| `task1_random_baseline.py` | `results/random_baseline_results.csv`, `results/random_baseline_summary.txt`, `plots/random_baseline_rewards.png`, `videos/random_agent_episode.mp4` |
| `task2_train_dqn.py` | `models/dqn_lunarlander.zip`, `results/dqn_training_curve.csv`, `plots/dqn_training_curve.png` |
| `task3_train_ppo_compare.py` | `models/ppo_lunarlander.zip`, `results/ppo_training_curve.csv`, `plots/ppo_training_curve.png`, `plots/dqn_vs_ppo_comparison.png` |
| `task4_evaluate.py` | `results/{algo}_eval_summary.txt`, `results/{algo}_eval_episodes.csv`, `videos/{algo}_landing_*.mp4` |

## Writing Task 5 (Reporting and Interpretation)

Once you have the outputs above, use them to answer the brief's questions:

- **Compare `dqn_eval_summary.txt` vs `ppo_eval_summary.txt`**: which had the
  higher average reward / landing success rate?
- **Look at `dqn_vs_ppo_comparison.png`**: which curve converged faster / more
  smoothly? PPO is usually more stable (on-policy, clipped updates); DQN can be
  noisier but sometimes reaches a high score faster early on.
- **Watch the landing videos**: note any "quirky" strategies (e.g. hovering,
  oscillating side-thrusters, off-center but still >200-reward landings).
- **Timing**: `time_elapsed` is printed in SB3's training logs — compare
  wall-clock time and sample efficiency between the two algorithms.

## Notes

- All scripts use SB3's `Monitor` wrapper so per-episode reward/length are
  available via `infos["episode"]`, which is what `RewardLoggerCallback`
  (in `utils.py`) reads to build training curves without needing TensorBoard.
- Videos require a working `imageio-ffmpeg` install; if rendering isn't
  available in your environment, the CSV/plot outputs are unaffected.
- Every script is runnable standalone; re-running Task 2/3 overwrites the
  previous model and curve files, so rename outputs first if you want to keep
  multiple experiment runs (e.g. different `--seed` values) side by side.
