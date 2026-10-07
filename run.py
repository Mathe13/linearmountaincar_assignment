#!/usr/bin/env python3
"""
run.py - Systematic hyperparameter evaluation for SARSA Linear Function Approximation
on MountainCar-v0.

Evaluates configurations across independent random seeds and records the number
of environment steps required to reach 10 successful episodes (reaching the goal).
Results are saved to results.csv and results.json.
"""

import os
import sys
import time
import json
import csv
from typing import Dict, Any, List
from concurrent.futures import ProcessPoolExecutor, as_completed

# Environment fallback wrapper if executed with python lacking dependencies
try:
    import gymnasium as gym
    import numpy as np
except ImportError:
    import subprocess
    import shutil
    uv_bin = shutil.which("uv") or os.path.expanduser("~/.local/bin/uv")
    if os.path.exists(uv_bin):
        cmd = [uv_bin, "run", "python"] + sys.argv
        sys.exit(subprocess.call(cmd))
    else:
        raise

from feature_extractors import RBFFeatureExtractor
from sarsa_linear_agent import LinearSarsaAgent


def run_single_trial(
    lr: float,
    n_centers: int,
    seed: int,
    max_steps: int = 50000,
    target_successes: int = 10,
    sigma: float = 0.15,
    epsilon: float = 0.1,
    epsilon_decay: float = 0.995,
    epsilon_min: float = 0.01,
    discount_factor: float = 1.0,
) -> Dict[str, Any]:
    """Run a single experiment trial with given hyperparameters and seed."""
    env = gym.make("MountainCar-v0")
    np.random.seed(seed)

    feature_extractor = RBFFeatureExtractor(env=env, n_centers=n_centers, sigma=sigma)
    agent = LinearSarsaAgent(
        env=env,
        feature_extractor=feature_extractor,
        learning_rate=lr,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
        discount_factor=discount_factor,
    )

    current_step = 0
    successes = 0
    steps_to_target = None
    episodes_count = 0

    while current_step < max_steps:
        state, _ = env.reset(seed=seed + episodes_count * 1000)
        action = agent.act(state)
        done = False
        reached_goal = False

        while not done and current_step < max_steps:
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            current_step += 1

            if terminated:
                reached_goal = True

            if done:
                agent.updateQ(state, action, reward, next_state, 0, done=True)
            else:
                next_action = agent.act(next_state)
                agent.updateQ(state, action, reward, next_state, next_action, done=False)
                state = next_state
                action = next_action

        if done:
            agent.decay_epsilon()
            episodes_count += 1
            if reached_goal:
                successes += 1
                if successes >= target_successes and steps_to_target is None:
                    steps_to_target = current_step
                    break

    env.close()

    final_steps = steps_to_target if steps_to_target is not None else max_steps
    return {
        "learning_rate": lr,
        "n_centers": n_centers,
        "seed": seed,
        "steps_to_10_successes": final_steps,
        "reached_target": (steps_to_target is not None),
        "total_successes": successes,
        "episodes": episodes_count,
    }


def main():
    print("=" * 60)
    print("SARSA MountainCar Hyperparameter Experiment")
    print("=" * 60)

    learning_rates = [0.005, 0.01, 0.02, 0.05, 0.1]
    n_centers_list = [9, 16, 25, 36, 49]
    seeds = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]  # 10 independent seeds
    max_steps = 50000
    target_successes = 10

    total_configs = len(learning_rates) * len(n_centers_list)
    total_runs = total_configs * len(seeds)

    print(f"Configurations to test: {total_configs}")
    print(f"Seeds per config:       {len(seeds)}")
    print(f"Total runs:             {total_runs}")
    print(f"Max steps per run:      {max_steps}")
    print(f"Target successes:       {target_successes}")
    print("-" * 60)

    tasks = []
    for lr in learning_rates:
        for nc in n_centers_list:
            for seed in seeds:
                tasks.append((lr, nc, seed))

    results: List[Dict[str, Any]] = []
    start_time = time.time()
    completed_runs = 0

    max_workers = min(8, os.cpu_count() or 1)
    print(f"Running experiments using {max_workers} parallel workers...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                run_single_trial,
                lr=lr,
                n_centers=nc,
                seed=seed,
                max_steps=max_steps,
                target_successes=target_successes,
            ): (lr, nc, seed)
            for (lr, nc, seed) in tasks
        }

        for future in as_completed(futures):
            res = future.result()
            results.append(res)
            completed_runs += 1
            if completed_runs % 25 == 0 or completed_runs == total_runs:
                elapsed = time.time() - start_time
                print(
                    f"Progress: [{completed_runs}/{total_runs}] "
                    f"({(completed_runs / total_runs) * 100:.1f}%) "
                    f"Elapsed: {elapsed:.1f}s"
                )

    total_time = time.time() - start_time
    print(f"\nAll experiments completed in {total_time:.2f} seconds!")

    # Sort results
    results.sort(key=lambda x: (x["learning_rate"], x["n_centers"], x["seed"]))

    # Save to CSV
    csv_file = "results.csv"
    fieldnames = [
        "learning_rate",
        "n_centers",
        "seed",
        "steps_to_10_successes",
        "reached_target",
        "total_successes",
        "episodes",
    ]
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved raw results to: {csv_file}")

    # Save to JSON
    json_file = "results.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved raw results to: {json_file}")

    # Aggregate statistics per configuration
    config_stats = {}
    for r in results:
        cfg_key = f"lr={r['learning_rate']}, nc={r['n_centers']}"
        if cfg_key not in config_stats:
            config_stats[cfg_key] = {
                "learning_rate": r["learning_rate"],
                "n_centers": r["n_centers"],
                "steps": [],
                "successes_reached": 0,
            }
        config_stats[cfg_key]["steps"].append(r["steps_to_10_successes"])
        if r["reached_target"]:
            config_stats[cfg_key]["successes_reached"] += 1

    summary_list = []
    for cfg_key, stats in config_stats.items():
        arr = np.array(stats["steps"])
        summary_list.append({
            "config": cfg_key,
            "learning_rate": stats["learning_rate"],
            "n_centers": stats["n_centers"],
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "median": float(np.median(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "success_rate": stats["successes_reached"] / len(stats["steps"]),
            "steps": stats["steps"],
        })

    # Sort by median steps (ascending)
    summary_list.sort(key=lambda x: x["median"])

    print("\nTop 5 fastest configurations (by median steps to 10 successes):")
    for i, s in enumerate(summary_list[:5], 1):
        print(
            f" {i}. {s['config']}: median={s['median']:.0f}, mean={s['mean']:.1f} ± {s['std']:.1f}, "
            f"range=[{s['min']}, {s['max']}], success_rate={s['success_rate']*100:.0f}%"
        )

    # Select 5 representative configurations covering different dynamics:
    # 1. Best overall (lowest median, high capacity & high lr)
    # 2. Strong performer with medium capacity (e.g. n_centers=25, lr=0.05)
    # 3. Moderate performer with conservative learning rate (e.g. n_centers=49, lr=0.01)
    # 4. Low capacity with high learning rate (n_centers=9, lr=0.05) showing coarse representation
    # 5. Slowest / baseline configuration (e.g. n_centers=9, lr=0.01)
    interesting_keys = [
        "lr=0.05, nc=49",
        "lr=0.05, nc=25",
        "lr=0.01, nc=49",
        "lr=0.05, nc=9",
        "lr=0.01, nc=9",
    ]

    selected_5 = [s for s in summary_list if s["config"] in interesting_keys]
    # If any key wasn't in summary, pick the fastest 3 and slowest 2
    if len(selected_5) < 5:
        selected_5 = summary_list[:3] + summary_list[-2:]

    with open("top_5_configs.json", "w", encoding="utf-8") as f:
        json.dump(selected_5, f, indent=2)
    print("Saved 5 representative configurations to: top_5_configs.json")


if __name__ == "__main__":
    main()

