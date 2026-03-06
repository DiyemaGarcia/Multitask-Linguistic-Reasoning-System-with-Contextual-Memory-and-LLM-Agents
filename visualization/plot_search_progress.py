# visualization/plot_search_progress.py

"""
Reproduce Figure 6 from Zoph & Le (2017):
"Improvement of Neural Architecture Search over random search over time."

Plots perplexity improvement of NAS controller vs random search
for top-1, top-5, top-15 unique models, evaluated every 400 models.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import os


def compute_top_k_average(rewards_list, k):
    """
    Compute average of top-k rewards seen so far at each checkpoint.

    Args:
        rewards_list: list of rewards in order received
        k: number of top models to average

    Returns:
        top_k_avgs: list of top-k averages at each checkpoint
    """
    top_k_avgs = []
    seen_rewards = []

    for reward in rewards_list:
        seen_rewards.append(reward)
        sorted_rewards = sorted(seen_rewards, reverse=True)
        top_k = sorted_rewards[:min(k, len(sorted_rewards))]
        top_k_avgs.append(np.mean(top_k))

    return top_k_avgs


def compute_improvement_over_random(nas_rewards, random_rewards, k_values,
                                     checkpoint_every=400):
    """
    Compute NAS improvement over random search for top-k models,
    sampled every checkpoint_every architectures.

    Args:
        nas_rewards: list of NAS controller rewards (ordered)
        random_rewards: list of random search rewards (ordered)
        k_values: list of k values to plot (e.g., [1, 5, 15])
        checkpoint_every: evaluation frequency

    Returns:
        iterations: x-axis values
        improvements: dict {k: list of improvements}
    """
    # Subsample at checkpoints
    n = min(len(nas_rewards), len(random_rewards))
    checkpoints = list(range(checkpoint_every, n + 1, checkpoint_every))

    improvements = {k: [] for k in k_values}

    for cp in checkpoints:
        nas_slice = nas_rewards[:cp]
        rnd_slice = random_rewards[:cp]

        for k in k_values:
            nas_top_k = np.mean(sorted(nas_slice, reverse=True)[:k])
            rnd_top_k = np.mean(sorted(rnd_slice, reverse=True)[:k])
            # Paper plots perplexity improvement (lower is better for PPL)
            # improvement = random_ppl - nas_ppl (positive => NAS is better)
            # For rewards (negative perplexity), improvement = nas_reward - random_reward
            improvement = nas_top_k - rnd_top_k
            improvements[k].append(improvement)

    return checkpoints, improvements


def plot_search_progress(nas_rewards, random_rewards, output_path,
                          k_values=(1, 5, 15), checkpoint_every=400,
                          task="ptb"):
    """
    Reproduce Figure 6: NAS improvement over random search.

    Args:
        nas_rewards: list of rewards from NAS controller
        random_rewards: list of rewards from random search
        output_path: path to save the figure
        k_values: top-k values to plot
        checkpoint_every: x-axis frequency
        task: "ptb" or "cifar10"
    """
    iterations, improvements = compute_improvement_over_random(
        nas_rewards, random_rewards, list(k_values), checkpoint_every
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    labels = {1: "Top_1_unique_models", 5: "Top_5_unique_models",
              15: "Top_15_unique_models"}
    colors = {1: "#1f77b4", 5: "#ff7f0e", 15: "#2ca02c"}

    for k in k_values:
        ax.plot(
            iterations,
            improvements[k],
            label=labels.get(k, f"Top_{k}"),
            color=colors.get(k, None),
            linewidth=2
        )

    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Perplexity Improvement", fontsize=12)
    ax.set_title(
        "Improvement of Neural Architecture Search over Random Search",
        fontsize=11
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Search progress plot saved to {output_path}")


def plot_reward_over_time(nas_rewards, output_path, window=50):
    """
    Plot rolling average reward over search iterations.

    Args:
        nas_rewards: list of rewards
        output_path: save path
        window: rolling average window size
    """
    rewards = np.array(nas_rewards)
    rolling_avg = np.convolve(rewards, np.ones(window) / window, mode="valid")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(rewards, alpha=0.3, color="blue", label="Raw reward")
    ax.plot(
        range(window - 1, len(rewards)),
        rolling_avg,
        color="red",
        linewidth=2,
        label=f"Rolling avg (window={window})"
    )
    ax.set_xlabel("Architecture index")
    ax.set_ylabel("Reward")
    ax.set_title("Controller reward over search")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Reward plot saved to {output_path}")