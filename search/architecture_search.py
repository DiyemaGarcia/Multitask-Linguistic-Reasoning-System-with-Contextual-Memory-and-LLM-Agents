# search/architecture_search.py

import torch
import numpy as np
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from controller.controller_rnn import ControllerRNN
from controller.reinforce import REINFORCETrainer
from training.async_trainer import AsyncNASTrainer


class NASSearch:
    """
    Orchestrates the full Neural Architecture Search process.
    """

    def __init__(self, config, task="cifar10", device="cpu"):
        self.config = config
        self.task = task
        self.device = device

        # Initialize controller
        self.controller = ControllerRNN(config, task=task).to(device)

        # Initialize REINFORCE trainer
        self.reinforce_trainer = REINFORCETrainer(self.controller, config)

        # Initialize async trainer
        self.async_trainer = AsyncNASTrainer(
            self.controller,
            self.reinforce_trainer,
            config,
            task=task
        )

        self.all_results = []
        self.best_architecture = None
        self.best_reward = -float("inf")

    def run(self, train_child_fn, total_architectures=None):
        """
        Run the full NAS search.

        Args:
            train_child_fn: callable(architecture) -> reward
            total_architectures: number of architectures to evaluate
                                 (default from config)

        Returns:
            best_architecture: dict
            best_reward: float
            all_results: list of dicts
        """
        if total_architectures is None:
            total_architectures = self.config.get("total_architectures", 12800)

        depth_schedule = True if self.task == "cifar10" else None

        results = self.async_trainer.run_search(
            total_architectures=total_architectures,
            sample_fn=None,           # handled internally
            train_child_fn=train_child_fn,
            depth_schedule=depth_schedule
        )

        self.all_results = results

        # Find best architecture
        for entry in results:
            if entry["reward"] > self.best_reward:
                self.best_reward = entry["reward"]
                self.best_architecture = entry["architecture"]

        return self.best_architecture, self.best_reward, self.all_results

    def save_results(self, output_dir):
        """Save search results to JSON files."""
        os.makedirs(output_dir, exist_ok=True)

        # Save best architecture
        best_path = os.path.join(output_dir, "best_architecture.json")
        with open(best_path, "w") as f:
            json.dump({
                "architecture": self.best_architecture,
                "reward": self.best_reward
            }, f, indent=2)

        # Save all results (convert tensors to Python types)
        all_path = os.path.join(output_dir, "all_architectures.json")
        serializable_results = []
        for entry in self.all_results:
            serializable_results.append({
                "architecture": _make_serializable(entry["architecture"]),
                "reward": float(entry["reward"])
            })
        with open(all_path, "w") as f:
            json.dump(serializable_results, f, indent=2)

        # Save search log as CSV
        log_path = os.path.join(output_dir, "search_log.csv")
        with open(log_path, "w") as f:
            f.write("index,reward\n")
            for i, entry in enumerate(self.all_results):
                f.write(f"{i},{float(entry['reward']):.6f}\n")

        print(f"Results saved to {output_dir}")

    def load_results(self, output_dir):
        """Load previously saved search results."""
        best_path = os.path.join(output_dir, "best_architecture.json")
        if os.path.exists(best_path):
            with open(best_path, "r") as f:
                data = json.load(f)
                self.best_architecture = data["architecture"]
                self.best_reward = data["reward"]
        return self.best_architecture, self.best_reward


def _make_serializable(obj):
    """Recursively convert non-serializable types to Python natives."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, torch.Tensor):
        return obj.item()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    else:
        return obj