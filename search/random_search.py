# search/random_search.py

"""
Random search baseline for Neural Architecture Search (Section 4.2, Control Experiment 2).

Instead of the learned controller (policy gradient), architectures are sampled
uniformly at random. Used to demonstrate that NAS outperforms random search.
"""

import numpy as np
import json
import os


class RandomSearch:
    """
    Random architecture search baseline.
    Samples architectures uniformly at random from the same search space
    as the NAS controller.
    """

    def __init__(self, config, task="cifar10"):
        self.config = config
        self.task = task
        self.all_results = []

    def sample_cifar10_architecture(self, num_layers):
        """
        Sample a random convolutional architecture for CIFAR-10.

        Returns:
            architecture: dict with randomly sampled hyperparameters
        """
        filter_heights = self.config["filter_heights"]
        filter_widths = self.config["filter_widths"]
        num_filters_choices = self.config["num_filters"]
        strides_choices = self.config["strides"]

        architecture = {
            "filter_heights": [
                np.random.choice(filter_heights) for _ in range(num_layers)
            ],
            "filter_widths": [
                np.random.choice(filter_widths) for _ in range(num_layers)
            ],
            "num_filters": [
                np.random.choice(num_filters_choices) for _ in range(num_layers)
            ],
            "strides": [
                np.random.choice(strides_choices) for _ in range(num_layers)
            ],
            "skip_connections": [
                [np.random.randint(0, 2) for _ in range(i)]
                for i in range(num_layers)
            ]
        }
        return architecture

    def sample_ptb_cell(self):
        """
        Sample a random recurrent cell architecture for PTB.

        Returns:
            cell_architecture: dict with randomly sampled operations
        """
        base_number = self.config.get("base_number", 8)
        combo_choices = self.config["combination_methods"]
        act_choices = self.config["activation_functions"]

        num_nodes = 2 * base_number - 1

        cell_architecture = {
            "combination_methods": [
                np.random.choice(combo_choices) for _ in range(num_nodes)
            ],
            "activation_functions": [
                np.random.choice(act_choices) for _ in range(num_nodes)
            ],
            "cell_ct_index": np.random.randint(0, base_number),
            "cell_ct_new_index": np.random.randint(0, base_number)
        }
        return cell_architecture

    def run(self, train_child_fn, total_architectures, num_layers=None):
        """
        Run random search for total_architectures evaluations.

        Args:
            train_child_fn: callable(architecture) -> reward
            total_architectures: number of random architectures to evaluate
            num_layers: fixed depth for CIFAR-10 (or None for PTB)

        Returns:
            best_architecture: dict
            best_reward: float
            all_results: list of dicts
        """
        best_reward = -float("inf")
        best_architecture = None

        print(f"Starting Random Search for {total_architectures} architectures...")

        for i in range(total_architectures):
            # Sample random architecture
            if self.task == "cifar10":
                depth = num_layers if num_layers else self.config.get("initial_depth", 6)
                arch = self.sample_cifar10_architecture(depth)
            else:
                arch = self.sample_ptb_cell()

            # Train child
            reward = train_child_fn(arch)

            self.all_results.append({
                "architecture": arch,
                "reward": float(reward)
            })

            if reward > best_reward:
                best_reward = reward
                best_architecture = arch

            if (i + 1) % 50 == 0:
                print(
                    f"Random Search [{i+1}/{total_architectures}] "
                    f"Best Reward: {best_reward:.4f}"
                )

        print(f"Random Search complete. Best reward: {best_reward:.4f}")
        return best_architecture, best_reward, self.all_results

    def get_top_k_rewards(self, k):
        """Return sorted top-k rewards from the search."""
        rewards = [entry["reward"] for entry in self.all_results]
        rewards_sorted = sorted(rewards, reverse=True)
        return rewards_sorted[:k]

    def save_results(self, output_dir):
        """Save random search results."""
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "random_search_results.json")
        
        # Convert numpy types to native Python types before serialization
        serializable_results = []
        for entry in self.all_results:
            serializable_entry = {}
            for k, v in entry.items():
                if k == "architecture":
                    serializable_arch = {}
                    for arch_k, arch_v in v.items():
                        if isinstance(arch_v, list):
                            serializable_arch[arch_k] = [
                                [int(x) for x in row] if isinstance(row, list)
                                else int(x) if hasattr(x, 'item') else x
                                for x in arch_v
                            ]
                        else:
                            serializable_arch[arch_k] = arch_v
                    serializable_entry[k] = serializable_arch
                elif hasattr(v, 'item'):
                    serializable_entry[k] = v.item()
                else:
                    serializable_entry[k] = v
            serializable_results.append(serializable_entry)
        
        with open(path, "w") as f:
            json.dump(serializable_results, f, indent=2)
        print(f"Random search results saved to {path}")