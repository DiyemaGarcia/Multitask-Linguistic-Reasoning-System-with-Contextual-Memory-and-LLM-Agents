# utils/logger.py

"""
Structured logging for NAS experiments.
Logs architecture search progress, rewards, and evaluation metrics.
"""

import os
import csv
import json
import time
import logging
from datetime import datetime


class NASLogger:
    """
    Logger for Neural Architecture Search experiments.
    Logs to console, CSV file, and JSON summary.
    """

    def __init__(self, log_dir, experiment_name="nas_experiment"):
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        os.makedirs(log_dir, exist_ok=True)

        # Timestamp
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = f"{experiment_name}_{ts}"

        # Console logger
        self.logger = logging.getLogger(self.run_name)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter(
                "%(asctime)s [%(levelname)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            ))
            self.logger.addHandler(ch)

            fh = logging.FileHandler(
                os.path.join(log_dir, f"{self.run_name}.log")
            )
            fh.setFormatter(logging.Formatter(
                "%(asctime)s [%(levelname)s] %(message)s"
            ))
            self.logger.addHandler(fh)

        # CSV log
        self.csv_path = os.path.join(log_dir, f"{self.run_name}_search.csv")
        self._init_csv()

        # Metrics storage
        self.metrics = {
            "rewards": [],
            "baseline": [],
            "loss": [],
            "best_reward": -float("inf"),
            "best_architecture": None
        }
        self.start_time = time.time()

    def _init_csv(self):
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "iteration", "arch_index", "reward",
                "baseline", "loss", "elapsed_s"
            ])

    def log_step(self, iteration, arch_index, reward, baseline, loss):
        """Log one architecture evaluation."""
        elapsed = time.time() - self.start_time

        self.metrics["rewards"].append(float(reward))
        self.metrics["baseline"].append(float(baseline) if baseline else 0.0)
        self.metrics["loss"].append(float(loss))

        if reward > self.metrics["best_reward"]:
            self.metrics["best_reward"] = float(reward)

        # CSV
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                iteration, arch_index,
                f"{reward:.6f}",
                f"{baseline:.6f}" if baseline else "0.0",
                f"{loss:.6f}",
                f"{elapsed:.1f}"
            ])

        # Console (every 100 steps)
        if arch_index % 100 == 0:
            self.logger.info(
                f"Iter {iteration} | Arch {arch_index} | "
                f"Reward {reward:.4f} | Baseline {baseline:.4f} | "
                f"Loss {loss:.4f} | Elapsed {elapsed:.0f}s"
            )

    def log_evaluation(self, metric_name, value):
        """Log a final evaluation metric."""
        self.logger.info(f"FINAL {metric_name}: {value}")
        self.metrics[metric_name] = value

    def save_summary(self):
        """Save experiment summary to JSON."""
        summary_path = os.path.join(self.log_dir, f"{self.run_name}_summary.json")
        summary = {
            "run_name": self.run_name,
            "total_time_s": time.time() - self.start_time,
            "total_architectures": len(self.metrics["rewards"]),
            "best_reward": self.metrics["best_reward"],
            "mean_reward": float(sum(self.metrics["rewards"]) /
                                 max(len(self.metrics["rewards"]), 1)),
            "final_metrics": {
                k: v for k, v in self.metrics.items()
                if k not in ["rewards", "baseline", "loss"]
            }
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        self.logger.info(f"Summary saved to {summary_path}")
        return summary