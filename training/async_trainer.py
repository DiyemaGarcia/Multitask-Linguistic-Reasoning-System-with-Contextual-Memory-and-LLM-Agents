# training/async_trainer.py

"""
Simulated asynchronous distributed training for Neural Architecture Search.

In the original paper, 800 GPUs (CIFAR-10) or 400 CPUs (PTB) are used.
Here we simulate the asynchronous parameter update scheme using
Python multiprocessing / threading, adapted for single-machine execution.

The parameter server scheme:
    - S parameter server shards store controller weights
    - K controller replicas each sample m child architectures
    - Each replica trains m children in parallel
    - Gradients are sent back to parameter server asynchronously
"""

import torch
import torch.multiprocessing as mp
import numpy as np
import time
import copy
from queue import Queue
from threading import Thread


class SimulatedParameterServer:
    """
    Simulates a parameter server for the controller RNN.
    Accumulates gradients from multiple replicas and updates controller.
    """

    def __init__(self, controller, optimizer, gradient_accumulation=1):
        self.controller = controller
        self.optimizer = optimizer
        self.gradient_accumulation = gradient_accumulation
        self.accumulated_grads = []
        self.lock = mp.Lock() if hasattr(mp, 'Lock') else None

    def receive_gradient(self, gradients, reward):
        """Receive gradient update from a controller replica."""
        self.accumulated_grads.append((gradients, reward))

        if len(self.accumulated_grads) >= self.gradient_accumulation:
            self._apply_gradients()

    def _apply_gradients(self):
        """Apply accumulated gradients to controller."""
        self.optimizer.zero_grad()

        # Average gradients
        for param_group in self.controller.parameters():
            param_group.grad = None

        # Apply each gradient update
        for grads, reward in self.accumulated_grads:
            for param, grad in zip(self.controller.parameters(), grads):
                if grad is not None:
                    if param.grad is None:
                        param.grad = grad.clone() / len(self.accumulated_grads)
                    else:
                        param.grad += grad / len(self.accumulated_grads)

        torch.nn.utils.clip_grad_norm_(self.controller.parameters(), max_norm=5.0)
        self.optimizer.step()
        self.accumulated_grads = []

    def get_current_weights(self):
        """Return current controller weights for replica synchronization."""
        return copy.deepcopy(self.controller.state_dict())


class AsyncNASTrainer:
    """
    Simulates the asynchronous NAS training loop on a single machine.

    For real-world replication with limited hardware:
    - Runs sequentially instead of truly parallel
    - Simulates the K replicas * m child pattern
    - Accumulates gradients before updating controller
    """

    def __init__(self, controller, trainer, config, task="cifar10"):
        self.controller = controller
        self.trainer = trainer
        self.config = config
        self.task = task

        # From paper:
        # CIFAR-10: K=100, m=8 => 800 concurrent
        # PTB: K=400, m=1 => 400 concurrent
        # We simulate with reduced numbers
        self.K = config.get("num_controller_replicas", 100)
        self.m = config.get("num_child_replicas", 8)
        self.gradient_acc = config.get("gradient_accumulation", 1)

        self.search_results = []

    def run_single_step(self, sample_fn, train_child_fn, num_layers=None):
        """
        Run one NAS step: sample m architectures, train them, update controller.

        Args:
            sample_fn: function that samples architecture from controller
            train_child_fn: function that trains a child and returns reward
            num_layers: number of layers for current step (CIFAR-10)

        Returns:
            rewards: list of rewards from this step
            architectures: list of sampled architectures
        """
        rewards = []
        log_probs_batch = []
        architectures = []

        # Sample m architectures
        for _ in range(self.m):
            if self.task == "cifar10":
                arch, log_probs = self.controller.forward_cifar10(
                    num_layers=num_layers
                )
            else:
                arch, log_probs = self.controller.forward_ptb(
                    base_number=self.config.get("base_number", 8)
                )

            architectures.append(arch)
            log_probs_batch.append(log_probs)

            # Train child (sequentially in simulation)
            reward = train_child_fn(arch)
            rewards.append(reward)

            self.search_results.append({
                "architecture": arch,
                "reward": reward
            })

        # Update controller with REINFORCE
        loss = self.trainer.step(log_probs_batch, rewards)

        return rewards, architectures, loss

    def run_search(self, total_architectures, sample_fn, train_child_fn,
                   depth_schedule=None):
        """
        Run full NAS search for total_architectures evaluations.

        Args:
            total_architectures: total number of child networks to evaluate
            sample_fn: architecture sampling function
            train_child_fn: child training function
            depth_schedule: dict mapping arch_count -> num_layers

        Returns:
            all_results: list of (architecture, reward) tuples
        """
        num_evaluated = 0
        iteration = 0
        current_depth = self.config.get("initial_depth", 6)

        print(f"Starting NAS search for {total_architectures} architectures...")

        while num_evaluated < total_architectures:
            # Update depth schedule (CIFAR-10)
            if depth_schedule is not None:
                depth_increase_every = self.config.get("depth_increase_every", 1600)
                depth_increase_amount = self.config.get("depth_increase_amount", 2)
                if num_evaluated > 0 and num_evaluated % depth_increase_every == 0:
                    current_depth += depth_increase_amount
                    print(f"Increasing depth to {current_depth} at iteration {num_evaluated}")

            # Run one step
            rewards, archs, loss = self.run_single_step(
                sample_fn, train_child_fn,
                num_layers=current_depth if self.task == "cifar10" else None
            )

            num_evaluated += len(rewards)
            iteration += 1

            if iteration % 10 == 0:
                print(
                    f"Iteration {iteration} | "
                    f"Evaluated {num_evaluated}/{total_architectures} | "
                    f"Loss: {loss:.4f} | "
                    f"Mean Reward: {np.mean(rewards):.4f} | "
                    f"Baseline: {self.trainer.baseline:.4f}"
                )

        print(f"Search complete. Total evaluated: {num_evaluated}")
        return self.search_results