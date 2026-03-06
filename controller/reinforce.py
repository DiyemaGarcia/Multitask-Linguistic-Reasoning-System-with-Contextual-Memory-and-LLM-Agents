# controller/reinforce.py

import torch
import torch.optim as optim
import numpy as np


class REINFORCETrainer:
    """
    Implements the REINFORCE policy gradient update for the controller RNN.

    Gradient update:
        nabla_theta J(theta) = sum_t [ nabla_theta log P(a_t | a_{t-1:1}; theta) * (R_k - b) ]

    where:
        - R_k is the validation accuracy (CIFAR-10) or -perplexity (PTB) of child k
        - b is an exponential moving average baseline
        - The sum is over T hyperparameter decisions
    """

    def __init__(self, controller, config):
        self.controller = controller
        self.config = config
        self.baseline = None
        self.baseline_decay = config.get("baseline_decay", 0.999)

        lr = config["controller_lr"]
        self.optimizer = optim.Adam(controller.parameters(), lr=lr)

    def update_baseline(self, reward):
        """
        Update exponential moving average baseline.
        b_new = decay * b_old + (1 - decay) * R
        """
        if self.baseline is None:
            self.baseline = reward
        else:
            self.baseline = (
                self.baseline_decay * self.baseline +
                (1 - self.baseline_decay) * reward
            )

    def compute_loss(self, log_probs_batch, rewards_batch):
        """
        Compute REINFORCE loss for a batch of m architectures.

        Args:
            log_probs_batch: list of m lists, each containing T log_prob tensors
            rewards_batch:   list of m scalar rewards R_k

        Returns:
            loss: scalar tensor (negative expected reward)
        """
        m = len(rewards_batch)
        total_loss = torch.tensor(0.0, requires_grad=True)

        for k in range(m):
            R_k = rewards_batch[k]
            advantage = R_k - (self.baseline if self.baseline is not None else 0.0)
            log_probs_k = log_probs_batch[k]

            # Flatten and collect all log probs as scalar tensors
            scalar_log_probs = []
            for lp in log_probs_k:
                if isinstance(lp, torch.Tensor):
                    # Flatten to scalar regardless of original shape
                    for val in lp.reshape(-1):
                        scalar_log_probs.append(val)
                else:
                    scalar_log_probs.append(torch.tensor(float(lp)))

            if len(scalar_log_probs) == 0:
                continue

            sum_log_prob = torch.stack(scalar_log_probs).sum()

            # Negative because we maximize reward (gradient ascent -> descent)
            total_loss = total_loss + (-sum_log_prob * advantage)

        total_loss = total_loss / m
        return total_loss

    def step(self, log_probs_batch, rewards_batch):
        """
        Perform one REINFORCE gradient update.

        Args:
            log_probs_batch: list of m lists of log_prob tensors
            rewards_batch:   list of m scalar rewards

        Returns:
            loss_value: float
        """
        # Update baseline with mean reward of batch
        mean_reward = np.mean(rewards_batch)
        self.update_baseline(mean_reward)

        self.optimizer.zero_grad()
        loss = self.compute_loss(log_probs_batch, rewards_batch)
        loss.backward()

        # Gradient clipping (standard practice for RNNs)
        torch.nn.utils.clip_grad_norm_(self.controller.parameters(), max_norm=5.0)

        self.optimizer.step()
        return loss.item()