# utils/metrics.py

"""
Metric computation utilities for NAS evaluation.
"""

import torch
import numpy as np
import math


def compute_accuracy(outputs, targets):
    """
    Compute top-1 classification accuracy.

    Args:
        outputs: [N, num_classes] logit tensor
        targets: [N] integer label tensor

    Returns:
        accuracy: float (0.0 to 1.0)
    """
    _, predicted = outputs.max(1)
    correct = predicted.eq(targets).sum().item()
    return correct / targets.size(0)


def compute_top5_accuracy(outputs, targets):
    """
    Compute top-5 classification accuracy.

    Args:
        outputs: [N, num_classes] logit tensor
        targets: [N] integer label tensor

    Returns:
        top5_acc: float
    """
    _, top5_pred = outputs.topk(5, dim=1)
    targets_expanded = targets.view(-1, 1).expand_as(top5_pred)
    correct = top5_pred.eq(targets_expanded).any(dim=1).sum().item()
    return correct / targets.size(0)


def compute_perplexity(loss):
    """
    Compute perplexity from cross-entropy loss.

    Args:
        loss: float (cross-entropy loss in nats)

    Returns:
        perplexity: float
    """
    return math.exp(loss)


def compute_bpc(loss):
    """
    Compute bits-per-character from cross-entropy loss.

    Args:
        loss: float (cross-entropy loss in nats)

    Returns:
        bpc: float
    """
    return loss / math.log(2)


def cifar10_reward(val_accs_last5):
    """
    Compute NAS reward for CIFAR-10 child model.
    Reward = max(val_acc[-5:])^3

    Args:
        val_accs_last5: list of last 5 validation accuracies

    Returns:
        reward: float
    """
    return max(val_accs_last5) ** 3


def ptb_reward(val_perplexity, c=80):
    """
    Compute NAS reward for PTB child model.
    Paper: "reward function is (validation perplexity)^c, c usually set at 80"
    Since we want to maximize reward and minimize perplexity,
    we interpret this as: reward = val_perplexity^(-c) or equivalently
    reward = -val_perplexity (simplified for controller training).

    Args:
        val_perplexity: float (positive, lower is better)
        c: exponent constant (80)

    Returns:
        reward: float (higher is better)
    """
    # To maximize: reward should increase as perplexity decreases
    # Interpretation consistent with REINFORCE maximization
    return -val_perplexity


def count_parameters(model):
    """Count trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters_M(model):
    """Count trainable parameters in millions."""
    return count_parameters(model) / 1e6


def compare_top_k(results_a, results_b, k_list=(1, 5, 15)):
    """
    Compare top-k average rewards between two search methods (e.g. NAS vs random).

    Args:
        results_a: list of reward floats from method A
        results_b: list of reward floats from method B
        k_list: tuple of k values to compare

    Returns:
        comparison: dict {k: {"a": avg_top_k_a, "b": avg_top_k_b, "diff": diff}}
    """
    comparison = {}
    for k in k_list:
        top_a = np.mean(sorted(results_a, reverse=True)[:k])
        top_b = np.mean(sorted(results_b, reverse=True)[:k])
        comparison[k] = {
            "method_a": float(top_a),
            "method_b": float(top_b),
            "diff_a_minus_b": float(top_a - top_b)
        }
    return comparison