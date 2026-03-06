# evaluation/evaluate_cifar10.py

"""
Final evaluation of the best CIFAR-10 architecture found by NAS.
Trains the best model until convergence with optimal hyperparameters.
"""

import torch
import torch.nn as nn
import numpy as np
import json
import os


def evaluate_final_cifar10(model, test_loader, device="cpu"):
    """
    Compute final test accuracy and error rate on CIFAR-10 test set.

    Args:
        model: trained PyTorch model
        test_loader: DataLoader for CIFAR-10 test set (10,000 samples)
        device: computation device

    Returns:
        test_acc: float (accuracy 0-1)
        test_error: float (error rate 0-1)
        test_error_pct: float (error rate in %)
    """
    model.eval()
    model = model.to(device)

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    test_acc = correct / total
    test_error = 1.0 - test_acc
    test_error_pct = test_error * 100.0

    print(f"CIFAR-10 Test Accuracy: {test_acc * 100:.2f}%")
    print(f"CIFAR-10 Test Error:    {test_error_pct:.2f}%")

    return test_acc, test_error, test_error_pct


def count_parameters(model):
    """Count total trainable parameters in model."""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_M = total / 1e6
    print(f"Model parameters: {total:,} ({total_M:.1f}M)")
    return total


def measure_inference_speed(model, input_shape=(1, 3, 32, 32),
                              num_runs=100, device="cpu"):
    """
    Measure inference speed relative to DenseNet baseline.
    Paper states NAS v3 is 1.05x faster than DenseNet (3.74% error).

    Returns:
        avg_ms: average inference time in milliseconds
    """
    import time
    model.eval()
    model = model.to(device)
    dummy_input = torch.randn(*input_shape).to(device)

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)

    # Measure
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(dummy_input)
        end = time.perf_counter()
        times.append((end - start) * 1000)

    avg_ms = np.mean(times)
    std_ms = np.std(times)
    print(f"Inference time: {avg_ms:.3f} ± {std_ms:.3f} ms (over {num_runs} runs)")
    return avg_ms


def generate_results_table(nas_results, baselines):
    """
    Generate comparison table matching Table 1 from the paper.

    Args:
        nas_results: dict with NAS variant results
        baselines: dict with baseline model results

    Returns:
        table_str: formatted string table
    """
    lines = []
    lines.append("=" * 75)
    lines.append(f"{'Model':<50} {'Depth':>6} {'Params':>8} {'Error%':>8}")
    lines.append("=" * 75)

    for name, info in baselines.items():
        depth = str(info.get("depth", "-"))
        params = info.get("params", "-")
        error = info.get("error", "-")
        lines.append(f"{name:<50} {depth:>6} {params:>8} {error:>8}")

    lines.append("-" * 75)

    for name, info in nas_results.items():
        depth = str(info.get("depth", "-"))
        params = info.get("params", "-")
        error = f"{info.get('error', 0.0):.2f}"
        lines.append(f"{name:<50} {depth:>6} {params:>8} {error:>8}")

    lines.append("=" * 75)
    return "\n".join(lines)


def save_evaluation_results(results_dict, output_path):
    """Save evaluation results to JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"Evaluation results saved to {output_path}")