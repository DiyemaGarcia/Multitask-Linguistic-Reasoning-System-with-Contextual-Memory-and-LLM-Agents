# evaluation/evaluate_ptb.py

"""
Final evaluation of the best PTB recurrent cell found by NAS.
"""

import torch
import torch.nn as nn
import numpy as np
import math
import json
import os


def evaluate_final_ptb(model, test_data, bptt=35, batch_size=20, device="cpu"):
    """
    Compute final test perplexity on Penn Treebank test set.

    Args:
        model: trained NASRNNModel
        test_data: tokenized test data tensor [batch_size, seq_len]
        bptt: backpropagation through time length (35)
        batch_size: evaluation batch size
        device: computation device

    Returns:
        test_perplexity: float
    """
    model.eval()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    num_batches = 0
    hidden = model.init_hidden(batch_size)

    with torch.no_grad():
        for i in range(0, test_data.size(1) - 1, bptt):
            seq_len = min(bptt, test_data.size(1) - 1 - i)
            data = test_data[:, i:i + seq_len].to(device)
            targets = test_data[:, i + 1:i + seq_len + 1].to(device).reshape(-1)

            logits, hidden = model(data.t(), hidden)
            loss = criterion(logits, targets)
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    perplexity = math.exp(avg_loss)

    print(f"PTB Test Perplexity: {perplexity:.2f}")
    return perplexity


def evaluate_character_ptb(model, test_data, bptt=100,
                             batch_size=32, device="cpu"):
    """
    Evaluate character-level language model on PTB.
    Returns bits-per-character (BPC).

    BPC = test_loss / log(2)
    """
    model.eval()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    num_batches = 0
    hidden = model.init_hidden(batch_size)

    with torch.no_grad():
        for i in range(0, test_data.size(1) - 1, bptt):
            seq_len = min(bptt, test_data.size(1) - 1 - i)
            data = test_data[:, i:i + seq_len].to(device)
            targets = test_data[:, i + 1:i + seq_len + 1].to(device).reshape(-1)

            logits, hidden = model(data.t(), hidden)
            loss = criterion(logits, targets)
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    bpc = avg_loss / math.log(2)

    print(f"PTB Character-level BPC: {bpc:.3f}")
    return bpc


def generate_perplexity_table(nas_results, baselines):
    """
    Generate comparison table matching Table 2 from the paper.

    Args:
        nas_results: dict {model_name: {"params": ..., "perplexity": ...}}
        baselines: dict with baseline model results

    Returns:
        table_str: formatted string
    """
    lines = []
    lines.append("=" * 65)
    lines.append(f"{'Model':<45} {'Params':>8} {'PPL':>8}")
    lines.append("=" * 65)

    for name, info in baselines.items():
        params = info.get("params", "-")
        ppl = str(info.get("perplexity", "-"))
        lines.append(f"{name:<45} {params:>8} {ppl:>8}")

    lines.append("-" * 65)

    for name, info in nas_results.items():
        params = info.get("params", "-")
        ppl = f"{info.get('perplexity', 0.0):.1f}"
        lines.append(f"{name:<45} {params:>8} {ppl:>8}")

    lines.append("=" * 65)
    return "\n".join(lines)


def save_ptb_results(results_dict, output_path):
    """Save PTB evaluation results."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"PTB results saved to {output_path}")