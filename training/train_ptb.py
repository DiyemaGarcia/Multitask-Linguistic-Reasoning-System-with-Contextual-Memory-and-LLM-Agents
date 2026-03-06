# training/train_ptb.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math


def train_child_ptb(model, train_data, val_data, config, device="cpu"):
    """
    Train a child RNN on Penn Treebank for config["child_epochs"] epochs.

    Reward: (val_perplexity)^c where c is usually -1 (we want to maximize,
    so we minimize perplexity => reward = -perplexity, or as in paper,
    reward = perplexity^c with c negative)

    Paper says: "reward function is (validation perplexity)^c where c is
    a constant, usually set at 80" — interpreted as inverse scaling.

    Returns:
        reward: float
        val_perplexities: list of validation perplexities per epoch
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=config.get("child_lr_ptb", 1.0),
        weight_decay=0
    )

    num_epochs = config["child_epochs"]   # 35
    bptt = config.get("bptt_length", 35)
    batch_size = config.get("batch_size_ptb", 20)

    val_perplexities = []

    for epoch in range(num_epochs):
        # --- Training phase ---
        model.train()
        total_loss = 0.0
        hidden = model.init_hidden(batch_size)
        num_batches = 0

        for i in range(0, train_data.size(1) - 1, bptt):
            # Get batch
            seq_len = min(bptt, train_data.size(1) - 1 - i)
            data = train_data[:, i:i + seq_len].to(device)
            targets = train_data[:, i + 1:i + seq_len + 1].to(device).reshape(-1)

            optimizer.zero_grad()
            logits, hidden = model(data.t(), hidden)
            loss = criterion(logits, targets)
            loss.backward()

            # Gradient clipping (max norm = 5)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        train_ppl = math.exp(avg_loss)

        # --- Validation phase ---
        val_ppl = evaluate_ptb(model, val_data, bptt, batch_size, criterion, device)
        val_perplexities.append(val_ppl)

        if (epoch + 1) % 5 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}] "
                f"Train PPL: {train_ppl:.2f} Val PPL: {val_ppl:.2f}"
            )

    # Compute reward
    c = config.get("reward_c", 80)
    best_val_ppl = min(val_perplexities)
    # Paper: reward = (val_perplexity)^c — since we maximize reward and want low perplexity,
    # we negate: reward = - val_perplexity (simplified interpretation)
    reward = -best_val_ppl

    return reward, val_perplexities


def evaluate_ptb(model, data, bptt, batch_size, criterion, device):
    """Evaluate model on PTB data, return perplexity."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    hidden = model.init_hidden(batch_size)

    with torch.no_grad():
        for i in range(0, data.size(1) - 1, bptt):
            seq_len = min(bptt, data.size(1) - 1 - i)
            input_data = data[:, i:i + seq_len].to(device)
            targets = data[:, i + 1:i + seq_len + 1].to(device).reshape(-1)

            logits, hidden = model(input_data.t(), hidden)
            loss = criterion(logits, targets)
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    perplexity = math.exp(avg_loss)
    return perplexity


def grid_search_ptb(best_cell_architecture, train_data, val_data, test_data,
                     build_fn, config, device="cpu"):
    """
    Run grid search over lr, weight_init, dropout_rates, decay_epoch
    for the best cell found.
    """
    lr_choices = [0.5, 1.0, 2.0]
    dropout_choices = [0.5, 0.65, 0.75]
    decay_epoch_choices = [20, 30]

    best_val_ppl = float("inf")
    best_model = None
    best_hyperparams = {}

    for lr in lr_choices:
        for dropout in dropout_choices:
            for decay_ep in decay_epoch_choices:
                cfg = dict(config)
                cfg["child_lr_ptb"] = lr
                cfg["child_dropout"] = dropout
                cfg["decay_epoch"] = decay_ep

                model = build_fn(best_cell_architecture, cfg)
                _, val_ppls = train_child_ptb(
                    model, train_data, val_data, cfg, device
                )
                val_ppl = min(val_ppls)

                if val_ppl < best_val_ppl:
                    best_val_ppl = val_ppl
                    best_model = model
                    best_hyperparams = {
                        "lr": lr, "dropout": dropout, "decay_epoch": decay_ep
                    }

    # Compute test perplexity
    criterion = nn.CrossEntropyLoss()
    test_ppl = evaluate_ptb(
        best_model, test_data,
        config.get("bptt_length", 35),
        config.get("batch_size_ptb", 20),
        criterion, device
    )

    print(f"Best hyperparams: {best_hyperparams}")
    print(f"Test Perplexity: {test_ppl:.2f}")

    return best_model, test_ppl, best_hyperparams