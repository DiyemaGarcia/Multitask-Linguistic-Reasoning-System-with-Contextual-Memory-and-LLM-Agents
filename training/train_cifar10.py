# training/train_cifar10.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np


def train_child_cifar10(model, train_loader, val_loader, config, device="cpu"):
    """
    Train a child CNN on CIFAR-10 for config["child_epochs"] epochs.

    Uses Nesterov SGD with:
        - lr = 0.1
        - weight_decay = 1e-4
        - momentum = 0.9
        - Nesterov = True

    Reward: max(val_acc[-5:])^3

    Returns:
        reward: float (cubed max validation accuracy over last 5 epochs)
        val_accs: list of validation accuracies per epoch
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=config["child_lr"],           # 0.1
        momentum=config["child_momentum"], # 0.9
        weight_decay=config["child_weight_decay"],  # 1e-4
        nesterov=True
    )

    # Learning rate schedule: decay at 50% and 75% of training
    num_epochs = config["child_epochs"]  # 50
    milestones = [int(0.5 * num_epochs), int(0.75 * num_epochs)]
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    val_accs = []

    for epoch in range(num_epochs):
        # --- Training phase ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()

        scheduler.step()

        # --- Validation phase ---
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

        val_acc = val_correct / val_total
        val_accs.append(val_acc)

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}] "
                f"Train Loss: {train_loss/len(train_loader):.4f} "
                f"Val Acc: {val_acc:.4f}"
            )

    # Compute reward: max(val_acc[-5:])^3
    last5_acc = val_accs[-5:]
    reward = max(last5_acc) ** 3

    return reward, val_accs


def grid_search_cifar10(best_architecture, train_loader, val_loader, test_loader,
                         build_fn, config, device="cpu"):
    """
    Run small grid search over hyperparameters for the best architecture.

    Searches over:
        - learning_rate
        - weight_decay
        - batchnorm_epsilon
        - lr_decay_epoch

    Returns best model and its test accuracy.
    """
    lr_choices = [0.05, 0.1, 0.025]
    wd_choices = [1e-4, 5e-4]
    bn_eps_choices = [1e-5, 1e-3]

    best_val_acc = 0.0
    best_model = None
    best_hyperparams = {}

    for lr in lr_choices:
        for wd in wd_choices:
            for bn_eps in bn_eps_choices:
                cfg = dict(config)
                cfg["child_lr"] = lr
                cfg["child_weight_decay"] = wd

                model = build_fn(best_architecture)
                # Set batchnorm epsilon
                for module in model.modules():
                    if isinstance(module, nn.BatchNorm2d):
                        module.eps = bn_eps

                reward, val_accs = train_child_cifar10(
                    model, train_loader, val_loader, cfg, device
                )
                val_acc = max(val_accs)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model = model
                    best_hyperparams = {
                        "lr": lr, "weight_decay": wd, "bn_eps": bn_eps
                    }

    # Compute test accuracy with best model
    best_model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = best_model(inputs)
            _, predicted = outputs.max(1)
            test_total += targets.size(0)
            test_correct += predicted.eq(targets).sum().item()

    test_acc = test_correct / test_total
    test_error = 1 - test_acc

    print(f"Best hyperparams: {best_hyperparams}")
    print(f"Test Error: {test_error * 100:.2f}%")

    return best_model, test_error, best_hyperparams