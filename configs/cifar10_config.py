# configs/cifar10_config.py

CIFAR10_CONFIG = {
    # Controller RNN
    "controller_num_layers": 2,
    "controller_hidden_units": 35,
    "controller_lr": 0.0006,
    "controller_weight_init": (-0.08, 0.08),
    "controller_optimizer": "adam",

    # Distributed training
    "num_param_server_shards": 20,      # S
    "num_controller_replicas": 100,      # K
    "num_child_replicas": 8,             # m (child par replica)
    # => 800 GPUs concurrents

    # Child network training
    "child_epochs": 50,
    "child_lr": 0.1,
    "child_weight_decay": 1e-4,
    "child_momentum": 0.9,
    "child_optimizer": "momentum_nesterov",
    "child_lr_decay_schedule": True,

    # Reward
    "reward_fn": "max_val_acc_last5_cubed",  # R = max(val_acc[-5:])^3

    # Validation split
    "val_size": 5000,
    "train_size": 45000,

    # Search space — filters
    "filter_heights": [1, 3, 5, 7],
    "filter_widths": [1, 3, 5, 7],
    "num_filters": [24, 36, 48, 64],
    "strides": [1],                      # v1: fixed; v2: [1,2,3]

    # Architecture depth schedule
    "initial_depth": 6,
    "depth_increase_every": 1600,        # samples
    "depth_increase_amount": 2,

    # Total architectures sampled
    "total_architectures": 12800,

    # Baseline
    "baseline_decay": 0.999,             # exponential moving average

    # Data preprocessing
    "whitening": True,
    "random_crop": True,
    "crop_size": 32,
    "upsample_size": 40,
    "random_horizontal_flip": True,

    # Grid search (final model)
    "grid_search_params": ["lr", "weight_decay", "batchnorm_epsilon", "lr_decay_epoch"],
}