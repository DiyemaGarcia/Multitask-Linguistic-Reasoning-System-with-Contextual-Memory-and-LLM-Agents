# configs/ptb_config.py

PTB_CONFIG = {
    # Controller RNN
    "controller_num_layers": 2,
    "controller_hidden_units": 35,
    "controller_lr": 0.0005,
    "controller_weight_init": (-0.08, 0.08),
    "controller_optimizer": "adam",

    # Distributed training
    "num_param_server_shards": 20,       # S
    "num_controller_replicas": 400,       # K
    "num_child_replicas": 1,              # m
    "gradient_accumulation": 10,          # accumulate 10 gradients before update

    # Child network training
    "child_epochs": 35,
    "child_num_layers": 2,
    "child_hidden_units": "adjusted_to_match_medium",  # ~10M params
    "child_optimizer": "SGD",

    # Regularization (Zaremba et al. 2014 + Gal 2015)
    "embedding_dropout": True,
    "recurrent_dropout": True,
    "shared_embeddings": False,          # True for "shared embeddings" variant

    # Reward
    "reward_fn": "neg_val_perplexity_power_c",
    "reward_c": 80,                      # R = (val_perplexity)^(-80) ... or formulation below
    # Exact formulation from paper: reward = (val_perplexity)^c, c typically negative
    # Paper says: "reward function is (validation perplexity)^c where c is a constant, usually set at 80"
    # => interpreted as R = perplexity^(-1) scaled, maximize => minimize perplexity

    # Search space
    "base_number": 8,                    # 8 leaf nodes => ~6e16 architectures
    "combination_methods": ["add", "elem_mult"],
    "activation_functions": ["identity", "tanh", "sigmoid", "relu"],

    # Vocabulary
    "vocab_size": 10000,
    "eos_token": "<eos>",

    # BPTT
    "bptt_length": 35,                   # standard for PTB

    # Baseline
    "baseline_decay": 0.999,

    # Control experiment extensions
    "extended_combination_methods": ["add", "elem_mult", "max"],
    "extended_activation_functions": ["identity", "tanh", "sigmoid", "relu", "sin"],

    # Character-level final experiment
    "char_model_params": 16_280_000,
    "char_weight_decay": 1e-4,
    "char_train_steps": 600_000,
    "char_optimizer": "adam",
    "char_lr": 0.001,
    "char_embedding_size": 128,
    "char_hidden_units": 800,
    "char_num_layers": 2,
    "char_dropout_rates": [0.2, 0.5],
    "char_bptt_length": 100,
    "char_batch_size": 32,
}