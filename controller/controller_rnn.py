# controller/controller_rnn.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ControllerRNN(nn.Module):
    """
    Controller RNN for Neural Architecture Search.
    Implements a 2-layer LSTM that generates architecture hyperparameters
    as a sequence of tokens (one softmax decision per step).

    For CIFAR-10: predicts filter_height, filter_width, stride_h, stride_w, num_filters
                  per layer, repeated for each layer.
    For PTB: predicts combination_method + activation_function per tree node,
             plus cell injection indices.
    """

    def __init__(self, config, task="cifar10"):
        super(ControllerRNN, self).__init__()
        self.task = task
        self.config = config
        self.hidden_size = config["controller_hidden_units"]  # 35
        self.num_layers = config["controller_num_layers"]     # 2

        # 2-layer LSTM controller
        self.lstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )

        if task == "cifar10":
            self._build_cifar10_heads(config)
        elif task == "ptb":
            self._build_ptb_heads(config)

        # Initialize weights uniformly in [-0.08, 0.08]
        self._init_weights()

    def _build_cifar10_heads(self, config):
        """Softmax classifiers for each hyperparameter type."""
        self.filter_height_head = nn.Linear(
            self.hidden_size, len(config["filter_heights"])
        )  # 4 choices: [1,3,5,7]
        self.filter_width_head = nn.Linear(
            self.hidden_size, len(config["filter_widths"])
        )  # 4 choices: [1,3,5,7]
        self.num_filters_head = nn.Linear(
            self.hidden_size, len(config["num_filters"])
        )  # 4 choices: [24,36,48,64]
        self.stride_head = nn.Linear(
            self.hidden_size, len(config["strides"])
        )  # 1 or 3 choices

        # Skip connection attention parameters (Section 3.3)
        self.W_prev = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.W_curr = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v = nn.Linear(self.hidden_size, 1, bias=False)

    def _build_ptb_heads(self, config):
        """Softmax classifiers for RNN cell generation."""
        self.combination_head = nn.Linear(
            self.hidden_size, len(config["combination_methods"])
        )  # 2 choices: [add, elem_mult]
        self.activation_head = nn.Linear(
            self.hidden_size, len(config["activation_functions"])
        )  # 4 choices: [identity, tanh, sigmoid, relu]
        # Cell index predictions (where to inject ct-1 and set ct)
        self.cell_index_head = nn.Linear(
            self.hidden_size, config["base_number"]
        )

    def _init_weights(self):
        """Initialize all weights uniformly in [-0.08, 0.08]."""
        for p in self.parameters():
            nn.init.uniform_(p, -0.08, 0.08)

    def _get_initial_hidden(self, batch_size=1):
        """Zero initial hidden state."""
        device = next(self.parameters()).device
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return (h0, c0)

    def forward_cifar10(self, num_layers, use_skip_connections=True):
        """
        Generate a convolutional architecture description for CIFAR-10.

        Returns:
            actions: dict with lists of sampled hyperparameters
            log_probs: list of log probabilities for REINFORCE
            hidden_states: list of hidden states (for skip connection anchors)
        """
        batch_size = 1
        hidden = self._get_initial_hidden(batch_size)
        device = next(self.parameters()).device

        # Start token: zero embedding
        inp = torch.zeros(batch_size, 1, self.hidden_size, device=device)

        actions = {
            "filter_heights": [],
            "filter_widths": [],
            "num_filters": [],
            "strides": [],
            "skip_connections": []
        }
        log_probs = []
        anchor_hidden_states = []  # h_j for skip connection attention

        config = self.config
        filter_heights = config["filter_heights"]
        filter_widths = config["filter_widths"]
        num_filters_choices = config["num_filters"]
        strides_choices = config["strides"]

        for layer_idx in range(num_layers):
            # --- Filter height ---
            out, hidden = self.lstm(inp, hidden)
            logits = self.filter_height_head(out.squeeze(1))
            probs = F.softmax(logits, dim=-1)
            sampled = torch.multinomial(probs, 1)
            log_probs.append(torch.log(probs[0, sampled[0, 0]]))
            actions["filter_heights"].append(filter_heights[sampled.item()])
            inp = self._embed(sampled, self.hidden_size, device)

            # --- Filter width ---
            out, hidden = self.lstm(inp, hidden)
            logits = self.filter_width_head(out.squeeze(1))
            probs = F.softmax(logits, dim=-1)
            sampled = torch.multinomial(probs, 1)
            log_probs.append(torch.log(probs[0, sampled[0, 0]]))
            actions["filter_widths"].append(filter_widths[sampled.item()])
            inp = self._embed(sampled, self.hidden_size, device)

            # --- Stride ---
            out, hidden = self.lstm(inp, hidden)
            logits = self.stride_head(out.squeeze(1))
            probs = F.softmax(logits, dim=-1)
            sampled = torch.multinomial(probs, 1)
            log_probs.append(torch.log(probs[0, sampled[0, 0]]))
            actions["strides"].append(strides_choices[sampled.item()])
            inp = self._embed(sampled, self.hidden_size, device)

            # --- Num filters ---
            out, hidden = self.lstm(inp, hidden)
            logits = self.num_filters_head(out.squeeze(1))
            probs = F.softmax(logits, dim=-1)
            sampled = torch.multinomial(probs, 1)
            log_probs.append(torch.log(probs[0, sampled[0, 0]]))
            actions["num_filters"].append(num_filters_choices[sampled.item()])
            inp = self._embed(sampled, self.hidden_size, device)

            # --- Skip connections (anchor points) ---
            # Get current hidden state for anchor
            curr_h = hidden[0][-1]  # last layer hidden state, shape [batch, hidden]
            anchor_hidden_states.append(curr_h.detach().clone())

            if use_skip_connections and layer_idx > 0:
                skip_layer = []
                for j in range(layer_idx):
                    h_j = anchor_hidden_states[j]
                    # P(layer j is input to layer i) = sigmoid(v^T tanh(W_prev*h_j + W_curr*h_i))
                    score = self.v(
                        torch.tanh(self.W_prev(h_j) + self.W_curr(curr_h))
                    )  # [batch, 1]
                    prob = torch.sigmoid(score)
                    skip = (torch.bernoulli(prob) > 0.5).float()
                    log_probs.append(
                        skip * torch.log(prob + 1e-8) +
                        (1 - skip) * torch.log(1 - prob + 1e-8)
                    )
                    skip_layer.append(int(skip.item()))
                actions["skip_connections"].append(skip_layer)
            else:
                actions["skip_connections"].append([])

        return actions, log_probs

    def forward_ptb(self, base_number):
        """
        Generate a recurrent cell architecture for PTB.
        base_number: number of leaf nodes (8 in the paper)

        Returns:
            actions: dict with combination methods, activations, cell indices
            log_probs: list of log probs for REINFORCE
        """
        batch_size = 1
        hidden = self._get_initial_hidden(batch_size)
        device = next(self.parameters()).device
        inp = torch.zeros(batch_size, 1, self.hidden_size, device=device)

        config = self.config
        combo_choices = config["combination_methods"]
        act_choices = config["activation_functions"]

        actions = {
            "combination_methods": [],
            "activation_functions": [],
            "cell_ct_index": None,     # where to inject ct-1
            "cell_ct_new_index": None, # where to set ct
        }
        log_probs = []

        num_nodes = 2 * base_number - 1  # total nodes in binary tree

        for node_idx in range(num_nodes):
            # --- Combination method ---
            out, hidden = self.lstm(inp, hidden)
            logits = self.combination_head(out.squeeze(1))
            probs = F.softmax(logits, dim=-1)
            sampled = torch.multinomial(probs, 1)
            log_probs.append(torch.log(probs[0, sampled[0, 0]]))
            actions["combination_methods"].append(combo_choices[sampled.item()])
            inp = self._embed(sampled, self.hidden_size, device)

            # --- Activation function ---
            out, hidden = self.lstm(inp, hidden)
            logits = self.activation_head(out.squeeze(1))
            probs = F.softmax(logits, dim=-1)
            sampled = torch.multinomial(probs, 1)
            log_probs.append(torch.log(probs[0, sampled[0, 0]]))
            actions["activation_functions"].append(act_choices[sampled.item()])
            inp = self._embed(sampled, self.hidden_size, device)

        # --- Cell injection: predict ct-1 index and ct index ---
        # Block for ct-1 (where to inject previous cell state)
        out, hidden = self.lstm(inp, hidden)
        logits = self.cell_index_head(out.squeeze(1))
        probs = F.softmax(logits, dim=-1)
        sampled = torch.multinomial(probs, 1)
        log_probs.append(torch.log(probs[0, sampled[0, 0]]))
        actions["cell_ct_index"] = sampled.item()
        inp = self._embed(sampled, self.hidden_size, device)

        # Block for ct (where to read new cell state)
        out, hidden = self.lstm(inp, hidden)
        logits = self.cell_index_head(out.squeeze(1))
        probs = F.softmax(logits, dim=-1)
        sampled = torch.multinomial(probs, 1)
        log_probs.append(torch.log(probs[0, sampled[0, 0]]))
        actions["cell_ct_new_index"] = sampled.item()

        return actions, log_probs

    def _embed(self, token_idx, hidden_size, device):
        """Simple learned embedding: one-hot projected to hidden_size via linear (approx)."""
        # In the original paper, the sampled token is fed as input to the next step.
        # We use a simple approach: embed the index as a one-hot and project.
        # For simplicity in replication, we use a zero vector (common simplification).
        return torch.zeros(1, 1, hidden_size, device=device)