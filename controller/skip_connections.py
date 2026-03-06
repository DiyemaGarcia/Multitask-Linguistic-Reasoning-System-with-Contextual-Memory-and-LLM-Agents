# controller/skip_connections.py

import torch
import torch.nn as nn
import numpy as np


class SkipConnectionAttention(nn.Module):
    """
    Set-selection attention mechanism for skip connections (Section 3.3).

    At layer N, adds N-1 content-based sigmoid decisions:
        P(Layer j is input to layer i) = sigmoid(v^T tanh(W_prev * h_j + W_curr * h_i))

    where:
        h_j = hidden state of controller at anchor point for layer j
        h_i = hidden state of controller at current layer i
        W_prev, W_curr, v = trainable parameters
    """

    def __init__(self, hidden_size):
        super(SkipConnectionAttention, self).__init__()
        self.hidden_size = hidden_size
        self.W_prev = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_curr = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, anchor_hidden_states, current_hidden):
        """
        Compute skip connection probabilities for current layer.

        Args:
            anchor_hidden_states: list of previous hidden states [h_0, ..., h_{N-1}]
            current_hidden: current layer hidden state h_N

        Returns:
            skip_probs: tensor of shape [N] with sigmoid probabilities
            skip_decisions: binary tensor of shape [N]
            log_probs: list of log probabilities for REINFORCE
        """
        log_probs = []
        skip_decisions = []
        skip_probs_list = []

        for h_j in anchor_hidden_states:
            # Ensure shapes are compatible
            if h_j.dim() == 1:
                h_j = h_j.unsqueeze(0)
            if current_hidden.dim() == 1:
                current_hidden = current_hidden.unsqueeze(0)

            score = self.v(
                torch.tanh(self.W_prev(h_j) + self.W_curr(current_hidden))
            )  # [1, 1]
            prob = torch.sigmoid(score)  # [1, 1]
            skip_probs_list.append(prob)

            # Sample binary decision
            decision = torch.bernoulli(prob)
            skip_decisions.append(int(decision.item()))

            # Log probability for REINFORCE
            lp = (
                decision * torch.log(prob + 1e-8) +
                (1 - decision) * torch.log(1 - prob + 1e-8)
            )
            log_probs.append(lp.squeeze())

        return skip_probs_list, skip_decisions, log_probs

    def resolve_skip_connections(self, skip_decisions, layer_outputs, image_input):
        """
        Resolve skip connection decisions to actual layer inputs.

        Rules from the paper:
        1. If a layer is not connected to any input -> use image as input
        2. At final layer -> concatenate all unconnected layer outputs
        3. If sizes differ -> pad smaller layers with zeros

        Args:
            skip_decisions: list of lists, skip_decisions[i][j] = 1 if layer j -> layer i
            layer_outputs: list of tensors, one per layer
            image_input: the original image tensor

        Returns:
            inputs_per_layer: list of resolved inputs for each layer
        """
        num_layers = len(layer_outputs)
        connected_as_output = [False] * num_layers
        inputs_per_layer = []

        for i in range(num_layers):
            if i == 0:
                inputs_per_layer.append(image_input)
                continue

            connected_inputs = []
            for j in range(i):
                if j < len(skip_decisions[i]) and skip_decisions[i][j] == 1:
                    connected_inputs.append(layer_outputs[j])
                    connected_as_output[j] = True

            if len(connected_inputs) == 0:
                # Rule 1: no input connected -> use image
                inputs_per_layer.append(image_input)
            else:
                # Pad and concatenate in depth dimension
                max_channels = max(t.shape[1] for t in connected_inputs)
                padded = []
                for t in connected_inputs:
                    if t.shape[1] < max_channels:
                        pad_size = max_channels - t.shape[1]
                        padding = torch.zeros(
                            t.shape[0], pad_size, t.shape[2], t.shape[3],
                            device=t.device
                        )
                        t = torch.cat([t, padding], dim=1)
                    padded.append(t)
                inputs_per_layer.append(torch.cat(padded, dim=1))

        return inputs_per_layer, connected_as_output