# child_networks/rnn_cell_builder.py

import torch
import torch.nn as nn
import numpy as np


def apply_activation(tensor, activation_name):
    """Apply named activation function to tensor."""
    if activation_name == "tanh":
        return torch.tanh(tensor)
    elif activation_name == "sigmoid":
        return torch.sigmoid(tensor)
    elif activation_name == "relu":
        return torch.relu(tensor)
    elif activation_name == "identity":
        return tensor
    elif activation_name == "sin":
        return torch.sin(tensor)
    else:
        raise ValueError(f"Unknown activation: {activation_name}")


def apply_combination(a, b, method):
    """Apply combination method to two tensors."""
    if method == "add":
        return a + b
    elif method == "elem_mult":
        return a * b
    elif method == "max":
        return torch.max(a, b)
    else:
        raise ValueError(f"Unknown combination method: {method}")


class NASCell(nn.Module):
    """
    Recurrent cell built from controller-generated architecture (Section 3.4).

    The cell takes x_t and h_{t-1} as inputs and produces h_t.
    The computation is defined by a binary tree with:
        - base_number leaf nodes
        - base_number - 1 internal nodes
        - total: 2*base_number - 1 nodes

    Each node has:
        - combination_method (add, elem_mult)
        - activation_function (identity, tanh, sigmoid, relu)

    Cell states c_{t-1} and c_t are injected at controller-specified positions.
    """

    def __init__(self, cell_architecture, input_size, hidden_size):
        """
        Args:
            cell_architecture: dict with:
                - combination_methods: list of methods per node
                - activation_functions: list of activations per node
                - cell_ct_index: node index where c_{t-1} is injected
                - cell_ct_new_index: node index where c_t is read
            input_size: dimension of x_t
            hidden_size: dimension of h_t
        """
        super(NASCell, self).__init__()
        self.architecture = cell_architecture
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.base_number = len(cell_architecture["combination_methods"]) // 2 + 1

        # Learnable weight matrices (one pair per leaf node)
        # Each leaf computes: W1 * x_t + W2 * h_{t-1}
        num_leaf_nodes = self.base_number
        self.W_x = nn.ParameterList([
            nn.Parameter(torch.Tensor(hidden_size, input_size))
            for _ in range(num_leaf_nodes)
        ])
        self.W_h = nn.ParameterList([
            nn.Parameter(torch.Tensor(hidden_size, hidden_size))
            for _ in range(num_leaf_nodes)
        ])
        self.biases = nn.ParameterList([
            nn.Parameter(torch.zeros(hidden_size))
            for _ in range(num_leaf_nodes)
        ])

        self._reset_parameters()

    def _reset_parameters(self):
        for i in range(self.base_number):
            nn.init.orthogonal_(self.W_x[i])
            nn.init.orthogonal_(self.W_h[i])

    def forward(self, x_t, h_prev, c_prev):
        """
        Compute h_t given x_t, h_{t-1}, c_{t-1}.

        Args:
            x_t:    [batch, input_size]
            h_prev: [batch, hidden_size]
            c_prev: [batch, hidden_size]

        Returns:
            h_t:    [batch, hidden_size]
            c_t:    [batch, hidden_size]
        """
        arch = self.architecture
        combo_methods = arch["combination_methods"]
        act_functions = arch["activation_functions"]
        ct_inject_idx = arch["cell_ct_index"]
        ct_new_idx = arch["cell_ct_new_index"]

        # Leaf node computations (indices 0 to base_number-1)
        leaf_outputs = []
        leaf_pre_act = []  # before activation (for ct)
        for i in range(self.base_number):
            pre_act = (
                torch.matmul(x_t, self.W_x[i].t()) +
                torch.matmul(h_prev, self.W_h[i].t()) +
                self.biases[i]
            )
            leaf_pre_act.append(pre_act)
            out = apply_activation(pre_act, act_functions[i])
            leaf_outputs.append(out)

        # Inject c_{t-1} at specified leaf node
        if ct_inject_idx < self.base_number:
            leaf_outputs[ct_inject_idx] = apply_activation(
                leaf_outputs[ct_inject_idx] + c_prev,
                act_functions[ct_inject_idx]
            )

        # Internal node computations (indices base_number to 2*base_number-2)
        all_outputs = leaf_outputs.copy()
        num_internal = self.base_number - 1
        node_idx_offset = self.base_number

        # Process tree bottom-up
        # Each internal node combines two children
        pair_outputs = list(leaf_outputs)
        internal_outputs = []

        for k in range(num_internal):
            node_global_idx = node_idx_offset + k
            combo = combo_methods[node_global_idx] if node_global_idx < len(combo_methods) else "add"
            act = act_functions[node_global_idx] if node_global_idx < len(act_functions) else "tanh"

            # Combine pairs left-to-right
            if len(pair_outputs) >= 2:
                a = pair_outputs.pop(0)
                b = pair_outputs.pop(0)
                combined = apply_combination(a, b, combo)
                activated = apply_activation(combined, act)
                pair_outputs.append(activated)
                internal_outputs.append(activated)

        # h_t is the output of the root node (last internal node)
        h_t = pair_outputs[0] if pair_outputs else leaf_outputs[-1]

        # c_t is read from specified leaf node (pre-activation)
        if ct_new_idx < self.base_number:
            c_t = leaf_pre_act[ct_new_idx]
        else:
            c_t = h_t.clone()

        return h_t, c_t


class NASRNNModel(nn.Module):
    """
    Full RNN language model using NASCell for Penn Treebank.
    Two-layer architecture as described in the paper.
    """

    def __init__(self, cell_architecture, vocab_size, embed_size, hidden_size,
                 num_layers=2, dropout_rate=0.65, embedding_dropout_rate=0.1):
        super(NASRNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.embedding_dropout = nn.Dropout(embedding_dropout_rate)

        # Two NASCell layers
        self.cells = nn.ModuleList()
        input_sz = embed_size
        for layer in range(num_layers):
            cell = NASCell(cell_architecture, input_sz, hidden_size)
            self.cells.append(cell)
            input_sz = hidden_size

        self.dropout = nn.Dropout(dropout_rate)
        self.output_projection = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_seq, hidden_states):
        """
        Args:
            input_seq: [seq_len, batch]
            hidden_states: list of (h, c) per layer

        Returns:
            logits: [seq_len * batch, vocab_size]
            new_hidden_states: updated list of (h, c)
        """
        seq_len, batch = input_seq.shape
        embeds = self.embedding(input_seq)              # [seq_len, batch, embed]
        embeds = self.embedding_dropout(embeds)

        outputs = []
        new_hidden_states = []

        for layer_idx, cell in enumerate(self.cells):
            h, c = hidden_states[layer_idx]
            layer_outputs = []
            for t in range(seq_len):
                x_t = embeds[t] if layer_idx == 0 else layer_in[t]
                h, c = cell(x_t, h, c)
                layer_outputs.append(h.unsqueeze(0))
            layer_in = torch.cat(layer_outputs, dim=0)
            layer_in = self.dropout(layer_in)
            new_hidden_states.append((h.detach(), c.detach()))

        logits = self.output_projection(layer_in.view(-1, self.hidden_size))
        return logits, new_hidden_states

    def init_hidden(self, batch_size):
        device = next(self.parameters()).device
        return [
            (torch.zeros(batch_size, self.hidden_size, device=device),
             torch.zeros(batch_size, self.hidden_size, device=device))
            for _ in range(self.num_layers)
        ]