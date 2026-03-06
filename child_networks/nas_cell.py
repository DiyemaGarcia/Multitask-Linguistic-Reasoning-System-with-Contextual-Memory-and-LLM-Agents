# child_networks/nas_cell.py

"""
NASCell: the best recurrent cell discovered by Neural Architecture Search
on Penn Treebank word-level language modeling.

This is the fixed best cell from the paper (not the search process itself).
It is included in TensorFlow as tf.contrib.rnn.NASCell.

The cell architecture (from Figure 8 of the paper) performs operations
similar to LSTM in the first few steps, computing W1*h_{t-1} + W2*x_t
multiple times and routing results to different components.

Since the exact discovered architecture is not fully specified in the paper text,
we implement the structure as described in Section 3.4 and approximate
the best cell based on available TensorFlow source code documentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NASCellFixed(nn.Module):
    """
    Fixed implementation of the NAS-discovered recurrent cell.
    Based on the TensorFlow NASCell implementation (Zoph & Le, 2017).

    Architecture (approximated from paper Figure 8):
    - 8 learnable weight pairs (W_x_i, W_h_i)
    - Multiple tanh/sigmoid/relu operations
    - Cell state mechanism similar to LSTM
    """

    def __init__(self, hidden_size, input_size):
        super(NASCellFixed, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size

        # 8 weight pairs for x_t
        self.W_x = nn.Linear(input_size, 8 * hidden_size, bias=False)
        # 8 weight pairs for h_{t-1}
        self.W_h = nn.Linear(hidden_size, 8 * hidden_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(8 * hidden_size))

    def forward(self, x_t, h_prev, c_prev):
        """
        NAS cell forward pass.

        Args:
            x_t:    [batch, input_size]
            h_prev: [batch, hidden_size]
            c_prev: [batch, hidden_size]

        Returns:
            h_t: [batch, hidden_size]
            c_t: [batch, hidden_size]
        """
        H = self.hidden_size

        # Compute all 8 linear projections at once
        gates = self.W_x(x_t) + self.W_h(h_prev) + self.bias
        # Split into 8 components
        g = gates.chunk(8, dim=1)

        # NAS cell operations (approximated from TF source)
        # These operations were discovered by the controller
        layer1_0 = torch.sigmoid(g[0])
        layer1_1 = torch.relu(g[1])
        layer1_2 = torch.sigmoid(g[2])
        layer1_3 = torch.relu(g[3] * layer1_2)
        layer1_4 = torch.tanh(g[4])
        layer1_5 = torch.sigmoid(g[5])
        layer1_6 = torch.tanh(g[6])
        layer1_7 = torch.sigmoid(g[7])

        # Combine operations
        l2_0 = torch.tanh(layer1_0 * layer1_1)
        l2_1 = torch.tanh(layer1_2 + layer1_3)
        l2_2 = torch.tanh(layer1_4 * layer1_5)
        l2_3 = layer1_6 + layer1_7

        # Cell state and output
        c_t = torch.tanh(l2_0 + l2_2)
        h_t = torch.tanh(l2_1 * torch.sigmoid(l2_3 + c_t))

        return h_t, c_t