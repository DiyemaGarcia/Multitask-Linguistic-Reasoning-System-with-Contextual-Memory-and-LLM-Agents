# child_networks/cnn_builder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ChildCNN(nn.Module):
    """
    Child CNN network for CIFAR-10, built from controller-generated architecture.

    Supports:
    - Variable depth (num_layers)
    - Variable filter sizes, widths, strides, num_filters
    - Skip connections (concatenation in depth dimension)
    - Batch normalization
    - ReLU activations
    - Optional max pooling layers
    """

    def __init__(self, architecture, num_classes=10, use_skip_connections=True):
        super(ChildCNN, self).__init__()
        self.architecture = architecture
        self.num_classes = num_classes
        self.use_skip_connections = use_skip_connections

        self.layers = nn.ModuleList()
        self.batchnorms = nn.ModuleList()

        filter_heights   = architecture["filter_heights"]
        filter_widths    = architecture["filter_widths"]
        num_filters      = architecture["num_filters"]
        strides          = architecture["strides"]
        skip_connections = architecture.get("skip_connections", [])

        num_layers = len(filter_heights)

        self.layer_configs = []
        for i in range(num_layers):
            cfg = {
                "filter_height": filter_heights[i],
                "filter_width":  filter_widths[i],
                "num_filters":   num_filters[i],
                "stride":        strides[i] if i < len(strides) else 1,
                "skip_from":     skip_connections[i] if i < len(skip_connections) else []
            }
            self.layer_configs.append(cfg)

        self._build_layers()

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(self._out_channels, num_classes)

    def _resolve_skip_sources(self, i, channel_out):
        """
        Resolve valid skip sources for layer i.

        A skip source j is valid only if j < i (can only skip from earlier layers)
        AND j < len(channel_out) (the layer has been built).

        Returns:
            valid_js: list of valid source indices
            valid_channels: list of corresponding channel counts
        """
        cfg = self.layer_configs[i]
        skip_from = cfg["skip_from"]

        if not self.use_skip_connections or len(skip_from) == 0:
            return [], []

        valid_js = []
        valid_channels = []
        for j in skip_from:
            # j must reference a strictly earlier layer
            if j < i and j < len(channel_out):
                valid_js.append(j)
                valid_channels.append(channel_out[j])

        return valid_js, valid_channels

    def _compute_in_channels(self, i, channel_out):
        """
        Compute the number of input channels for layer i.

        - No valid skip sources: input = previous layer output (or image = 3)
        - Valid skip sources: input = concatenation => sum of their channels
        """
        valid_js, valid_channels = self._resolve_skip_sources(i, channel_out)

        if len(valid_channels) == 0:
            # No valid skip — sequential input
            return channel_out[i - 1] if i > 0 else 3

        return sum(valid_channels)

    def _build_layers(self):
        """
        Pre-computes input channels for each layer by simulating the forward
        pass, then builds Conv2d and BatchNorm2d modules accordingly.

        channel_out[i] = number of output channels of layer i (0-indexed).
        The image input has 3 channels (RGB).
        """
        channel_out = []

        for i, cfg in enumerate(self.layer_configs):
            in_ch  = self._compute_in_channels(i, channel_out)
            out_ch = cfg["num_filters"]
            fh     = cfg["filter_height"]
            fw     = cfg["filter_width"]
            s      = cfg["stride"]
            pad_h  = fh // 2
            pad_w  = fw // 2

            conv = nn.Conv2d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=(fh, fw),
                stride=(s, s),
                padding=(pad_h, pad_w),
                bias=False
            )
            bn = nn.BatchNorm2d(out_ch)

            self.layers.append(conv)
            self.batchnorms.append(bn)
            channel_out.append(out_ch)

        self._channel_out  = channel_out
        self._out_channels = channel_out[-1]

    def forward(self, x):
        layer_outputs = []   # layer_outputs[i] = output tensor of layer i
        image_input   = x
        current       = x

        for i, (conv, bn, cfg) in enumerate(
            zip(self.layers, self.batchnorms, self.layer_configs)
        ):
            # Resolve valid skip sources using the same logic as _build_layers
            valid_js, _ = self._resolve_skip_sources(i, self._channel_out[:i])

            if len(valid_js) > 0:
                # Gather tensors from valid skip sources
                inputs = [layer_outputs[j] for j in valid_js]

                if len(inputs) == 1:
                    current = inputs[0]
                else:
                    # Align spatial dimensions to the smallest
                    target_h = min(t.shape[2] for t in inputs)
                    target_w = min(t.shape[3] for t in inputs)
                    aligned = []
                    for t in inputs:
                        if t.shape[2] != target_h or t.shape[3] != target_w:
                            t = F.adaptive_avg_pool2d(t, (target_h, target_w))
                        aligned.append(t)

                    # Concatenate along channel dimension
                    current = torch.cat(aligned, dim=1)
            else:
                # No valid skip — use sequential input (previous layer or image)
                current = layer_outputs[i - 1] if i > 0 else image_input

            out = conv(current)
            out = bn(out)
            out = F.relu(out)
            layer_outputs.append(out)
            current = out

        # Global average pooling then classifier
        x = self.global_avg_pool(current)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def build_child_cnn(architecture, num_classes=10):
    """Factory function to build a child CNN from an architecture description."""
    return ChildCNN(architecture, num_classes=num_classes)