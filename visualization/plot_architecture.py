# visualization/plot_architecture.py

"""
Visualize discovered convolutional architectures (Figure 7 from the paper).
Draws layer-by-layer architecture with skip connections as arrows.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
matplotlib.use("Agg")
import numpy as np
import os


def plot_cnn_architecture(architecture, output_path, title="NAS CNN Architecture"):
    """
    Visualize a convolutional architecture found by NAS.

    Args:
        architecture: dict with filter_heights, filter_widths,
                       num_filters, strides, skip_connections
        output_path: path to save figure
        title: plot title
    """
    filter_heights = architecture["filter_heights"]
    filter_widths = architecture["filter_widths"]
    num_filters = architecture["num_filters"]
    strides = architecture.get("strides", [1] * len(filter_heights))
    skip_connections = architecture.get("skip_connections", [])

    num_layers = len(filter_heights)

    fig, ax = plt.subplots(figsize=(max(12, num_layers * 1.2), 6))

    # Draw layers as rectangles
    layer_positions = []
    box_width = 0.8
    box_height = 2.0

    for i in range(num_layers):
        x = i * 1.5
        y = 0
        height = box_height * (num_filters[i] / max(num_filters))

        rect = mpatches.FancyBboxPatch(
            (x - box_width / 2, y),
            box_width, height,
            boxstyle="round,pad=0.05",
            facecolor="#AED6F1",
            edgecolor="#2E86C1",
            linewidth=1.5
        )
        ax.add_patch(rect)
        layer_positions.append((x, y + height / 2))

        # Label
        ax.text(
            x, y - 0.3,
            f"L{i+1}\n{filter_heights[i]}x{filter_widths[i]}\n"
            f"F={num_filters[i]}\nS={strides[i]}",
            ha="center", va="top", fontsize=7
        )

    # Draw skip connections as arcs
    for i, skips in enumerate(skip_connections):
        for j, connected in enumerate(skips):
            if connected and j < i:
                x_start = j * 1.5
                x_end = i * 1.5
                y_mid = box_height + 0.3 + 0.2 * (i - j)
                ax.annotate(
                    "",
                    xy=(x_end, box_height * 0.5),
                    xytext=(x_start, box_height * 0.5),
                    arrowprops=dict(
                        arrowstyle="->",
                        color="#E74C3C",
                        lw=1.0,
                        connectionstyle=f"arc3,rad=-{0.2 + 0.05 * (i-j)}"
                    )
                )

    # Draw sequential connections
    for i in range(num_layers - 1):
        x_s = i * 1.5 + box_width / 2
        x_e = (i + 1) * 1.5 - box_width / 2
        ax.annotate(
            "",
            xy=(x_e, box_height * 0.5),
            xytext=(x_s, box_height * 0.5),
            arrowprops=dict(arrowstyle="->", color="#27AE60", lw=1.5)
        )

    ax.set_xlim(-1, num_layers * 1.5)
    ax.set_ylim(-1.0, box_height + 1.5)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axis("off")

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor="#AED6F1", edgecolor="#2E86C1", label="Conv Layer"),
        mpatches.Patch(facecolor="none", edgecolor="#E74C3C", label="Skip Connection"),
        mpatches.Patch(facecolor="none", edgecolor="#27AE60", label="Sequential"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"CNN architecture plot saved to {output_path}")