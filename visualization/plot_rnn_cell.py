# visualization/plot_rnn_cell.py

"""
Visualize discovered RNN cell architecture (Figure 8 from the paper).
Draws the computation graph of the recurrent cell as a tree.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
matplotlib.use("Agg")
import numpy as np
import os


def plot_rnn_cell(cell_architecture, output_path,
                   title="NAS Recurrent Cell Architecture"):
    """
    Visualize the recurrent cell computation graph found by NAS.

    Args:
        cell_architecture: dict with combination_methods, activation_functions,
                           cell_ct_index, cell_ct_new_index
        output_path: path to save figure
        title: plot title
    """
    combo_methods = cell_architecture["combination_methods"]
    act_functions = cell_architecture["activation_functions"]
    ct_idx = cell_architecture.get("cell_ct_index", 0)
    ct_new_idx = cell_architecture.get("cell_ct_new_index", 0)

    # Infer base_number from number of nodes
    num_nodes = len(combo_methods)
    base_number = (num_nodes + 1) // 2

    fig, ax = plt.subplots(figsize=(14, 8))

    node_positions = {}
    node_labels = {}

    # Leaf nodes (bottom row)
    leaf_y = 0
    for i in range(base_number):
        x = i * 2.5
        node_positions[i] = (x, leaf_y)

        ct_marker = " [+ct-1]" if i == ct_idx else ""
        ct_new_marker = " [ct]" if i == ct_new_idx else ""
        label = (
            f"Node {i}\n"
            f"{combo_methods[i]}\n"
            f"{act_functions[i]}"
            f"{ct_marker}{ct_new_marker}"
        )
        node_labels[i] = label

    # Internal nodes (upper rows)
    current_nodes = list(range(base_number))
    level = 1
    internal_idx = base_number

    while len(current_nodes) > 1:
        next_nodes = []
        level_y = level * 2.5

        for pair_idx in range(0, len(current_nodes) - 1, 2):
            left = current_nodes[pair_idx]
            right = current_nodes[pair_idx + 1] if pair_idx + 1 < len(current_nodes) else left

            x_left = node_positions[left][0]
            x_right = node_positions[right][0]
            x_mid = (x_left + x_right) / 2

            node_positions[internal_idx] = (x_mid, level_y)

            if internal_idx < len(combo_methods):
                label = (
                    f"Node {internal_idx}\n"
                    f"{combo_methods[internal_idx]}\n"
                    f"{act_functions[internal_idx]}"
                )
            else:
                label = f"Node {internal_idx}"

            node_labels[internal_idx] = label

            # Draw edges
            ax.annotate(
                "",
                xy=(x_mid, level_y - 0.3),
                xytext=(x_left, node_positions[left][1] + 0.3),
                arrowprops=dict(arrowstyle="->", color="#2C3E50", lw=1.2)
            )
            ax.annotate(
                "",
                xy=(x_mid, level_y - 0.3),
                xytext=(x_right, node_positions[right][1] + 0.3),
                arrowprops=dict(arrowstyle="->", color="#2C3E50", lw=1.2)
            )

            next_nodes.append(internal_idx)
            internal_idx += 1

        if len(current_nodes) % 2 == 1:
            next_nodes.append(current_nodes[-1])

        current_nodes = next_nodes
        level += 1

    # Draw nodes
    for node_id, (x, y) in node_positions.items():
        is_leaf = node_id < base_number
        color = "#AED6F1" if is_leaf else "#A9DFBF"
        edge_color = "#2E86C1" if is_leaf else "#27AE60"

        if node_id == ct_idx:
            color = "#F9E79F"
            edge_color = "#F39C12"
        if node_id == ct_new_idx:
            color = "#FADBD8"
            edge_color = "#E74C3C"

        circle = plt.Circle(
            (x, y), radius=0.6,
            facecolor=color, edgecolor=edge_color, linewidth=2, zorder=3
        )
        ax.add_patch(circle)
        ax.text(
            x, y, node_labels.get(node_id, str(node_id)),
            ha="center", va="center", fontsize=6.5, zorder=4,
            wrap=True
        )

    # Add input labels
    for i in range(base_number):
        x, y = node_positions[i]
        ax.text(x, y - 0.9, f"xt, h_{{t-1}}", ha="center",
                va="top", fontsize=7, color="#7F8C8D", style="italic")

    # Output label
    if current_nodes:
        root = current_nodes[0]
        rx, ry = node_positions[root]
        ax.text(rx, ry + 0.8, "h_t (output)", ha="center",
                va="bottom", fontsize=9, fontweight="bold", color="#C0392B")

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor="#AED6F1", label="Leaf node"),
        mpatches.Patch(facecolor="#A9DFBF", label="Internal node"),
        mpatches.Patch(facecolor="#F9E79F", label="c_{t-1} injection"),
        mpatches.Patch(facecolor="#FADBD8", label="c_t output"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

    all_x = [p[0] for p in node_positions.values()]
    all_y = [p[1] for p in node_positions.values()]
    ax.set_xlim(min(all_x) - 1.5, max(all_x) + 1.5)
    ax.set_ylim(min(all_y) - 1.5, max(all_y) + 2.0)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"RNN cell plot saved to {output_path}")