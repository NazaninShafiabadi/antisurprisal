"""
This script visualizes the attention and hidden states of a transformer model over time.

Usage:
python analysis/attn_hs_visualization.py \
--attn_hs_dir <directory> \
--token <token> \
--batch <batch_index> \
--visualize <attn|hidden> \
--save_path <path>

Example:
python analysis/attn_hs_visualization.py \
--attn_hs_dir results/attn_hs \
--token "assistance" \
--tokenizer "google/multiberts-seed_0" \
--batch 0 \
--visualize attn \
--animate \
--save_path img/assistance_attn_anim.gif
"""

import argparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
import torch
from tqdm import tqdm
import os
import glob
from transformers import AutoTokenizer


anim = None

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attn_hs_dir', type=str, required=True, help='Directory containing attention and hidden states')
    parser.add_argument('--token', type=str, required=True, help='Token to visualize')
    parser.add_argument('--tokenizer', type=str, required=True, help='Tokenizer to convert token IDs to token names')
    parser.add_argument('--batch_idx', type=int, default=0, help='Batch index (sentence) to visualize')
    # Flag to select whether to visualize the attention or the hidden states
    parser.add_argument('--visualize', type=str, choices=['attn', 'hidden'], default='attn', help='Type of visualization')
    parser.add_argument('--save_path', type=str, default='animation.gif', help='Path to save the animation')
    parser.add_argument('--animate', action='store_true', help='Create an animation')
    return parser


def load_data(attn_hs_dir, token, tokenizer, batch_idx):
    # Load files for selected token and sort by step
    # Assuming the file names follow the pattern: token_{token}_step_{step}.bin
    files = glob.glob(os.path.join(attn_hs_dir, f"token_{token.lower()}_step_*.bin"))
    if not files:
        print(f"No files found for token '{token}' in directory {attn_hs_dir}")
        return []
    sorted_files = sorted(files, key=lambda x: int(x.split("_step_")[-1].split(".bin")[0]))

    data = []
    for f in tqdm(sorted_files, desc="Loading files"):
        obj = torch.load(f, weights_only=True)
        # Densify the sparse attention tensor and dequantize
        attn_dense = obj['Attentions'][0].to_dense().float() / 10000    # [batch_size, num_heads, seq_len]
        # Get the key tokens for the attention head
        key_tokens = tokenizer.convert_ids_to_tokens(obj['TokenIDs'][batch_idx])
        seq_len = len(key_tokens)
        data.append({
            'step': obj['Step'],
            'token': obj['Token'],
            'key_tokens': key_tokens,
            'attn': attn_dense[args.batch_idx][:, :seq_len],  # [heads, seq_len]
            'hs': obj['HiddenStates'][0][args.batch_idx],     # [hidden_dim]
        })
    return data

def animate_data(data, args):
    """
    Create an animation of the attention or hidden states over time.
    """
    global anim

    # Set the figure size based on the number of tokens
    token_count = max(len(d['key_tokens']) for d in data)
    width = max(token_count, 10) # if args.visualize == 'attn' else 10  # 0.6 inch per token, min 6 inches
    height = max(2 * data[0]['attn'].shape[0], 8) if args.visualize == 'attn' else 8

    if args.visualize == 'attn':
        num_heads = data[0]['attn'].shape[0]
        fig = plt.figure(figsize=(width, height))
        gs = gridspec.GridSpec(num_heads, 2, width_ratios=[20, 1], wspace=0.05)  # 2nd column for colorbar

        axes = [fig.add_subplot(gs[i, 0]) for i in range(num_heads)]
        cbar_ax = fig.add_subplot(gs[:, 1])
        if num_heads == 1:
            axes = [axes]   # Ensure axes is a list for consistency
        
        heatmaps = []
        for head_idx, ax_i in enumerate(axes):
            # Initial attention matrix (1st step)
            attn_matrix = data[0]['attn'][head_idx].numpy()[np.newaxis, :]  # [1, seq_len]
            heatmap = sns.heatmap(
                attn_matrix,
                cmap='coolwarm',
                cbar=(head_idx == num_heads - 1),  # Only show one colorbar
                cbar_ax=cbar_ax if head_idx == num_heads - 1 else None,
                ax=ax_i,
                vmin=0, vmax=1, # for consistent color scale across frames
                xticklabels=data[0]['key_tokens'] if head_idx == num_heads - 1 else False,
                yticklabels=False,
                linewidths=0.5, linecolor='white'
            )
            ax_i.set_ylabel(f"Head {head_idx}", rotation=0, labelpad=50, va='center', fontsize=20)
            heatmaps.append(heatmap)
        
        axes[-1].set_xticks(np.arange(len(data[0]['key_tokens'])) + 0.8)
        axes[-1].set_xticklabels(data[0]['key_tokens'], rotation=45, ha='right', fontsize=20)
        
    else:   # Hidden state line plot
        fig, ax = plt.subplots(figsize=(width, height))
        # Initial hidden state (1st step)
        line, = ax.plot(data[0]['hs'].numpy())
        ax.set_title(f"Hidden States at Step {data[0]['step']} | Token {args.token}")
        ax.set_ylabel("Activation")
        ax.set_xlabel("Hidden Dimension")

    def update(i):
        if args.visualize == 'attn':
            for head_idx, heatmap in enumerate(heatmaps):
                # Update the attention matrix for the current step
                attn_matrix = data[i]['attn'][head_idx].numpy()#[np.newaxis, :]
                heatmap.collections[0].set_array(attn_matrix.ravel())
            fig.suptitle(
                f"Attention at Step {data[i]['step']} | Token {args.token}",
                fontsize=30, y=0.9
            )
        else:
            line.set_ydata(data[i]['hs'].numpy())
            ax.set_title(f"Hidden States at Step {data[i]['step']} | Token {args.token}")

        return fig.axes
    
    anim = animation.FuncAnimation(fig, update, frames=len(data), interval=800, repeat=False)
    anim.save(args.save_path, writer=PillowWriter(fps=2), dpi=100)
    print(f"Animation saved to {args.save_path}")


def main(args):
    # Set the style for seaborn
    sns.set_theme(style="whitegrid")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # Load data
    data = load_data(args.attn_hs_dir, args.token, tokenizer, args.batch_idx)

    if args.animate:
        animate_data(data, args)


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)