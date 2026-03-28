import sys
import os

# Adds the parent directory to sys.path
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torch.cuda import is_available
import Transformer_from_scratch
import torch
import torch.nn as nn
import numpy as np 
import altair as alt
import pandas as pd 
import warnings 
from config import get_config, get_weights_file_path
from train import get_model, get_ds, greedy_decode
import seaborn as sns
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

config = get_config()
train_dataloader, val_dataloader, vocab_src, vocab_tgt = get_ds(config)
model = get_model(config, vocab_src.get_vocab_size(), vocab_tgt.get_vocab_size()).to(device)

model_filename = get_weights_file_path(config, f"00")
state = torch.load(model_filename)
model.load_state_dict(state['model_state_dict'])


def load_next_batch():
    # Load a sample batch from the validation set
    batch = next(iter(val_dataloader))
    encoder_input = batch["encoder_input"].to(device)
    encoder_mask = batch["encoder_mask"].to(device)
    decoder_input = batch["decoder_input"].to(device)
    decoder_mask = batch["decoder_mask"].to(device)

    encoder_input_tokens = [vocab_src.id_to_token(idx) for idx in encoder_input[0].cpu().numpy()]
    decoder_input_tokens = [vocab_tgt.id_to_token(idx) for idx in decoder_input[0].cpu().numpy()]

    # check that the batch size is 1
    assert encoder_input.size(
        0) == 1, "Batch size must be 1 for validation"

    model_out = greedy_decode(
        model, encoder_input, encoder_mask, vocab_src, vocab_tgt, config['seq_len'], device)
    
    return batch, encoder_input_tokens, decoder_input_tokens


def mtx2df(m, max_row, max_col, row_tokens, col_tokens):
    return pd.DataFrame(
        [
            (
                r,
                c,
                float(m[r, c]),
                "%.3d %s" % (r, row_tokens[r] if len(row_tokens) > r else "<blank>"),
                "%.3d %s" % (c, col_tokens[c] if len(col_tokens) > c else "<blank>"),
            )
            for r in range(m.shape[0])
            for c in range(m.shape[1])
            if r < max_row and c < max_col
        ],
        columns=["row", "column", "value", "row_token", "col_token"],
    )

def retreiveKQV(KQV_src: str, layer: int, head: int):
    if KQV_src == "encoder":
        K = model.encoder.layers[layer].self_attention_block.K
        Q = model.encoder.layers[layer].self_attention_block.Q
        V = model.encoder.layers[layer].self_attention_block.V
    
    return [K[0, head].data, Q[0, head].data, V[0, head].data]



batch, encoder_tokens, decoder_tokens = load_next_batch()
encoder_input = batch["encoder_input"].to(device)
encoder_mask = batch["encoder_mask"].to(device)

model.encode(encoder_input, encoder_mask)
K, Q, V = retrieveKQV("encoder", 1, 1)


# Create 3 heatmaps side by side
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot each matrix
sns.heatmap(K, ax=axes[0], cmap='viridis')
axes[0].set_title('Key (K)')
axes[0].set_xlabel('d_k')
axes[0].set_ylabel('Token')

sns.heatmap(Q, ax=axes[1], cmap='viridis')
axes[1].set_title('Query (Q)')
axes[1].set_xlabel('d_k')

sns.heatmap(V, ax=axes[2], cmap='viridis')
axes[2].set_title('Value (V)')
axes[2].set_xlabel('d_k')

plt.tight_layout()
plt.show()

