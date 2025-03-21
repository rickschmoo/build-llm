# gpt_download_test.py
import torch
import torch.nn as nn
import tiktoken

from gpt_config import GPT_CONFIG_124M, GPT_CONFIG_355M, GPT_CONFIG_774M, GPT_CONFIG_1558M
from transformer_block import TransformerBlock
from layer_norm import LayerNorm
from utility_functions import generate_text_simple, text_to_token_ids, token_ids_to_text
from utility_functions import calc_loss_batch, calc_loss_loader 
from dataset import GPTDatasetV1
from gpt_download import download_and_load_gpt2

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}



settings, params = download_and_load_gpt2(
    model_size="124M", models_dir="gpt2"
)
print("Settings:", settings)
print("Parameter dictionary keys:", params.keys())

print(params["wte"])
print("Token embedding weight tensor dimensions:", params["wte"].shape)
