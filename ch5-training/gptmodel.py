# gptmodel.py
import torch
import torch.nn as nn
import tiktoken
import numpy as np

from gpt_config import GPT_CONFIG_124M, GPT_CONFIG_355M, GPT_CONFIG_774M, GPT_CONFIG_1558M
from transformer_block import TransformerBlock
from layer_norm import LayerNorm
from utility_functions import generate_text_simple, text_to_token_ids, token_ids_to_text
from utility_functions import calc_loss_batch, calc_loss_loader 
from dataset import GPTDatasetV1

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # create stack of Transformer blocks
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)

        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

def train_model_simple(model, train_loader, val_loader,
                       optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, "
                      f"Val loss {val_loss:.3f}"
                )

        generate_and_print_sample(
            model, tokenizer, device, start_context
        )
    return train_losses, val_losses, track_tokens_seen

# calc training and validation set losses after each model update so we can
# evaluate whether the training improves the model.
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter
        )
    model.train()
    return train_loss, val_loss

# convenience function that we use to track whether the model 
# improves during the training by visual inspection
def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size, 
            top_k=25, temperature=1.4
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()

# modified text generation function with more diversity
# incorporates temperature sampling and top-k sampling
def generate(model, idx, max_new_tokens, context_size,
             temperature=0.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float('-inf')).to(logits.device),
                logits
            )
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        if idx_next == eos_id:
            break
        idx = torch.cat((idx, idx_next), dim=1)
    return idx




####################################
# MAIN()
####################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = tiktoken.get_encoding("gpt2")

# DO_LOAD_TEST = False
# if (DO_LOAD_TEST):
#     checkpoint = torch.load("model_and_optimizer.pth", map_location=device)
#     model = GPTModel(GPT_CONFIG_124M)
#     model.load_state_dict(checkpoint["model_state_dict"])
#     optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
#     optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
#     print("Loaded model and optimizer state from checkpoint.")
#     model.train();
#     exit()

############
# GPT2 code
############
USE_GPT = True
if (USE_GPT):
    print("Loading pre-trained GPT-2 model...")
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

    model_name = "gpt2-small (124M)"
    NEW_CONFIG = GPT_CONFIG_124M.copy()
    NEW_CONFIG.update(model_configs[model_name])
    NEW_CONFIG.update({"context_length": 1024})
    NEW_CONFIG.update({"qkv_bias": True})
    print("Updated config:", NEW_CONFIG)

    gpt = GPTModel(NEW_CONFIG)
    gpt.eval()

    def assign(left, right):
        if left.shape != right.shape:
            raise ValueError(f"Shape mismatch. Left: {left.shape}, "
                            "Right: {right.shape}"
            )
        return torch.nn.Parameter(torch.tensor(right))


    def load_weights_into_gpt(gpt, params):
        gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
        gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])


        for b in range(len(params["blocks"])):
            q_w, k_w, v_w = np.split(
                (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
            gpt.trf_blocks[b].att.W_query.weight = assign(
                gpt.trf_blocks[b].att.W_query.weight, q_w.T)
            gpt.trf_blocks[b].att.W_key.weight = assign(
                gpt.trf_blocks[b].att.W_key.weight, k_w.T)
            gpt.trf_blocks[b].att.W_value.weight = assign(
                gpt.trf_blocks[b].att.W_value.weight, v_w.T)

            q_b, k_b, v_b = np.split(
                (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
            gpt.trf_blocks[b].att.W_query.bias = assign(
                gpt.trf_blocks[b].att.W_query.bias, q_b)
            gpt.trf_blocks[b].att.W_key.bias = assign(
                gpt.trf_blocks[b].att.W_key.bias, k_b)
            gpt.trf_blocks[b].att.W_value.bias = assign(
                gpt.trf_blocks[b].att.W_value.bias, v_b)

            gpt.trf_blocks[b].att.out_proj.weight = assign(
                gpt.trf_blocks[b].att.out_proj.weight, 
                params["blocks"][b]["attn"]["c_proj"]["w"].T)
            gpt.trf_blocks[b].att.out_proj.bias = assign(
                gpt.trf_blocks[b].att.out_proj.bias, 
                params["blocks"][b]["attn"]["c_proj"]["b"])

            gpt.trf_blocks[b].ff.layers[0].weight = assign(
                gpt.trf_blocks[b].ff.layers[0].weight, 
                params["blocks"][b]["mlp"]["c_fc"]["w"].T)
            gpt.trf_blocks[b].ff.layers[0].bias = assign(
                gpt.trf_blocks[b].ff.layers[0].bias, 
                params["blocks"][b]["mlp"]["c_fc"]["b"])
            gpt.trf_blocks[b].ff.layers[2].weight = assign(
                gpt.trf_blocks[b].ff.layers[2].weight, 
                params["blocks"][b]["mlp"]["c_proj"]["w"].T)
            gpt.trf_blocks[b].ff.layers[2].bias = assign(
                gpt.trf_blocks[b].ff.layers[2].bias, 
                params["blocks"][b]["mlp"]["c_proj"]["b"])

            gpt.trf_blocks[b].norm1.scale = assign(
                gpt.trf_blocks[b].norm1.scale, 
                params["blocks"][b]["ln_1"]["g"])
            gpt.trf_blocks[b].norm1.shift = assign(
                gpt.trf_blocks[b].norm1.shift, 
                params["blocks"][b]["ln_1"]["b"])
            gpt.trf_blocks[b].norm2.scale = assign(
                gpt.trf_blocks[b].norm2.scale, 
                params["blocks"][b]["ln_2"]["g"])
            gpt.trf_blocks[b].norm2.shift = assign(
                gpt.trf_blocks[b].norm2.shift, 
                params["blocks"][b]["ln_2"]["b"])

        gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
        gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
        gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])

        print ("DEBUG:")
        print(gpt.tok_emb.weight is None)
        print(gpt.tok_emb.weight.shape)
        print("Token embedding weight tensor dimensions:", gpt.tok_emb.weight.shape)
        print("Positional embedding weight tensor dimensions:", gpt.pos_emb.weight.shape)

    load_weights_into_gpt(gpt, params)
    gpt.to(device)

    torch.manual_seed(123)
    # DEBUG: Check if weights exist
    print("DEBUG: ")
    print(gpt.tok_emb.weight is None)  # Should print False
    print(gpt.tok_emb.weight.shape)    # Should print expected dimensions
    token_ids = generate(
        model=gpt,
        idx=text_to_token_ids("Every effort moves you", tokenizer).to(device),
        max_new_tokens=25,
        context_size=NEW_CONFIG["context_length"],
        top_k=50,
        temperature=1.5
    )
    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
    token_ids = generate(
        model=gpt,
        idx=text_to_token_ids("The rocket launched", tokenizer).to(device),
        max_new_tokens=25,
        context_size=NEW_CONFIG["context_length"],
        top_k=50,
        temperature=1.5
    )
    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
    exit()







####################################
# Chapter 5 pre-GPT2 code
####################################
# start_context = "Every effort moves you"


# load the text data
file_path = "the-verdict.txt"
with open(file_path, "r", encoding="utf-8") as file:
    text_data = file.read()
total_characters = len(text_data)
total_tokens = len(tokenizer.encode(text_data))
print("Characters:", total_characters)
print("Tokens:", total_tokens)

# define a train_ratio to use 90% of the data for training 
# and the remaining 10% as validation data for model evaluation during training
train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

torch.manual_seed(123)

train_loader = GPTDatasetV1.create_dataloader_v1(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0
)

val_loader = GPTDatasetV1.create_dataloader_v1(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0
)

print("Training loader:")
for x, y in train_loader:
    print(x.shape, y.shape)

print("\nValidation loader:")
for x, y in val_loader:
    print(x.shape, y.shape)

# with torch.no_grad():
#     train_loss = calc_loss_loader(train_loader, model, device)
#     val_loss = calc_loss_loader(val_loader, model, device)
# print("Training loss:", train_loss)
# print("Validation loss:", val_loss)

###########
# TRAINING
###########
model = GPTModel(GPT_CONFIG_124M)
model.to(device)
optimizer = torch.optim.AdamW(
     model.parameters(),
    lr=0.0004, weight_decay=0.1
)
num_epochs = 10
train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context="Every effort moves you", tokenizer=tokenizer
)

# code to plot losses over course of training
# epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
# plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

##########################
# GENERATE SOME TEST TEXT
##########################
model.to("cpu")
model.eval()
tokenizer = tiktoken.get_encoding("gpt2")
token_ids = generate(
    model=model,
    idx=text_to_token_ids("Every effort moves you", tokenizer),
    max_new_tokens=25,
    context_size=GPT_CONFIG_124M["context_length"],
    top_k=25, temperature=1.4
)
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

# save model
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    }, 
    "model_and_optimizer.pth"
)


