# GPT-2 small model configuration
GPT_CONFIG_124M = {
    "vocab_size": 50257,     # Vocabulary size in tokenizer
    "context_length": 1024,  # Context length, i.e. max num of input tokens the model can handle via positional embedding
    "emb_dim": 768,          # Embedding dimension
    "n_heads": 12,           # Number of attention heads in multi-head attention mechanism
    "n_layers": 12,          # Number of layers, AKA transformer blocks
    "drop_rate": 0.1,        # Dropout rate
    "qkv_bias": False        # Query-Key-Value bias
}

#GPT-2 medium model configuration
GPT_CONFIG_355M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 1024,
    "n_heads": 16,
    "n_layers": 24,
    "drop_rate": 0.1,
    "qkv_bias": False
}

#GPT-2 large model configuration
GPT_CONFIG_774M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 1280,
    "n_heads": 20,
    "n_layers": 36,
    "drop_rate": 0.1,
    "qkv_bias": False
}

#GPT-2 XL model configuration
GPT_CONFIG_1558M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 1600,
    "n_heads": 25,
    "n_layers": 48,
    "drop_rate": 0.1,
    "qkv_bias": False
}
