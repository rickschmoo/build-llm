# attention4-multihead-compact.py
import torch
import torch.nn as nn
from attention4_multihead_compact import MultiHeadAttention

# try out with simple example
inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

d_in = inputs.shape[1]
d_out = 2

batch = torch.stack((inputs, inputs), dim=0)
print("Input batch shape: ", batch.shape)

# MultiHeadAttentionWrapper class test
# torch.manual_seed(123)
# context_length = batch.shape[1] # This is the number of tokens
# d_in, d_out = 3, 2
# mha = MultiHeadAttentionWrapper(
#     d_in, d_out, context_length, 0.0, num_heads=2
# )
# context_vecs = mha(batch)

# print(context_vecs)
# print("context_vecs.shape:", context_vecs.shape)

# MultiHeadAttention class test
torch.manual_seed(123)
batch_size, context_length, d_in = batch.shape
d_out = 2
mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
context_vecs = mha(batch)
print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)
