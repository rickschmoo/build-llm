import torch
import torch.nn as nn
from attention2_trainable_weights_compact import SelfAttention_v2

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
torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)

# compute the attention weights using the softmax function
queries = sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs) 

attn_scores = queries @ keys.T
context_length = attn_scores.shape[0]

# APPROACH 1
# attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
# print(attn_weights)

# # use PyTorch’s tril function to create a mask where 
# # the values above the diagonal are zero
# context_length = attn_scores.shape[0]
# mask_simple = torch.tril(torch.ones(context_length, context_length))
# print(mask_simple)

# # multiply this mask with the attention weights to zero-out
# # the values above the diagonal
# masked_simple = attn_weights*mask_simple
# print(masked_simple)

# # normalize the attention weights
# row_sums = masked_simple.sum(dim=-1, keepdim=True)
# masked_simple_norm = masked_simple / row_sums
# print(masked_simple_norm)

# APPROACH 2
# more efficient mask
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
print("Mask: ", masked)

attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=1)
print("Attention weights: ", attn_weights)

# create dropout mask to zero out some of the attention weights at random
torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5)
example = torch.ones(6, 6)
print("Dropout mask: ", dropout(example))

# apply dropout to the attention weight matrix
torch.manual_seed(123)
print("Attention weights, post drpout: ", dropout(attn_weights))
