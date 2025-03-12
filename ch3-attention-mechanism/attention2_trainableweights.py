import torch
inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

x_2 = inputs[1]
d_in = inputs.shape[1]
d_out = 2

# initialize weight matrices for query, key and value weight vectors
torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key   = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

# compute the query, key, and value vectors for noddy example
query_2 = x_2 @ W_query 
key_2 = x_2 @ W_key 
value_2 = x_2 @ W_value
print(query_2)

# calculate for all the inputs 
keys = inputs @ W_key 
values = inputs @ W_value
print("keys.shape:", keys.shape)
print("values.shape:", values.shape)

# compute attention scores
# attention score computation is a dot-product computation similar 
# to what we used in the simplified self-attention mechanism. 
# The new aspect here is that we are not directly computing 
# the dot-product between the input elements 
# but using the query and key obtained by transforming t
keys_2 = keys[1]
attn_score_22 = query_2.dot(keys_2)
print(attn_score_22)

attn_scores_2 = query_2 @ keys.T
print("Attention scores for other inputs on this input: ", attn_scores_2)

# normalize scores using softmax to get attention weights
# scale attention scores by dividing them by the square root of the 
# embedding dimension of the keys (taking the square root is 
# mathematically the same as exponentiating by 0.5)
d_k = keys.shape[-1]
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
print("Normalized attention weights: ", attn_weights_2)

# compute the context vector for an INPUT as a weighted sum over the 
# value vectors for the other inputs
context_vec_2 = attn_weights_2 @ values
print("Context vector example for 2nd input only: ", context_vec_2)
