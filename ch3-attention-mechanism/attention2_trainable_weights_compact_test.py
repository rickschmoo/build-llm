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
print("Context vector from SA model: ", sa_v2(inputs))


