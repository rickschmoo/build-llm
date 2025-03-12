# dataloader_test
from dataset import GPTDatasetV1
import torch

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# dataloader = GPTDatasetV1.create_dataloader_v1(
#     raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)

# data_iter = iter(dataloader)

# first_batch = next(data_iter)
# print(first_batch)

# second_batch = next(data_iter)
# print(second_batch)





# dataloader2 = GPTDatasetV1.create_dataloader_v1(
#     raw_text, batch_size=1, max_length=8, stride=2, shuffle=False)

# data_iter2 = iter(dataloader2)

# first_batch2 = next(data_iter2)
# print(first_batch2)

# second_batch2 = next(data_iter2)
# print(second_batch2)

print ('-'*50)
print ('Data loader test')
print ('-'*50)

dataloader3 = GPTDatasetV1.create_dataloader_v1(
    raw_text, batch_size=8, max_length=4, stride=4,
    shuffle=False
)

data_iter3 = iter(dataloader3)
inputs, targets = next(data_iter3)
print("Inputs:\n", inputs)
print("\nTargets:\n", targets)



# weight matrix for the embedding layer
print ('-'*50)
print ('Weight matrix for the embedding layer')
print ('-'*50)
input_ids = torch.tensor([2, 3, 5, 1])
vocab_size = 6
output_dim = 3
torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
print(embedding_layer.weight)

# get embedding vector for a token ID by applying weight
# i.e. convert a single token ID into a three-dimensional embedding vector
print ('-'*50)
print ('Get embedding vector for token IDs by applying weight')
print ('-'*50)
print(embedding_layer(input_ids))


# more realistic example
# embed each token in each batch into a 256-dimensional vector
print ('-'*50)
print ('A bit more realistic: vocab 50257, output dims 256')
print ('-'*50)
vocab_size = 50257
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

# more realistic data loader
# the token ID tensor is 8 × 4 dimensional, meaning that the data batch consists of eight text samples with four tokens each
max_length = 4
dataloader = GPTDatasetV1.create_dataloader_v1(
    raw_text, batch_size=8, max_length=max_length,
   stride=max_length, shuffle=False
)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Token IDs:\n", inputs)
print("\nInputs shape:\n", inputs.shape)

# use the embedding layer to embed these token IDs into 256-dimensional vectors:
token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)

# now absolute embedding
print ('-'*50)
print ('With absolute position embeddings')
print ('-'*50)
context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print(pos_embeddings.shape)
input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)
