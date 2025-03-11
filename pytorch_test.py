# Build an LLM book
# # https://github.com/rasbt/stat453-deep-learning-ss21/blob/main/L08/code/cross-entropy-pytorch.ipynb
# 
# import torch
import torch.nn.functional as F

# TEST CODE
print(torch.__version__)
print("CUDA compatibility: ", torch.cuda.is_available())
print("Mac Silicon MPS: ", torch.backends.mps.is_available())

def to_onehot(y, num_classes):
    y_onehot = torch.zeros(y.size(0), num_classes)
    y_onehot.scatter_(1, y.view(-1, 1).long(), 1).float()
    return y_onehot

y = torch.tensor([0, 1, 2, 2])

y_enc = to_onehot(y, 3)

print('one-hot encoding:\n', y_enc)

Z = torch.tensor( [[-0.3,  -0.5, -0.5],
                   [-0.4,  -0.1, -0.5],
                   [-0.3,  -0.94, -0.5],
                   [-0.99, -0.88, -0.5]])

print(Z)

def softmax(z):
    return (torch.exp(z.t()) / torch.sum(torch.exp(z), dim=1)).t()

# Next, we convert them to "probabilities" via softmax:
smax = softmax(Z)
print('softmax:\n', smax)

# The probabilties can then be converted back to class labels based on the largest probability in each row:
def to_classlabel(z):
    return torch.argmax(z, dim=1)

print('predicted class labels: ', to_classlabel(smax))
print('true class labels: ', to_classlabel(y_enc))

# Next, we compute the cross entropy for each training example:
def cross_entropy(softmax, y_target):
    return - torch.sum(torch.log(softmax) * (y_target), dim=1)

xent = cross_entropy(smax, y_enc)
print('Cross Entropy:', xent)

print(F.nll_loss(torch.log(smax), y, reduction='none'))
print(F.cross_entropy(Z, y, reduction='none'))
print(F.cross_entropy(Z, y))
print(torch.mean(cross_entropy(smax, y_enc)))

