# pytorch_tut
import torch

tensor0d = torch.tensor(1)

tensor1d = torch.tensor([1, 2, 3])

tensor2d = torch.tensor([[1, 2], 
                         [3, 4]])

tensor3d = torch.tensor([[[1, 2], [3, 4]], 
                         [[5, 6], [7, 8]]])
# print(tensor2d)
# print(f'transposed: {tensor2d.T}')
# print(tensor2d.shape)
# print(tensor2d.size())
# print(tensor2d.dim())
# print(tensor2d.dtype)
# print(tensor2d.device)

# print(tensor2d.matmul(tensor2d.T))

print('=======================================')
print('Simple logistic regression forward pass')
print('=======================================')
import torch.nn.functional as F
y = torch.tensor([1.0])
x1 = torch.tensor([1.1])
w1 = torch.tensor([2.2], requires_grad=True)
b = torch.tensor([0.0], requires_grad=True)
z = x1 * w1 + b
a = torch.sigmoid(z)
loss = F.binary_cross_entropy(a, y)
print(f'loss: {loss}')


# compute gradients
from torch.autograd import grad
from torch.utils.data import Dataset

# grad_L_w1 = grad(loss, w1, retain_graph=True)
# grad_L_b = grad(loss, b, retain_graph=True)
# print(f'WEIGHT: grad_L_w1: {grad_L_w1}')
# print(f'BIAS: grad_L_b: {grad_L_b}')

# shortcut - use backward()
loss.backward()
print(f'WEIGHT: grad_L_w1: {w1.grad}') 
print(f'BIAS: grad_L_b: {b.grad}')




print('\n==========================================')
print('Multilayer perceptron with 2 hidden layers')
print('==========================================')

class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        self.layers = torch.nn.Sequential(

            # Note: ReLU outputs the input directly if it's positive, and outputs zero if the input is negative.
            # This is a non-linear activation function, which helps the model learn complex patterns.
            # returns 0 if input negative, else returns input
            # 1st hidden layer
            # .Linear lyer multiplies input by a weights matrix and adds a bias vector
            torch.nn.Linear(num_inputs, 30),
            torch.nn.ReLU(),

            # 2nd hidden layer
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),

            # output layer
            torch.nn.Linear(20, num_outputs),
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits

# create new NN object
torch.manual_seed(123)
model = NeuralNetwork(50, 3)
print(model)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total number of trainable model parameters:", num_params)
print('Weight parameter matrix for first linear layer:')
print(model.layers[0].weight)
print(f'Shape: {model.layers[0].weight.shape}')

# do a forward pass
torch.manual_seed(123)
X = torch.rand((1, 50))
out = model(X)
print(f'First forward pass - unscaled output: {out}')

# simple inference: default class membership probabilities for our predictions
# softmax function converts to probabilities
with torch.no_grad():
    out = torch.softmax(model(X), dim=1)
print(out)

print('=======================================')
print('DataSet / Dataloader introduction')
print('=======================================')
# toy dataset of five training examples with two features each
X_train = torch.tensor([
    [-1.2, 3.1],
    [-0.9, 2.9],
    [-0.5, 2.6],
    [2.3, -1.1],
    [2.7, -1.5]
])

# tensor containing the corresponding class labels: three examples belong to class 0, and two examples belong to class 1
y_train = torch.tensor([0, 0, 0, 1, 1])


# test set consisting of two entries
X_test = torch.tensor([
    [-0.8, 2.8],
    [2.6, -1.6],
])
y_test = torch.tensor([0, 1])


# create a custom dataset class, ToyDataset, by subclassing from PyTorch’s Dataset parent class
from torch.utils.data  import Dataset, DataLoader

class ToyDataset(Dataset):
    def __init__(self, X, y):
        self.features = X
        self.labels = y

    def __getitem__(self, index):
        one_x = self.features[index]
        one_y = self.labels[index]
        return one_x, one_y

    def __len__(self):
        return self.labels.shape[0]

train_ds = ToyDataset(X_train, y_train)
test_ds = ToyDataset(X_test, y_test)



torch.manual_seed(123)

train_loader = DataLoader(
    dataset=train_ds,
    batch_size=2,
    shuffle=True,
    num_workers=0,
    drop_last=True
)

test_loader = DataLoader(
    dataset=test_ds,
    batch_size=2,
    shuffle=False,
    num_workers=0
)

for idx, (x, y) in enumerate(train_loader):
    print(f"Batch {idx+1}:", x, y)



print('=======================================')
print('Training of simple NN: 1 NO MAC GPU')
print('=======================================')


import torch.nn.functional as F

torch.manual_seed(123)
model = NeuralNetwork(num_inputs=2, num_outputs=2)
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total number of trainable model parameters:", num_params)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.to(device)
optimizer = torch.optim.SGD(
    model.parameters(), lr=0.5
)

num_epochs = 3

for epoch in range(num_epochs): 

    model.train()
    for batch_idx, (features, labels) in enumerate(train_loader):
        # features, labels = features.to(device), labels.to(device)
        logits = model(features)

        loss = F.cross_entropy(logits, labels)

        # use the gradients to update the model parameters to minimize the loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ### LOGGING
        print(f"Epoch: {epoch+1:03d}/{num_epochs:03d}"
              f" | Batch {batch_idx:03d}/{len(train_loader):03d}"
              f" | Train Loss: {loss:.2f}")

    model.eval()

with torch.no_grad():
    outputs = model(X_train)
    print(outputs)

# display the class membership probabilities
print('Class membership probabilities:')
torch.set_printoptions(sci_mode=False)
probas = torch.softmax(outputs, dim=1)
print(probas)

predictions = torch.argmax(probas, dim=1)
print(predictions)

# compute prediction accuracy
def compute_accuracy(model, dataloader):

    model = model.eval()
    correct = 0.0
    total_examples = 0

    for idx, (features, labels) in enumerate(dataloader):

        with torch.no_grad():
            logits = model(features)

        predictions = torch.argmax(logits, dim=1)
        compare = labels == predictions
        correct += torch.sum(compare)
        total_examples += len(compare)

    return (correct / total_examples).item()

print(f'Accuracy: {compute_accuracy(model, train_loader)}')


print('=======================================')
print('Training of simple NN: 2 WITH MAC GPU')
print('=======================================')
model = NeuralNetwork(num_inputs=2, num_outputs=2)
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total number of trainable model parameters:", num_params)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
optimizer = torch.optim.SGD(
    model.parameters(), lr=0.5
)

num_epochs = 3

for epoch in range(num_epochs): 

    model.train()
    for batch_idx, (features, labels) in enumerate(train_loader):
        features, labels = features.to(device), labels.to(device)
        logits = model(features)

        loss = F.cross_entropy(logits, labels)

        # use the gradients to update the model parameters to minimize the loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ### LOGGING
        print(f"Epoch: {epoch+1:03d}/{num_epochs:03d}"
              f" | Batch {batch_idx:03d}/{len(train_loader):03d}"
              f" | Train Loss: {loss:.2f}")

    model.eval()

with torch.no_grad():
    outputs = model(X_train)
    print(outputs)

# display the class membership probabilities
print('Class membership probabilities:')
torch.set_printoptions(sci_mode=False)
probas = torch.softmax(outputs, dim=1)
print(probas)

predictions = torch.argmax(probas, dim=1)
print(predictions)



print(f'Accuracy: {compute_accuracy(model, train_loader)}')

