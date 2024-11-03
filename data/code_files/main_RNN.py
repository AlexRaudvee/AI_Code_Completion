import torch 
import torch.nn as nn
import torch.optim as optim
from Net import RNN
from config import device
from load_dataset import test_loader, train_loader

# Hyperparameters 
input_size = 28
sequence_length = 28 
num_layers = 2
hidden_size = 256 
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 2

# initialize network
model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)

# Loss and optimizer 
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train network 
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        # get data to cuda if possible
        data = data.to(device).squeeze(1)
        targets = targets.to(device)

        # forward 
        scores = model(data)
        loss = criterion(scores, targets)

        # backward 
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()


# check accuracy on training and test to see how good our model
        
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0

    # Set model to eval
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device).squeeze(1)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    # Toggle model back to train
    model.train()
    return num_correct / num_samples


print(f"Accuracy on training set: {check_accuracy(train_loader, model)*100:2f}")
print(f"Accuracy on test set: {check_accuracy(test_loader, model)*100:.2f}")


