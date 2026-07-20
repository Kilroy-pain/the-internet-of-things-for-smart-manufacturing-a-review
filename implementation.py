import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple IoMT-inspired framework: A virtual machine network
# This is a basic neural network model to simulate a "virtual machine network" concept
class VirtualMachineNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(VirtualMachineNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

# Simulate IoMT data (e.g., sensor readings, equipment states, etc.)
def generate_dummy_iomt_data(num_samples, input_size):
    return np.random.rand(num_samples, input_size).astype(np.float32), np.random.randint(0, 2, size=(num_samples,))

# Training function
def train_model(model, criterion, optimizer, data, labels, epochs=100):
    for epoch in range(epochs):
        # Convert data to PyTorch tensors
        inputs = torch.tensor(data)
        targets = torch.tensor(labels, dtype=torch.long)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Test the model
def test_model(model, data):
    with torch.no_grad():
        inputs = torch.tensor(data)
        outputs = model(inputs)
        predictions = torch.argmax(outputs, dim=1)
        return predictions.numpy()

if __name__ == '__main__':
    # Hyperparameters
    input_size = 10  # Number of features in IoMT data
    hidden_size = 16  # Number of hidden units in the virtual machine network
    output_size = 2  # Number of output classes (e.g., normal/abnormal state)
    learning_rate = 0.01
    epochs = 50

    # Generate dummy IoMT data
    num_samples = 100
    data, labels = generate_dummy_iomt_data(num_samples, input_size)

    # Split data into training and testing sets
    train_size = int(0.8 * num_samples)
    train_data, test_data = data[:train_size], data[train_size:]
    train_labels, test_labels = labels[:train_size], labels[train_size:]

    # Initialize the model, loss function, and optimizer
    model = VirtualMachineNetwork(input_size, hidden_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    print("Training the Virtual Machine Network...")
    train_model(model, criterion, optimizer, train_data, train_labels, epochs)

    # Test the model
    print("Testing the Virtual Machine Network...")
    predictions = test_model(model, test_data)

    # Evaluate the model
    accuracy = np.mean(predictions == test_labels)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")