import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralNetwork import NeuralNetwork

# read binary files and convert them to numpy arrays
with open('train-images.idx3-ubyte', 'rb') as f:
    train_images = np.fromfile(f, dtype=np.uint8, count=16 + (28 * 28 * 60000))[16:]
    train_images = train_images.reshape((60000, 28 * 28))
with open('train-labels.idx1-ubyte', 'rb') as file:
    train_labels = np.frombuffer(file.read(), np.uint8, offset=8)
with open('t10k-images.idx3-ubyte', 'rb') as file:
    t10k_img = np.frombuffer(file.read(), np.uint8, offset=16)
    t10k_img = t10k_img.reshape((10000, 28 * 28))
with open('t10k-labels.idx1-ubyte', 'rb') as file:
    t10k_label = np.frombuffer(file.read(), np.uint8, offset=8)

# convert numpy arrays to pandas dataframes
train_images_df = pd.DataFrame(train_images)
train_labels_df = pd.DataFrame(train_labels, columns=['label'])
t10k_img_df = pd.DataFrame(t10k_img)
t10k_label_df = pd.DataFrame(t10k_label, columns=['label'])

# Split the training data into training and validation sets
train_len = len(train_images_df)
val_len = int(train_len * 0.2)
train_images_df = train_images_df.iloc[val_len:]
train_labels_df = train_labels_df.iloc[val_len:]
val_images_df = train_images_df.iloc[:val_len]
val_labels_df = train_labels_df.iloc[:val_len]

# Split the test data into test and validation sets
test_len = len(t10k_img_df)
val_len = int(test_len * 0.5)
t10k_img_df = t10k_img_df.iloc[val_len:]
t10k_label_df = t10k_label_df.iloc[val_len:]
val_img_df = t10k_img_df.iloc[:val_len]
val_label_df = t10k_label_df.iloc[:val_len]
# Create the training dataset
train_dataset = torch.utils.data.TensorDataset(torch.tensor(train_images_df.values, dtype=torch.float32), torch.tensor(train_labels_df.values, dtype=torch.long))

# Create the validation dataset
val_dataset = torch.utils.data.TensorDataset(torch.tensor(val_images_df.values, dtype=torch.float32), torch.tensor(val_labels_df.values, dtype=torch.long))

# Create the test dataset
test_dataset = torch.utils.data.TensorDataset(torch.tensor(t10k_img_df.values, dtype=torch.float32), torch.tensor(t10k_label_df.values, dtype=torch.long))

# Create the validation test dataset
val_test_dataset = torch.utils.data.TensorDataset(torch.tensor(val_img_df.values, dtype=torch.float32), torch.tensor(val_label_df.values, dtype=torch.long))
model = NeuralNetwork()
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Set number of epochs
epochs = 5

# Loop through each epoch
for epoch in range(epochs):
    running_loss = 0
    for i, (images, labels) in enumerate(train_dataset):
        # Clear gradient
        optimizer.zero_grad()
        # Forward pass
        outputs = model(images)
        # Calculate loss
        labels = labels.view(-1, 1)
        if outputs.dim() == 1:
            outputs = outputs.unsqueeze(0)
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        # Update parameters
        optimizer.step()
        running_loss += loss.item()
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, epochs, running_loss / len(train_dataset)))
