import numpy as np
import pandas as pd
import torch

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

# concatenate dataframes to create single dataframes
train_df = pd.concat([train_images_df, train_labels_df], axis=1)
t10k_df = pd.concat([t10k_img_df, t10k_label_df], axis=1)

# Split the training data into training and validation sets
train_images_df, val_images_df, train_labels_df, val_labels_df = train_test_split(train_images_df, train_labels_df, test_size=0.2, random_state=42)

# Split the test data into test and validation sets
t10k_img_df, val_img_df, t10k_label_df, val_label_df = train_test_split(t10k_img_df, t10k_label_df, test_size=0.5, random_state=42)