#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 15:30:23 2024

@author: felix
"""

from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

uninjured = np.load('noninjured_positional_all_players_tensor.npy')
injured = np.load('positional_all_players_tensor.npy')

# Concatenate along the first dimension
data = np.concatenate((uninjured, injured), axis=0)

X = data[:,:,:-2]
y = data[:,:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

y_train_aggregated = y_train[:, -1].unsqueeze(1)


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])  
        return torch.sigmoid(out)  


input_size = X.shape[2]
hidden_size = 128
num_layers = 2
output_size = 1

model = RNN(input_size, hidden_size, num_layers, output_size)

criterion = nn.BCELoss()  
optimizer = optim.Adam(model.parameters(), lr=0.000001)

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

num_epochs = 10

for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(train_loader):
        
        outputs = model(inputs)
        labels = labels[:, -1] 
        loss = criterion(outputs.squeeze(), labels)

        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')


with torch.no_grad():
    all_player_predictions = []
    true_labels = []
    for i in range(len(X_test)):
        player_predictions = []
        for j in range(X_test.size(1)):
            input_matches = X_test[i:i + 1, :j + 1, :]
            output = model(input_matches)
            predicted = output[0, -1].item()
            player_predictions.append(predicted)
        all_player_predictions.append(player_predictions)
        true_labels.append(y_test[i, -1].item())



num_players = len(all_player_predictions)
num_matches = len(all_player_predictions[0])
# Convert predictions and true labels to numpy arrays
all_player_predictions = np.array(all_player_predictions)
true_labels = np.array(true_labels)

# Flatten the arrays
all_player_predictions = all_player_predictions.flatten()
true_labels = true_labels.flatten()

# Binarize the predictions
all_player_predictions_binary = (all_player_predictions > 0.5).astype(int)

# Calculate F1 score
f1 = f1_score(true_labels, all_player_predictions_binary)

print("F1 Score:", f1)

