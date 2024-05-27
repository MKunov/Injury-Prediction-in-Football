import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score
from scipy.optimize import minimize_scalar
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

if torch.cuda.is_available():
    print("CUDA is available!")
# Set the device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the range of hyperparameters
batch_size = 32
hidden_size = 1024
num_layer = 1
learning_rate = 0.0001
alpha = 0.97
gamma = 8.0
dropout_prob = 0.11
num_epochs = 20
ESpatience = 7

injured = np.load('positional_all_players_tensor.npy')
uninjured = np.load('noninjured_positional_all_players_tensor.npy')
data = np.concatenate((injured, uninjured),axis=0)
data = np.delete(data, -2, axis=2)
data = np.delete(data, 13, axis=2)

num_players = data.shape[0]
num_matches = data.shape[1]
num_features = data.shape[2] - 1  # Assuming the last column is the label
window_size = 5
stride = 1

# Collect sequences and labels
sequences = []
labels = []
player_ids = []

for i in range(num_players):
    player_data = data[i]
    num_sequences = (num_matches - window_size) // stride + 1
    for seq in range(num_sequences):
        start_index = seq * stride
        end_index = start_index + window_size
        X_seq = player_data[start_index:end_index, :-1].reshape(-1)
        y_seq = player_data[end_index - 1, -1]
        sequences.append(X_seq)
        labels.append(y_seq)
        player_ids.append(i)

sequences = np.array(sequences)
labels = np.array(labels)
player_ids = np.array(player_ids)

# Calculate sum of injuries per player and distribute them for balancing
unique_player_ids, indices = np.unique(player_ids, return_index=True)
total_injuries_per_player = [labels[player_ids == id].sum() for id in unique_player_ids]

# Sorting players by their injury sum to balance splits
sorted_player_ids = unique_player_ids[np.argsort(total_injuries_per_player)]

# Set a random seed for reproducibility
np.random.seed(42)

# Shuffle player IDs in a reproducible manner
np.random.shuffle(sorted_player_ids)

# Calculate split indices
num_players = len(sorted_player_ids)
train_size = int(0.70 * num_players)
validation_size = int(0.15 * num_players)

# Assign players to train, validation, and test sets
train_players = sorted_player_ids[:train_size]
validation_players = sorted_player_ids[train_size:train_size + validation_size]
test_players = sorted_player_ids[train_size + validation_size:]

# Create masks using assigned groups
train_mask = np.isin(player_ids, train_players)
validation_mask = np.isin(player_ids, validation_players)
test_mask = np.isin(player_ids, test_players)

# Mask arrays
train_sequences = sequences[train_mask]
train_labels = labels[train_mask]
validation_sequences = sequences[validation_mask]
validation_labels = labels[validation_mask]
test_sequences = sequences[test_mask]
test_labels = labels[test_mask]

train_data = [(torch.tensor(seq).view(window_size, num_features).float(), torch.tensor([lbl]).float()) for seq, lbl in zip(train_sequences, train_labels)]
val_data = [(torch.tensor(seq).view(window_size, num_features).float(), torch.tensor([lbl]).float()) for seq, lbl in zip(validation_sequences, validation_labels)]
test_data = [(torch.tensor(seq).view(window_size, num_features).float(), torch.tensor([lbl]).float()) for seq, lbl in zip(test_sequences, test_labels)]

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.95, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)

        # Apply different alpha values based on class labels
        alpha = targets * self.alpha + (1 - targets) * (1-self.alpha)
        modulation = (1 - pt) ** self.gamma
        F_loss = alpha * modulation * BCE_loss

        return F_loss.mean()

class MyRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_prob=0.5):
        super(MyRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout_prob = dropout_prob
        self.num_features = input_size  # Total number of input features

    def forward(self, x):
        # Check if we are in training mode
        if self.training:
            # Apply dropout to the last 10 features
            # Generate a mask for the last 10 features for each element in the batch
            mask = torch.ones_like(x, device=x.device)  # Mask with ones
            # Calculate dropout mask for the last 10 features
            dropout_mask = torch.bernoulli((1 - self.dropout_prob) * torch.ones((x.size(0), x.size(1), 10), device=x.device))
            # Apply the dropout mask to the last 10 features
            mask[:, :, -10:] = dropout_mask
            x = x * mask

        out, _ = self.rnn(x)
        out = self.dropout(out)
        return self.fc(out[:, -1, :])
    
# Early stopping helper class
class EarlyStopping:
    """
    Args:
        patience (int): How long to wait after last time validation loss improved.
                        Default: 5
        verbose (bool): If True, prints a message for each validation loss improvement. 
                        Default: False
        delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                        Default: 0
        path (str): Path for the checkpoint to be saved to.
                    Default: 'checkpoint.pt'
        trace_func (function): trace print function.
    """
    def __init__(self, patience=5, verbose=False, path='checkpoint.pt'):
            self.patience = patience
            self.verbose = verbose
            self.path = path
            self.counter = 0
            self.best_loss = float('inf')
            self.early_stop = False
            self.first_iteration = True

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        if self.first_iteration:
            self.save_checkpoint(val_loss, model)
            self.first_iteration = False

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        # if self.verbose:
            # print(f'Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)

    def reset(self):
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.first_iteration = True
        
#Inintiate early stopping
early_stopping = EarlyStopping(patience=ESpatience, verbose=True, path='best_model.pt')

# Create the risk line
riskline = []

# Randomly sample combinations
for i in range(1000):
    print(f'{batch_size}_{hidden_size}_{num_layer}_{learning_rate}_{alpha}_{gamma}_{dropout_prob}')
    # Reset EarlyStopping for the new training session
    early_stopping.reset()

    # Initialize model, loss function, and optimizer
    model = MyRNN(num_features, hidden_size, num_layer, 1, dropout_prob=dropout_prob).to(device)
    criterion = FocalLoss(alpha=alpha, gamma=gamma).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Batch the data for training
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True)
    # Training the model with early stopping
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_data)
        
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_data:
                inputs = inputs.to(device)
                labels = labels.to(device)
                inputs = inputs.unsqueeze(0)  # Batch dimension
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(1), labels)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_data)
        #print(f'Epoch {epoch+1}: Training Loss: {avg_train_loss:.6f}, Validation Loss: {avg_val_loss:.6f}')
        
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            # print("Stopped early due to no improvement in validationLoss.")
            break

    # Load Model
    model.load_state_dict(torch.load('best_model.pt', map_location=device))
    model.to(device)  # Ensure the model is on the right device
    model.eval()  # Set the model to evaluation mode

    probabilities_list = []
    actual_labels_list = []

    with torch.no_grad():
        for inputs, labels in test_data:
            inputs = inputs.to(device)
            labels = labels.to(device)
            # Assuming inputs are already in the correct shape or you adjust them as necessary
            inputs = inputs.unsqueeze(0)  # Ensure there is a batch dimension if not already
            outputs = model(inputs)
            
            # Apply sigmoid to convert logits to probabilities
            probabilities = torch.sigmoid(outputs)
            
            # Store probabilities and labels for later analysis
            probabilities_list.append(probabilities.cpu().item())  # Get the scalar value for each probability
            actual_labels_list.append(labels.cpu().item())  # Get the scalar value for each label

    # Convert lists to numpy arrays
    probabilities_np = np.array(probabilities_list)
    actuals_np = np.array(actual_labels_list)
        
    # Define bins for the histogram
    bins = np.linspace(np.min(probabilities_np), np.max(probabilities_np), 51)  # 40 bins

    # Calculate histogram for probabilities
    hist, bin_edges = np.histogram(probabilities_np, bins=bins)

    # Calculate the number of positive labels (label == 1) in each bin
    positive_label_counts = np.zeros_like(hist)
    for i in range(len(bin_edges) - 1):
        bin_mask = (probabilities_np >= bin_edges[i]) & (probabilities_np < bin_edges[i+1])
        positive_label_counts[i] = np.sum(actuals_np[bin_mask])

    # Initialize the output array with zeros
    proportion_positive = np.zeros_like(positive_label_counts, dtype=float)
    # Perform the division only where hist is not zero
    np.divide(positive_label_counts, hist, out=proportion_positive, where=(hist != 0))

    # Append values to riskline
    riskline.append(proportion_positive)
    # Every 10 iterations, save the rows to a file and clear the list
    if (i + 1) % 5 == 0:
        # Convert list to a NumPy array
        array_to_save = np.vstack(riskline)
        # Save the array to a file, here using the iteration count to label the file
        np.save(f'data_chunk_long.npy', array_to_save)