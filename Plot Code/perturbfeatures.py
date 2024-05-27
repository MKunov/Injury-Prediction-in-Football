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


categories = ["Under Pressure", "50-50", "Pass", "Shot", "Dribble", "Pressure", "Duel",
                "Interception", "Foul Committed", "Carry", "Minutes Played"]
points = ['1st', '2nd', '3rd', '4th', '5th']
positions = ['Goalkeeper', 'Left Center Back', 'Left Back', 'Left Wing Back', 
              'Center Back','Right Back', 'Right Center Back', 'Right Wing Back', 
              'Left Midfield', 'Left Attacking Midfield', 'Left Defensive Midfield', 
              'Left Center Midfield','Center Defensive Midfield',
              'Center Attacking Midfield', 'Right Center Midfield', 
              'Right Defensive Midfield', 'Right Attacking Midfield', 'Right Midfield',
              'Left Wing', 'Left Center Forward', 'Center Forward', 'Right Wing', 
              'Right Center Forward']
print(len(positions))

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
model_path = 'BESTRECALL/F0.318_model_32_1024_1_0.97_8.0_0.0001_0.11.pt'

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
    
def mask_features(test_data, features_to_mask):
    modified_test_data = []
    for sequence, label in test_data:
        modified_sequence = sequence.clone()
        modified_sequence[:, features_to_mask] = 0
        modified_test_data.append((modified_sequence, label))
    return modified_test_data

def mask_time_points(test_data, time_points_to_mask):
    modified_test_data = []
    for sequence, label in test_data:
        modified_sequence = sequence.clone()
        modified_sequence[time_points_to_mask, :] = 0
        modified_test_data.append((modified_sequence, label))
    return modified_test_data


model = MyRNN(num_features, hidden_size, num_layer, 1, dropout_prob=dropout_prob).to(device)
# Load Model
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)  # Ensure the model is on the right device
model.eval()  # Set the model to evaluation mode

## ORIGINAL F1 ##
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

beta = 1
def f1_optimize(threshold, probabilities, actuals):
    # Convert probabilities to binary predictions based on the threshold
    predictions = (probabilities > threshold).astype(int)
    # Calculate F1 score
    return -fbeta_score(actuals, predictions, beta=beta)  # Negative F1 for minimization

# Find best 3% threshold and best threshold and corresponding probabilities
threeP_threshold = np.percentile(probabilities_np, 97)
result = minimize_scalar(f1_optimize, bounds=(0, np.max(probabilities_np)+0.1), args=(probabilities_np, actuals_np), method='bounded')
best_threshold = result.x
predictions = (probabilities_np >  best_threshold).astype(int)

# Calc Metrics
original_f1 = fbeta_score(actuals_np, predictions, beta=beta)
print(original_f1)


# INITIALISE FOR PLOTS #
modi_point_f1 = []
modi_feature_f1 = []
modi_position_f1 = []

## TIME POINTS ##
for point in range(window_size):
    modified_test_data = mask_time_points(test_data, [point])

    probabilities_list = []
    actual_labels_list = []

    with torch.no_grad():
        for inputs, labels in modified_test_data:
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

    beta = 1
    def f1_optimize(threshold, probabilities, actuals):
        # Convert probabilities to binary predictions based on the threshold
        predictions = (probabilities > threshold).astype(int)
        # Calculate F1 score
        return -fbeta_score(actuals, predictions, beta=beta)  # Negative F1 for minimization

    # Find best 3% threshold and best threshold and corresponding probabilities
    threeP_threshold = np.percentile(probabilities_np, 97)
    result = minimize_scalar(f1_optimize, bounds=(0, np.max(probabilities_np)+0.1), args=(probabilities_np, actuals_np), method='bounded')
    best_threshold = result.x
    predictions = (probabilities_np >  best_threshold).astype(int)

    # Calc Metrics
    f1 = fbeta_score(actuals_np, predictions, beta=beta)
    modi_point_f1.append(f1)

## FEATURES ##
for parameter in range(len(categories)):
    modified_test_data = mask_features(test_data, [-(parameter+1)])

    probabilities_list = []
    actual_labels_list = []

    with torch.no_grad():
        for inputs, labels in modified_test_data:
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

    beta = 1
    def f1_optimize(threshold, probabilities, actuals):
        # Convert probabilities to binary predictions based on the threshold
        predictions = (probabilities > threshold).astype(int)
        # Calculate F1 score
        return -fbeta_score(actuals, predictions, beta=beta)  # Negative F1 for minimization

    # Find best 3% threshold and best threshold and corresponding probabilities
    threeP_threshold = np.percentile(probabilities_np, 97)
    result = minimize_scalar(f1_optimize, bounds=(0, np.max(probabilities_np)+0.1), args=(probabilities_np, actuals_np), method='bounded')
    best_threshold = result.x
    predictions = (probabilities_np >  best_threshold).astype(int)

    # Calc Metrics
    f1 = fbeta_score(actuals_np, predictions, beta=beta)
    modi_feature_f1.append(f1)

## Positions ##
for position in range(len(positions)):
    modified_test_data = mask_features(test_data, [position])

    probabilities_list = []
    actual_labels_list = []

    with torch.no_grad():
        for inputs, labels in modified_test_data:
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

    beta = 1
    def f1_optimize(threshold, probabilities, actuals):
        # Convert probabilities to binary predictions based on the threshold
        predictions = (probabilities > threshold).astype(int)
        # Calculate F1 score
        return -fbeta_score(actuals, predictions, beta=beta)  # Negative F1 for minimization

    # Find best 3% threshold and best threshold and corresponding probabilities
    threeP_threshold = np.percentile(probabilities_np, 97)
    result = minimize_scalar(f1_optimize, bounds=(0, np.max(probabilities_np)+0.1), args=(probabilities_np, actuals_np), method='bounded')
    best_threshold = result.x
    predictions = (probabilities_np >  best_threshold).astype(int)

    # Calc Metrics
    f1 = fbeta_score(actuals_np, predictions, beta=beta)
    modi_position_f1.append(f1)




modi_point_f1 = [((original_f1-f1)/original_f1)*100 for f1 in modi_point_f1]
modi_feature_f1 = [((original_f1-f1)/original_f1)*100 for f1 in modi_feature_f1]
sorted_data = sorted(zip(modi_feature_f1, categories), key=lambda x: x[0], reverse=False)
sorted_modi_feature_f1, sorted_categories = zip(*sorted_data)
modi_position_f1 = [((original_f1-f1)/original_f1)*100 for f1 in modi_position_f1]
sorted_data_positions = sorted(zip(modi_position_f1, positions), key=lambda x: x[0], reverse=False)
sorted_modi_position_f1, sorted_positions = zip(*sorted_data_positions)

fig, (ax1,ax2,ax3) = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
bins1 = np.linspace(1,5,6)
widths = np.diff(bins1)
# Creating the bar plot
ax1.bar(bins1[:-1] + widths / 2, modi_point_f1, width=widths, color='blue', alpha=0.8, label='F1 % Decrease')
# Setting the x-axis ticks and labels
ax1.set_xticks(bins1[:-1] + widths / 2)  # Setting tick positions to the center of the bars
ax1.set_xticklabels(points, rotation=45, ha="right")  # ha is horizontal alignment
# Adding grid, label, legend, and formatting
ax1.grid(True)
ax1.set_xlabel('Sequence Point')
ax1.set_ylabel('% Decrease In F1')
ax1.legend()
ax1.set_title('Percentage Decrease in F1 Score In Absence Of Sequence Point')

bins2 = np.linspace(1,11,12)
widths = np.diff(bins2)
# Creating the bar plot
ax2.bar(bins2[:-1] + widths / 2, sorted_modi_feature_f1, width=widths, color='blue', alpha=0.8, label='F1 % Decrease')
# Setting the x-axis ticks and labels
ax2.set_xticks(bins2[:-1] + widths / 2)  # Setting tick positions to the center of the bars
ax2.set_xticklabels(sorted_categories, rotation=45, ha="right")  # ha is horizontal alignment
# Adding grid, label, legend, and formatting
ax2.grid(True)
ax2.set_xlabel('Feature Type')
ax2.set_ylabel('% Decrease In F1')
ax2.legend()
ax2.set_title('Percentage Decrease in F1 Score In Absence Of Feature')

bins3 = np.linspace(1,23,24)
widths = np.diff(bins3)
# Creating the bar plot
ax3.bar(bins3[:-1] + widths / 2, sorted_modi_position_f1, width=widths, color='blue', alpha=0.8, label='F1 % Decrease')
# Setting the x-axis ticks and labels
ax3.set_xticks(bins3[:-1] + widths / 2)  # Setting tick positions to the center of the bars
# List comprehension to convert each position to its initials
abbreviated_positions = [''.join([word[0] for word in position.split()]) for position in sorted_positions]
ax3.set_xticklabels(abbreviated_positions, rotation=45, ha="right")  # ha is horizontal alignment
# Adding grid, label, legend, and formatting
ax3.grid(True)
ax3.set_xlabel('Position Type')
ax3.set_ylabel('% Decrease In F1')
ax3.legend()
ax3.set_title('Percentage Decrease in F1 Score In Absence Of Position')

fig.tight_layout()
fig.savefig(f'Impact_Of_Features.png', format='png', dpi=300)  # Save the figure

# # Create a figure with 11 subplots arranged in 3 rows and 4 columns
# fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(15, 15))
# axes = axes.flatten()  # Flatten the array of axes to make indexing easier
        
#     # Define bins for the histogram
#     bins = np.linspace(np.min(probabilities_np), np.max(probabilities_np), 21)  # 20 bins

#     # Calculate histogram for probabilities
#     hist, bin_edges = np.histogram(probabilities_np, bins=bins)

#     # Calculate the number of positive labels (label == 1) in each bin
#     positive_label_counts = np.zeros_like(hist)
#     for i in range(len(bin_edges) - 1):
#         bin_mask = (probabilities_np >= bin_edges[i]) & (probabilities_np < bin_edges[i+1])
#         positive_label_counts[i] = np.sum(actuals_np[bin_mask])

#     ax = axes[parameter]
#     # Plotting the histograms and lines
#     ax.bar(bins[:-1], hist, width=np.diff(bins), color='blue', alpha=0.5, label='All Probabilities')
#     ax.bar(bins[:-1], positive_label_counts, width=np.diff(bins), color='red', alpha=0.5, label='Positive Labels (Label=1)')
#     ax.axvline(x=best_threshold, color='red', linestyle='dashed', linewidth=1, label=f'Maximised F1 Threshold {best_threshold:.2f}')
#     ax.axvline(x=threeP_threshold, color='green', linestyle='dashed', linewidth=1, label=f'3% Threshold: {threeP_threshold:.2f}')

#     # Setting titles and labels
#     ax.set_title(f'{categories[parameter]}, F1 Loss:{round(original_f1-f1,4)}')
#     ax.set_xlabel('Probability')
#     ax.set_ylabel('Frequency')
#     ax.grid(True)
#     ax.legend()

# # Adjust layout to prevent overlap
# fig.tight_layout()
# fig.savefig(f'perturbed_model_{batch_size}_{hidden_size}_{num_layer}_{alpha}_{gamma}_{learning_rate}_{dropout_prob}.png', format='png', dpi=300)  # Save the figure