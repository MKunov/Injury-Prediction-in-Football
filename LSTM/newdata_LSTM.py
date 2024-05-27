#RACHEL
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, average_precision_score, confusion_matrix, roc_curve, auc
from sklearn.utils.class_weight import compute_class_weight

#Load data 
data = np.load('positional_all_players_tensor.npy') #shape: [num_players, num_matches, num_features]

#INPUT INTO LSTM RNN
num_players = 307  # total players
num_matches = 38   # matches per player
num_features = 37  # features per match
window_size = 7

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prepare dataset
labels = []
for i in range(num_players):
    player_data = data[i]
    labels.extend(player_data[window_size:, -1])  # Collecting all labels

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

train_data = []
val_data = []

#Data preprocessing and splitting 
for i in range(num_players):
    player_data = data[i]  # Get data for the ith player
    num_sequences = num_matches - window_size  # Calculate the number of sequences

    for seq in range(num_sequences):
        # Get the features for current window
        start_index = seq 
        end_index = start_index + window_size
        X_seq = player_data[start_index:end_index, :-2]  # ignoring injury features 

        # Get the label for the last match in the window
        y_seq = player_data[end_index - 1, -1]  # last feature is injured

        # Convert to torch tensors
        X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y_seq, dtype=torch.float32).to(device) 

        # Add to train or validation data
        if random.random() > 0.2:  # 80% chance to add to training
            train_data.append((X_tensor, y_tensor))
        else:
            val_data.append((X_tensor, y_tensor))


# Convert to DataLoader for easier batch processing
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

# Define LSTM Model using PyTorch
class LSTMModel(nn.Module):
    def __init__(self, num_features, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(num_features, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        print("Forward input shape:", x.shape) #[batch, seq, feature]
        _, (h_n, _) = self.lstm(x) #h_n shape: [num_layers, batch, hidden_dim]
        print("LSTM output shape:", h_n.shape)
        last_hidden_state = h_n[-1] #Taking last layer's output
        out = self.fc(last_hidden_state) #Shape after fc should be [batch, output_dim]
        return out.squeeze() #Squeezing to make sure it fits [batch] if output_dim is 1

# Model instantiation
model = LSTMModel(num_features=num_features - 2, hidden_dim=50, num_layers=4, output_dim=1).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss(weight=class_weights)

# Training the model
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss= 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets.squeeze())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Average Loss: {total_loss / len(train_loader)}")

# Model evaluation simplified
def evaluate_model(model, val_loader, device):
    model.eval()
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            # Add your evaluation metrics

# Example of running training and evaluation
for epoch in range(10):  # Number of epochs
    print(f"Epoch {epoch + 1}")
    train_epoch(model, train_loader, criterion, optimizer, device)
    evaluate_model(model, val_loader, device)

#Evaluation Metrics in PyTorch 
def evaluate(model, loader, device):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            predicted_probs = torch.sigmoid(outputs).cpu().numpy()
            predictions.extend(predicted_probs.flatten())  # Convert to 1D array
            actuals.extend(targets.cpu().numpy().flatten())
    
    return np.array(predictions), np.array(actuals)

# After your model has been trained:
preds, actuals = evaluate(model, val_loader, device)
pred_binary = np.where(preds > 0.5, 1, 0)

print("ROC AUC Score:", roc_auc_score(actuals, preds))
print("F1 Score:", f1_score(actuals, pred_binary))

#Confusion Matrix
cm = confusion_matrix(actuals, pred_binary)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(actuals, preds)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(actuals, preds)
average_precision = average_precision_score(actuals, preds)

plt.step(recall, precision, where='post', label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title(f'Precision-Recall curve: AP={average_precision:0.2f}')
plt.legend(loc="upper right")
plt.show()

# Assuming you have player_ids_test corresponding to validation samples
"""player_ids_test = np.random.randint(0, 307, size=len(preds))  # Placeholder

for unique_id in np.unique(player_ids_test):
    print(f"Player {unique_id}:")
    player_predictions = preds[player_ids_test == unique_id]
    for i, prediction in enumerate(player_predictions):
        print(f"\tSequence {i + 1}: {prediction:.4f}")
    print("\n")
"""