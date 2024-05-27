import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

data_tensor = np.load('positional_all_players_tensor.npy')
num_players, num_matches, num_features = data_tensor.shape

#Ignoring the positions of players, so starting from 25th feature
feature_names = ["Under Pressure", "50_50", "Pass", "Shot", "Dribble","Pressure","Duel", "Interception","Foul Committed", "Carry", "minutes played"]

#create sequences of matches
sequence_length = 5
# Start from 25th feature
start_feature_index = 24 
# Prepare the data for logistic regression
X = {feature: [] for feature in feature_names}  # exclude injury feature for X
y = []

# Extracting data
for player in range(num_players):
    for start in range(num_matches - sequence_length + 1):
        end = start + sequence_length
        sequence = data_tensor[player, start:end, :]
        
        # The injury status is considered from the last match in the sequence
        y.append(sequence[-1, -1])
        
        for i, feature in enumerate(feature_names):  # exclude the injury status
            # Summing the feature occurrences over the sequence
            X[feature].append(np.sum(sequence[:, i + start_feature_index]))

#Iterates over each player, then over each possible sequences of matches for that player. Appends the injury status of the last match of the sequence to 'y' list
#Next, loops through each feature and appends the sum of values for that feature over the sequence to its corresponding list in 'X'
            
# Determine the layout of the subplot grid
rows = 3  # Number of rows of subplots
cols = 4  # Number of columns of subplots

fig, axes = plt.subplots(rows, cols, figsize=(20, 10))
fig.suptitle('Logistic Regression Fits for Various Features')

# Flatten the axes array for easier iteration
axes = axes.flatten()

# Fit a logistic regression model for each feature and plot
for i, feature in enumerate(feature_names):
    ax=axes[i]
    # Logistic regression
    model = LogisticRegression()
    model.fit(np.array(X[feature]).reshape(-1, 1), y)
    coef = model.coef_[0][0]
    
    # Scatter plot
    ax.scatter(X[feature], y, alpha=0.5, label='Data Points')
    ax.set_xlabel(f'Sum of {feature} over 5 matches')
    ax.set_ylabel('Injury Status (0=No, 1=Yes)')
    ax.set_title(f'Feature: {feature} / Coeff: {coef:.2f}')
    
    # Create a sequence of values for the feature to plot the model predictions
    x_values = np.linspace(min(X[feature]), max(X[feature]), 300).reshape(-1, 1)
    y_values = model.predict_proba(x_values)[:, 1]
    ax.plot(x_values, y_values, color='red', label='Model Fit')


    ax.legend()

plt.tight_layout(rect=[0,0.03,1,0.95])
plt.show()

#For each feature I fit a logistic regression model using feature dataand injury status
#Create a scatter plot of aggregated feature data against the injury status, and plot logistic regression model's predicted probabilities fto show the fit