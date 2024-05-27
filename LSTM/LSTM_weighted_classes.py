#RACHEL
import numpy as np
import pandas as pd
import pydot

import tensorflow as tf
from tensorflow.keras import preprocessing
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.models import Sequential 
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight

import matplotlib.pyplot as plt
import seaborn as sns 
sns.set_theme()


#Load data 
data = np.load('positional_all_players_tensor.npy') #shape: [num_players, num_matches, num_features]

#INPUT INTO LSTM RNN
# Example of creating sequences (windowing) for each player
# Assuming 'player_id', 'match_date', and 'injured' columns exist
window_size = 5  # Number of consecutive matches to consider for predicting the next match's injury status


def create_sequences(data, window_size):
    sequences = []
    targets = []
    # Assuming 'data' is a 3D numpy array with shape [num_players, num_matches, num_features]
    num_players = data.shape[0]
    for player in range(num_players):
        player_data = data[player]
        num_matches = player_data.shape[0]
        if num_matches > window_size:
            for i in range(num_matches - window_size):
                # Exclude the second to last feature from the sequence, but include the last feature
                seq = player_data[i:(i + window_size), :-2]
                # The target is the second to last feature of the next match in the sequence
                target = player_data[i + window_size, -2]
                sequences.append(seq)
                targets.append(target)
    return np.array(sequences), np.array(targets)



sequences, targets = create_sequences(data, window_size)


# Now, split these sequences into training and testing sets to get X_train and y_train 
# Followed by the usual preprocessing and training steps

#MAYBE CHANGE THIS TO TIME BASED SPLITS
#Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(sequences, targets, test_size=0.2, random_state=42)

# Normalize features
# Note: You should reshape the data for scaling and then reshape back to the original shape
num_features = X_train.shape[2]
X_train_reshaped = X_train.reshape(-1, num_features)
X_test_reshaped = X_test.reshape(-1, num_features)

scaler = StandardScaler().fit(X_train_reshaped)

X_train_scaled = scaler.transform(X_train_reshaped).reshape(X_train.shape)
X_test_scaled = scaler.transform(X_test_reshaped).reshape(X_test.shape)
X_train_scaled = X_train_scaled.reshape(-1, window_size, num_features)
X_test_scaled = X_test_scaled.reshape(-1, window_size, num_features)

#Building Model

class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(enumerate(class_weights))

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(units=50, #number of LSTM units in each layer 
                    return_sequences=True, #layer outputs full sequence to the next layer
                    ))
    model.add(Dropout(0.2)) #randomly sets input units 0 with a frequency of rate 0.2 - temporarily removes a number of output features of the layer during training. making the mdodel more robust
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(units=1, activation='sigmoid')) #

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Assuming your input shape is (window_size, number_of_features)
input_shape = (5 , X_train_scaled.shape[2])
model_lstm = build_lstm_model(input_shape)


#plot_model(model_lstm, show_shapes = True, show_layer_names=True)

#Train the model
training = model_lstm.fit(
                        X_train_scaled, 
                        y_train, 
                        batch_size=32, 
                        epochs=10, 
                        validation_split=0.2,
                        class_weight=class_weights_dict)

#Evaluate model 

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc, average_precision_score, precision_recall_curve

y_pred = model_lstm.predict(X_test_scaled)
y_pred_binary = np.where(y_pred > 0.5, 1, 0)


print(type(y_test), y_test.shape)
print(type(y_pred_binary), y_pred_binary.shape)

# Evaluation metrics
print(confusion_matrix(y_test, y_pred_binary))
print(classification_report(y_test, y_pred_binary))
print(roc_auc_score(y_test, y_pred))

#Visualisation 

#Confusion Matrix
# Assuming y_test and y_pred_binary are your actual and predicted labels respectively
cm = confusion_matrix(y_test, y_pred_binary)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

#ROC curve
# Compute ROC curve and ROC area for each class
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()

#Precision-Recall Curve
# Calculate precision-recall curve
precision, recall, _ = precision_recall_curve(y_test, y_pred)

# Calculate average precision
average_precision = average_precision_score(y_test, y_pred)

# Plot the precision-recall curve
plt.step(recall, precision, where='post', label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
plt.legend(loc="upper right")
plt.show()

#Class Distribution 
plt.hist([y_test, y_pred_binary], label=['Actual', 'Predicted'], alpha=0.5, bins=2)
plt.xticks([0, 1])
plt.legend()
plt.title('Distribution of Actual vs. Predicted Classes')
plt.show()
