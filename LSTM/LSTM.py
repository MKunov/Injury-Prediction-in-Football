from MTO_SW_LSTM import MTO_SW_LSTM

window_size = 5
hidden_size = 100
num_layers = 2
n_features = 35
stride = 1
bsize = 1
device = 'cpu'
bidir = False
nout = [33, 307, 1]
dropout = 0.1
dropout2 = 0.2

rnn = MTO_SW_LSTM(window_size,hidden_size,num_layers,n_features,stride,bsize,device,bidir,nout,dropout,dropout2)
print(rnn)

import torch
import numpy as np

#Load data 
data = np.load('positional_all_players_tensor.npy') #shape: [num_players, num_matches, num_features]
import torch
import random


num_players = 307  # total players
num_matches = 38   # matches per player
num_features = 35  # features per match
#nout = 1      # output size, assuming your model ouTtputs a vector of length 5 for each input

train_data = []
val_data = []
device = torch.device("cpu")  # or "cuda" if you have GPU support

## Generate sliding window sequences for each player
for i in range(num_players):
    player_data = data[i]  # Get data for the ith player
    num_sequences = (num_matches - window_size) // stride + 1  # Calculate the number of sequences

    for seq in range(num_sequences):
        # Get the features for current window
        start_index = seq * stride
        end_index = start_index + window_size
        X_seq = player_data[start_index:end_index, :-2]  # ignoring injury features 

        # Get the label for the last match in the window
        y_seq = player_data[end_index - 1, -1]  # last feature is injured

        # Convert to torch tensors
        X_tensor = torch.tensor(X_seq, dtype=torch.float).to(device)
        y_tensor = torch.tensor([y_seq], dtype=torch.float).to(device) 

        # Add to train or validation data
        if random.random() > 0.2:  # 80% chance to add to training
            train_data.append((X_tensor, y_tensor))
        else:
            val_data.append((X_tensor, y_tensor))
 
epochs = 10
learning_rate = 0.1
using_gpu = False
testnbr = 10
rnn.train_model(train_data, val_data, epochs, bsize, learning_rate, using_gpu, testnbr)




