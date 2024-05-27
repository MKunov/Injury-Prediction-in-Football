#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 12:47:24 2024

@author: felix
"""

import numpy as np
import json

# Load the numpy array (assuming it's saved in a file)
player_data = np.load('positional_all_players_tensor.npy')

# Load player info from JSON
with open('all_player_info_new_dates.json', 'r') as file:
    player_info = json.load(file)

# Get the list of player names from the JSON in the same order as in the numpy array
# This assumes the order of the numpy array matches the order in the JSON file
player_names = list(player_info.keys())

# List to store indices of players to remove
indices_to_remove = []

# Check each player's data in the numpy array
for i in range(player_data.shape[0]):
    # Check if all elements in the data slice (except last two columns) are zero
    if np.all(player_data[i, :, :-2] == 0):
        indices_to_remove.append(i)

# Collect names of players to be removed
players_to_remove = [player_names[idx] for idx in indices_to_remove]
print(indices_to_remove)

# Remove players from numpy array and JSON dictionary using the collected indices
new_player_data = np.delete(player_data, indices_to_remove, axis=0)

# Print each player to be removed
for player_name in players_to_remove:
    print(f"Removing player: {player_name}")

# Update the JSON dictionary
for player_name in players_to_remove:
    del player_info[player_name]

# Save the modified numpy array
np.save('updated_player_data.npy', new_player_data)

# Save the modified player info JSON
with open('updated_player_info.json', 'w', encoding='utf-8') as file:
    json.dump(player_info, file, indent=4, ensure_ascii=False)

print(f"Removed {len(indices_to_remove)} players. Updated data saved.")