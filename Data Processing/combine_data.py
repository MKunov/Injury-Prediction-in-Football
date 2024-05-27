#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 00:31:34 2024

@author: felix
"""

import numpy as np

# Example tensors
heatmap_tensor = np.load('all_players_heatmap_tensor.npy')
features_tensor = np.load('all_players_tensor.npy') 

# Combine into a dictionary for easy access
combined_data = {
    'heatmap': heatmap_tensor,
    'features': features_tensor
}

# Example access
player_index = 0
match_index = 2
player_match_heatmap = combined_data['heatmap'][player_index, match_index]
player_match_features = combined_data['features'][player_index, match_index]

print("Shape of the selected player's match heatmap:", player_match_heatmap.shape)
print("Shape of the selected player's match features:", player_match_features.shape)
print()
print(player_match_features)
