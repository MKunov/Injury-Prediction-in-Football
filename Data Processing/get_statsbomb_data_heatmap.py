#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 20:24:36 2024

@author: felix
"""

import numpy as np
import pandas as pd
import json
from scipy.ndimage import gaussian_filter
from statsbombpy import sb
import warnings
from tqdm import tqdm

def process_all_players_and_save(player_info, competition_id, season_id, bins=(24, 24), smoothing=1, save_filename='all_players_heatmap_tensor.npy'):
    warnings.filterwarnings("ignore", category=UserWarning, message=".*credentials were not supplied.*")

    all_players_data = []
    matches = sb.matches(competition_id=competition_id, season_id=season_id)
    
    # Wrap the iteration over players with tqdm for progress indication
    for player_name, info in player_info.items():
        club_matches = matches[(matches['home_team'] == info["Team"]) | (matches['away_team'] == info["Team"])]
        club_matches = club_matches.sort_values(by='match_date')

        player_matches_data = []

        # tqdm bar here for match processing if desired
        for _, match_row in tqdm(club_matches.iterrows(), total=len(club_matches), desc=f"Matches for {player_name}"):
            match_id = match_row['match_id']
            match_date = match_row['match_date']
            events = sb.events(match_id=match_id)
            player_events = events[events['player'] == player_name]

            heatmap_extended = np.zeros((bins[0], bins[1], 2))  # Initialize with zeros including two extra channels

            if not player_events.empty:
                locations = np.vstack(player_events['location'].dropna())
                x_locations, y_locations = locations[:, 0], locations[:, 1]
                heatmap, _, _ = np.histogram2d(x_locations, y_locations, bins=bins, range=[[0, 120], [0, 80]])
                
                # Reflect the heatmap about the diagonal (top-left to bottom-right)
                heatmap = np.transpose(heatmap)  # Transpose the matrix
                
                if smoothing > 0:
                    heatmap = gaussian_filter(heatmap, smoothing)

                heatmap_extended[:, :, 0] = heatmap  # Heatmap data
                
                # Set additional data
                if pd.isna(player_events.iloc[0].get('substitution_outcome')):
                    minutes_played = player_events.iloc[-1]['minute']
                else:
                    minutes_played = player_events.iloc[-1]['minute'] - player_events.iloc[0]['minute']
                injury_status = 1 if match_date in info["Injury_dates"] else 0

                # Store minutes played and injury status
                heatmap_extended[0, 0, 1] = minutes_played
                heatmap_extended[0, 1, 1] = injury_status

            player_matches_data.append(heatmap_extended)

        while len(player_matches_data) < 38:
            player_matches_data.append(np.zeros((bins[0], bins[1], 2)))  # Padding

        all_players_data.append(np.array(player_matches_data))
        
        # Save the current state of all_players_data after adding each new player
        current_tensor = np.array(all_players_data)
        np.save(save_filename, current_tensor)
        print(f"Saved data for {player_name} to {save_filename}. The current shape is {current_tensor.shape}")

    return np.load(save_filename)

# Assuming player_info is loaded
with open('player_info.json', 'r', encoding='utf-8') as f:
    player_info = json.load(f)

# Example usage
competition_id = 2  # Premier League
season_id = 27     # 2015/16 season
filename = 'all_players_heatmap_tensor.npy'
all_players_heatmap_tensor = process_all_players_and_save(player_info, competition_id, season_id, save_filename=filename)
print(f"Final tensor shape: {all_players_heatmap_tensor.shape}")