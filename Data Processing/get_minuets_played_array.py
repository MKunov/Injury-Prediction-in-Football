#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 15:30:36 2024

@author: felix
"""

import numpy as np
import pandas as pd
from statsbombpy import sb
import json
from tqdm import tqdm

def process_player(player_name, club, player_id, competition_id, season_id):
    # Fetch matches for the specified competition and season
    matches = sb.matches(competition_id=competition_id, season_id=season_id)
    
    # Filter matches to those involving the specified club
    club_matches = matches[(matches['home_team'] == club) | (matches['away_team'] == club)]
    
    # Sort club_matches by match_date in ascending order to ensure chronological processing
    club_matches = club_matches.sort_values(by='match_date')
    
    # Placeholder for minutes played data
    minutes_played = []

    # Process each match
    for index, row in tqdm(club_matches.iterrows(), total=club_matches.shape[0], desc=f"Processing Matches for {player_name}"):
        match_id = row['match_id']
        
        # Fetch events for the match
        events = sb.events(match_id=match_id)
        
        # Filter events for the specific player
        player_events = events[events['player'] == player_name]
        
        # If no events are found using the player's name, try filtering by player's ID
        if player_events.empty and pd.notna(player_id):
            player_events = events[events['player_id'] == player_id]

        if not player_events.empty:
            player_events = player_events.sort_values(by=['minute', 'second'])
            # Calculate minutes played
            if pd.isna(player_events.iloc[0].get('substitution_outcome')):
                minutes = player_events.iloc[-1]['minute']
            else:
                minutes = player_events.iloc[-1]['minute'] - player_events.iloc[0]['minute']
            minutes_played.append(minutes)
        else:
            # No participation in the match
            minutes_played.append(0)

    # Ensure data for 38 matches, padding with zeros if necessary
    while len(minutes_played) < 38:
        minutes_played.append(0)
    
    return np.array(minutes_played)

# Load player info from JSON
with open('all_player_info.json', 'r') as file:
    player_info = json.load(file)

# Create a numpy array to store all player's minutes played data
all_minutes_played = []

for player, info in player_info.items():
    minutes_data = process_player(player, info["Team"], info["Id"], competition_id=2, season_id=27)
    all_minutes_played.append(minutes_data)
    # Convert list to a numpy array
    all_minutes_played_array = np.array(all_minutes_played)

    # Saving the numpy array to file
    np.save('player_minutes_played.npy', all_minutes_played_array)



# If you need to check the shape of the array
print("Shape of all_minutes_played_array:", all_minutes_played_array.shape)
