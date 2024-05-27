#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 15:07:03 2024

@author: felix
"""
import numpy as np
import pandas as pd
import json

# Load player data
with open('all_player_info.json', 'r') as file:
    player_info = json.load(file)

# Load match dates
teams_matches = pd.read_csv('teams_matches.csv', parse_dates=True)
for col in teams_matches.columns:
    teams_matches[col] = pd.to_datetime(teams_matches[col], errors='coerce')

# Load players' match stats
player_stats = np.load('player_minutes_played.npy')

# Load players injury data
players_injury = pd.read_csv('old_injury_data.csv', parse_dates=['Injury_Date_New'])

# Update the Injury_Date_New based on the nearest match the player actually played in
for index, row in players_injury.iterrows():
    player_name = row['Name']
    team_name = row['Team']
    injury_date = row['Injury_Date_New']

    try:
        # Find player index and team match dates
        player_index = list(player_info.keys()).index(player_name)
        match_dates = teams_matches[team_name].dropna()

        # Find closest match dates
        closest_indices = np.abs(match_dates - injury_date).argsort()[:2]
        closest_dates = match_dates.iloc[closest_indices]

        # Determine valid matches: within 7 days before or after the injury date
        valid_matches = [date for date in closest_dates if abs((date - injury_date).days) <= 7]
        valid_indices = [idx for idx, date in zip(closest_indices, closest_dates) if abs((date - injury_date).days) <= 7]

        # Check player participation on valid match dates
        minutes_played = [player_stats[player_index, idx] for idx in valid_indices if match_dates.iloc[idx] in valid_matches]

        # Find the latest match date within the valid matches where the player played
        latest_date = None
        for match_date, minutes in zip(valid_matches, minutes_played):
            if minutes > 0:
                if latest_date is None or match_date > latest_date:
                    latest_date = match_date

        # Update Injury_Date_New based on the latest match they actually played in
        if latest_date is not None:
            players_injury.at[index, 'Injury_Date_New'] = latest_date.strftime('%Y-%m-%d')
        else:
            players_injury.at[index, 'Team'] += ' - no valid matches with player participation'

    except ValueError:
        print(f"Player {player_name} not found in player_info. Skipping...")
        continue

# Save the updated DataFrame to a new CSV
players_injury.to_csv('chosen_dates_players_injury.csv', index=False)


