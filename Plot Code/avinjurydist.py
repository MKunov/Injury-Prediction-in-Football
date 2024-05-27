#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 17:34:19 2024

@author: felix
"""
import numpy as np
import json
import matplotlib.pyplot as plt

player_data = open('all_player_info.json')
player_data = json.load(player_data)

injured = np.load('positional_all_players_tensor.npy')
uninjured = np.load('noninjured_positional_all_players_tensor.npy')
player_stats = np.concatenate((injured, uninjured),axis=0)
player_stats = np.delete(player_stats, -2, axis=2) # delete injury duration
player_stats = np.delete(player_stats, 13, axis=2) # delete cm position (all 0s)

num_players, num_matches, num_features = player_stats.shape

# Initialize dictionaries to keep track of injuries and minutes per team
team_injuries_per_match = {i: {} for i in range(num_matches)}
team_minutes = {}
team_injuries = {}

# Process each player
for i in range(num_players):
    player_name = list(player_data.keys())[i]
    team_name = player_data[player_name]['Team']
    
    # Initialize the team in dictionaries if not already
    if team_name not in team_minutes:
        team_minutes[team_name] = 0
        team_injuries[team_name] = 0
    
    for j in range(num_matches):
        # Check if the player was injured in this match
        injured = player_stats[i, j, -1] == 1
        minutes_played = player_stats[i, j, -2]
        
        # Update team minutes and injuries
        team_minutes[team_name] += minutes_played
        
        if injured:
            team_injuries[team_name] += 1
            if team_name not in team_injuries_per_match[j]:
                team_injuries_per_match[j][team_name] = 0
            team_injuries_per_match[j][team_name] += 1

# Calculate injuries per 1000 hours for each team
# injuries_per_1000_hours = {}
# for team, minutes in team_minutes.items():
#     hours = (minutes / 60)
#     injuries_per_1000_hours[team] = (team_injuries[team] / hours)*1000

# Output results
# print("Injuries per Team per Matchday:")
# for matchday, teams in team_injuries_per_match.items():
#     print(f"Matchday {matchday + 1}:")
#     for team, injuries in teams.items():
#         print(f"  {team}: {injuries} injuries")

# print("\nAverage Injuries per 1000 Hours:")
# for team, injuries_rate in injuries_per_1000_hours.items():
#     print(f"{team}: {injuries_rate:.2f} injuries/match")
    
print("\nTotal Sum of Injuries per Team:")
for team, total_injuries in team_injuries.items():
    print(f"{team}: {round(total_injuries/38,2)} injuries per match")

