#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 21:01:14 2024

@author: felix
"""

import numpy as np 
total = 1

non_injured = np.load('noninjured_positional_all_players_tensor.npy')
injured = np.load('positional_all_players_tensor.npy')

# Concatenate along the first dimension
concatenated_array = np.concatenate((non_injured, injured), axis=0)
shape = concatenated_array.shape


# Define categories
categories = ['Goalkeeper', 'Left Center Back', 'Left Back', 'Left Wing Back', 'Center Back',
              'Right Back', 'Right Center Back', 'Right Wing Back', 'Left Midfield', 
              'Left Attacking Midfield', 'Left Defensive Midfield', 'Left Center Midfield',
              'Center Defensive Midfield', 'Center Midfield', 'Center Attacking Midfield',
              'Right Center Midfield', 'Right Defensive Midfield', 'Right Attacking Midfield',
              'Right Midfield', 'Left Wing', 'Left Center Forward', 'Center Forward',
              'Right Wing', 'Right Center Forward', "under_pressure", "50_50", "Pass",
              "Shot", "Dribble", "Pressure", "Duel", "Interception", "Foul Committed",
              "Carry", "minutes_played", "injury_duration", "injury"]

if total != 1:
    # Array to store distribution of data for injured and uninjured for each match and category
    stats_data = np.zeros((len(categories), 38, 2), dtype=object)
    
    # Loop through categories
    for i, category in enumerate(categories):
        # Extract index of the category
        category_index = categories.index(category)
        
        # Loop through match days
        for match_day in range(shape[1]):
            injured = []
            uninjured = []
            # Loop players and check if they're injured
            for player in range(shape[0]):
                if concatenated_array[player, match_day, categories.index('minutes_played')] != 0:
                    if concatenated_array[player, match_day, categories.index('injury')] == 0:
                        uninjured.append(concatenated_array[player, match_day, category_index])
                    else:
                        injured.append(concatenated_array[player, match_day, category_index])
            
            # Store the statistics in the structured array
            stats_data[i, match_day, 0] = np.array(uninjured).astype(np.float64)
            stats_data[i, match_day, 1] = np.array(injured).astype(np.float64)
        
    # Save Data
    filename = 'match_histogram_array.npy'
    np.save(filename, stats_data)
    
else:
    # Array to store distribution of data for injured and uninjured for each category
    stats_data = np.zeros((len(categories), 2), dtype=object)
    # Loop through categories
    for i, category in enumerate(categories):
        # Extract index of the category
        category_index = categories.index(category)
        injured = []
        uninjured = []
        
        # Loop through match days
        for match_day in range(shape[1]):
            # Loop players and check if they're injured
            for player in range(shape[0]):
                if concatenated_array[player, match_day, categories.index('minutes_played')] != 0:
                    if concatenated_array[player, match_day, categories.index('injury')] == 0:
                        uninjured.append(concatenated_array[player, match_day, category_index])
                    else:
                        injured.append(concatenated_array[player, match_day, category_index])
                
            
        # Store the statistics in the structured array
        stats_data[i, 0] = np.array(uninjured).astype(np.float64)
        stats_data[i, 1] = np.array(injured).astype(np.float64)

    # Save Data
    filename = 'total_histogram_array.npy'
    np.save(filename, stats_data)

