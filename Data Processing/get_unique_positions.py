#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 11:30:50 2024

@author: felix
"""
from statsbombpy import sb
import pandas as pd

def get_unique_positions(competition_id, season_id):
    matches = sb.matches(competition_id=competition_id, season_id=season_id)
    
    # Initialize an empty set to collect unique position names
    all_unique_positions = set()

    # Iterate through each match
    for index, match in matches.iterrows():
        match_id = match['match_id']
        events = sb.events(match_id=match_id)
        
        # Check if the 'position' column exists and is not empty
        if 'position' in events.columns and not events['position'].isnull().all():
            # Extract positions; assuming 'position' field contains dictionaries
            # Extract the 'name' key from each position dictionary if it exists
            match_positions = events['position']
            # Update the set of unique positions with non-null entries
            all_unique_positions.update(filter(None, match_positions.unique()))
            
            print(all_unique_positions)
    
    return all_unique_positions

# Example usage
competition_id = 2  # Adjust these IDs based on your data availability
season_id = 27
unique_positions = get_unique_positions(competition_id, season_id)
print("Unique positions found in the competition and season:")
print(unique_positions)
