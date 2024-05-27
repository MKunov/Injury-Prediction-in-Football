#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 10:38:01 2024

@author: felix
"""

import pandas as pd
import json

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('updated_injury_data.csv')

# Step 2: Filter the DataFrame
# Remove all injuries not due to a match
df = df[~df['Team'].str.contains('no matches within 7 days with player participation', na=False)]

# Transform the DataFrame into the desired dictionary structure
player_info = {}

for index, row in df.iterrows():
    name = row['Name']
    name_id = row['Id']
    team = row['Team']
    injury_date = row['Injury_Date_New']
    injury_duration = row['New_Injury_Duration']
    
    if name not in player_info:
        player_info[name] = {
            "Team": team,
            "Id": name_id,
            "Injury_dates": [injury_date],
            "Injury_duration": [injury_duration],
        }
    else:
        player_info[name]["Injury_dates"].append(injury_date)
        player_info[name]["Injury_duration"].append(injury_duration)

# Save the dictionary as a JSON file
with open('all_player_info_new_dates.json', 'w', encoding='utf-8') as f:
    json.dump(player_info, f, ensure_ascii=False, indent=4)

print("The dictionary has been saved as all_player_info_new_dates.json.")
