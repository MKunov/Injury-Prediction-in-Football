#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 16:54:40 2024

@author: felix
"""

import json
from statsbombpy import sb

def get_player_data(competition_id, season_id):
    # Fetch all matches for the specified competition and season
    matches = sb.matches(competition_id=competition_id, season_id=season_id)
    
    # Initialize an empty dictionary to store team data
    team_data = {}

    # List of team names as provided
    team_names = [
        "Leicester City", "AFC Bournemouth", "West Bromwich Albion", "Sunderland",
        "Newcastle United", "Aston Villa", "Everton", "Crystal Palace",
        "Watford", "Arsenal", "Liverpool", "Tottenham Hotspur", "Manchester City",
        "Norwich City", "Chelsea", "Stoke City", "Manchester United", "West Ham United",
        "Swansea City", "Southampton"
    ]

    # Loop through each team name
    for team in team_names:
        # Initialize a dictionary to store player data for the current team
        team_players = {}

        # Filter matches involving the current team
        team_matches = matches[(matches['home_team'] == team) | (matches['away_team'] == team)]
        
        # Process each match
        for index, match in team_matches.iterrows():
            # Retrieve the lineup for the team in this match
            lineup = sb.lineups(match['match_id'])[team]

            # Add each player from the lineup to the team_players dictionary
            for index, player_name in enumerate(lineup['player_name']):
                # Prevent duplicates by checking if the player is already added
                if player_name not in team_players:
                    team_players[player_name] = int(lineup['player_id'][index])

        # Add the team's player dictionary to the main dictionary
        team_data[team] = team_players
        print(f'Finished {team}')

    # Convert the dictionary to JSON format
    with open('team_player_data.json', 'w') as f:
        json.dump(team_data, f, indent=4)

    print("Data has been saved to team_player_data.json")

# Specify the competition ID and season ID
competition_id = 2
season_id = 27

# Call the function to process data
get_player_data(competition_id, season_id)
