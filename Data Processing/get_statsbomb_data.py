import numpy as np
import pandas as pd
import json
from statsbombpy import sb
import warnings
from tqdm import tqdm

def process_player(player_name, player_id, club, injury_dates, injury_duration, competition_id, season_id):
    # Fetch matches for the specified competition and season
    matches = sb.matches(competition_id=competition_id, season_id=season_id)
    
    # Filter matches to those involving the specified club
    club_matches = matches[(matches['home_team'] == club) | (matches['away_team'] == club)]

    # Sort club_matches by match_date in ascending order to ensure chronological processing
    club_matches = club_matches.sort_values(by='match_date')
    
    # Placeholder for match categories data
    matches_data = []

    # Define the categories, including "minutes_played"
    categories = ['Goalkeeper', 'Left Center Back', 'Left Back', 'Left Wing Back', 'Center Back',
                 'Right Back', 'Right Center Back', 'Right Wing Back', 'Left Midfield', 
                 'Left Attacking Midfield', 'Left Defensive Midfield', 'Left Center Midfield',
                 'Center Defensive Midfield', 'Center Midfield', 'Center Attacking Midfield',
                 'Right Center Midfield', 'Right Defensive Midfield', 'Right Attacking Midfield',
                 'Right Midfield', 'Left Wing', 'Left Center Forward', 
                 'Center Forward', 'Right Wing', 'Right Center Forward',
                 
                 "under_pressure", "50_50", "Pass", "Shot", "Dribble", "Pressure", "Duel", 
                 "Interception", "Foul Committed", "Carry",
                 
                 "minutes_played", "injury_duration", "injury"]
    
    # Process each match
    for index, row in tqdm(club_matches.iterrows(), total=club_matches.shape[0], desc=f"Processing Matches for {player_name}"):
        match_id = row['match_id']
        match_date = row['match_date']
        
        # Initialize dictionary for this match's data
        match_categories = dict.fromkeys(categories, 0)  # Initialize categories to 0        
        # Fetch events for the match
        events = sb.events(match_id=match_id)
        
        # Filter events for the specific player
        player_events = events[events['player'] == player_name]

        if player_events.empty and pd.notna(player_id):
            player_events = events[events['player_id'] == player_id]
            
        # Sort events by time (minute, second)
        player_events = player_events.sort_values(by=['minute', 'second'])

        # Track if position has been set
        position_set = False

        if not player_events.empty:
            # Check for player's position from event data
            for _, event in player_events.iterrows():
                if not position_set and 'position' in event and event['position'] is not None:
                    position_name = event['position']
                    if position_name in categories:
                        match_categories[position_name] = 1
                        position_set = True  # Avoid checking position after the first occurrence

                event_type = event['type']
                if event_type in match_categories:
                    match_categories[event_type] += 1
                if event.get('under_pressure') is not None:
                    match_categories["under_pressure"] += 1
                if '50_50' in event:
                    match_categories["50_50"] += 1
            
            # Calculate minutes played
            if pd.isna(player_events.iloc[0].get('substitution_outcome')):
                match_categories["minutes_played"] = player_events.iloc[-1]['minute']
            else:
                match_categories["minutes_played"] = player_events.iloc[-1]['minute'] - player_events.iloc[0]['minute']
        
        # Create a dictionary from injury dates and durations for easy lookup
        injury_dict = dict(zip(injury_dates, injury_duration))
    
        # Check for injury
        if match_date in injury_dict:
            match_categories['injury'] = 1
            match_categories['injury_duration'] = injury_dict[match_date]
        else:
            match_categories['injury'] = 0
            match_categories['injury_duration'] = 0
        
        # Add match data to matches_data
        matches_data.append([match_categories[cat] for cat in categories])
    
    # Ensure data for 38 matches, padding with zeros (or 0 for "minutes_played") if necessary
    while len(matches_data)<38:
        matches_data.append([0] * (len(categories)-1) + [0])  # Use 0 for "minutes_played" padding
    
    return np.array(matches_data[:38])  # Return first 38 matches data as a 2D numpy array


# Suppress the NoAuthWarning from statsbombpy
warnings.filterwarnings("ignore", category=UserWarning, message=".*credentials were not supplied.*")

# Load the player_info.json file
with open('all_player_info_new_dates.json', 'r', encoding='utf-8') as f:
    player_info = json.load(f)

# Initialize a list to collect data for all players
all_players_data = []

# Process each player
for player, info in player_info.items():
    player_data = process_player(player, info["Id"], info["Team"], info["Injury_dates"], info["Injury_duration"], competition_id=2, season_id=27)
    all_players_data.append(player_data)

    # Save the current state of all_players_data to a file
    # Convert the current list of all player data to a 3D numpy array (tensor) before saving
    all_players_tensor = np.array(all_players_data)
    # Define a file name. Here, we're using a generic file name, but you could also include player names or timestamps
    filename = 'positional_all_players_tensor.npy'
    np.save(filename, all_players_tensor)

    print(f"Data for {player} added and saved to {filename}.")


