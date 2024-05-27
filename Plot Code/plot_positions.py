#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 11:26:18 2024

@author: felix
"""
import numpy as np
import matplotlib.pyplot as plt

# Load the data
hist_data = np.load('total_histogram_array.npy', allow_pickle=True)

# For reference
categories = ['Goalkeeper', 'Left Center Back', 'Left Back', 'Left Wing Back', 'Center Back',
              'Right Back', 'Right Center Back', 'Right Wing Back', 'Left Midfield', 
              'Left Attacking Midfield', 'Left Defensive Midfield', 'Left Center Midfield',
              'Center Defensive Midfield', 'Center Midfield', 'Center Attacking Midfield',
              'Right Center Midfield', 'Right Defensive Midfield', 'Right Attacking Midfield',
              'Right Midfield', 'Left Wing', 'Left Center Forward', 'Center Forward',
              'Right Wing', 'Right Center Forward', "under_pressure", "50_50", "Pass",
              "Shot", "Dribble", "Pressure", "Duel", "Interception", "Foul Committed",
              "Carry", "minutes_played", "injury_duration", "injury"]

# Select position categories
position_categories = ['Goalkeeper', 'Left Center Back', 'Left Back', 'Left Wing Back', 'Center Back',
              'Right Back', 'Right Center Back', 'Right Wing Back', 'Left Midfield', 
              'Left Attacking Midfield', 'Left Defensive Midfield', 'Left Center Midfield',
              'Center Defensive Midfield', 'Center Midfield', 'Center Attacking Midfield',
              'Right Center Midfield', 'Right Defensive Midfield', 'Right Attacking Midfield',
              'Right Midfield', 'Left Wing', 'Left Center Forward', 'Center Forward',
              'Right Wing', 'Right Center Forward']
# Abbreviate position categories
position_categories_abbr = [''.join(word[0] for word in category.split()) for category in position_categories]


# Create count of each position injured or not
positions_injured = np.zeros((len(position_categories),2))
for category in position_categories:
    for injured in range(hist_data.shape[1]):
        positions_injured[position_categories.index(category),injured] = np.sum(hist_data[categories.index(category),injured]==1)

# Extract the data for the first and second columns
uninjured_data = positions_injured[:, 0]
injured_data = positions_injured[:, 1]
# Replace float 0 in the denominator with a small non-zero value
uninjured_data[uninjured_data == 0] = 1e-10  # Set to a small non-zero value

# Calculate the ratio between the two bar charts
ratios = injured_data / uninjured_data




'''
plotting raw data with ratios
'''

# Generate y-axis values (1 to 24) and x-axis values based on data
y = np.arange(len(position_categories))

# Set the width of the bars
bar_width = 0.4

plt.figure(figsize=(8,6))
# Plot the bars with flipped axes
plt.barh(y - bar_width/2, injured_data, bar_width, color='orange', label='Injured')
plt.barh(y + bar_width/2, uninjured_data, bar_width, color='blue', label='Uninjured')

# Annotate the ratio above each tick on the horizontal line
for i, ratio in enumerate(ratios):
    plt.text(800, y[i], f'{ratio:.2f}', ha='left', va='center', fontsize=10, color='red')

# Add labels, title, and legend
plt.ylabel('Positions')
plt.xlabel('Frequency')
plt.title('Total Injured and Uninjured (during match) Players for Each Position Over Season\nwith Ratios Overlayed')
plt.yticks(y, position_categories_abbr)  # Set y-axis labels with abbreviations
plt.legend()

# Show the plot
plt.show()




