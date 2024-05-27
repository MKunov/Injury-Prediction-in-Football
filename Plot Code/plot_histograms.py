#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 22:45:41 2024

@author: felix
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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


# Select aggregate categories
aggregate_categories = ["under_pressure", "50_50", "Pass", "Shot", "Dribble",
                        "Pressure", "Duel", "Interception", "Foul Committed",
                        "Carry"]
# Change to things per minute
for category in aggregate_categories:
    for injured in range(hist_data.shape[1]):
        hist_data[categories.index(category),injured]/=hist_data[categories.index("minutes_played"),injured]

# Create a subplot with 10 plots
fig, axs = plt.subplots(3, 4, figsize=(16, 12))

# Plot each distribution
for i, ax in enumerate(axs.flatten()):
    if i < 10:
        cat_index = categories.index(aggregate_categories[i])
        
        # Plot histogram for uninjured on the left y-axis
        ax.hist(hist_data[cat_index, 0], alpha=1, color='blue', edgecolor='black', label='Uninjured', bins=30)
        ax.set_ylabel('Uninjured Frequency')
        ax.set_xlabel(f'{categories[cat_index]}s per minute')
        ax.set_title(f'{categories[cat_index]} Distribution')

        # Create a second y-axis for the injured histogram on the right side
        ax_twin = ax.twinx()
        ax_twin.hist(hist_data[cat_index, 1], alpha=0.5, color='orange', edgecolor='black', label='Injured', bins=30)
        ax_twin.set_ylabel('Injured Frequency')

        # Combine legends for both histograms into a single legend box
        lines = [Line2D([0], [0], color='blue', lw=3),
                 Line2D([0], [0], color='orange', lw=3)]
        labels = ['Uninjured', 'Injured']
        ax.legend(lines, labels, loc='upper right')
        
# Set an overall title
fig.suptitle('Distribution of per match aggregate data over season for injured vs uninjured', fontsize=16)

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()







        
