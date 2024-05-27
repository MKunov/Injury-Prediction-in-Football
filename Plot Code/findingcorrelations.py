# Example: Plotting
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import pandas as pd
from pandas.plotting import table
import plotly.express as px
from tabulate import tabulate

image_dir = "photos"

hidden_size = 1024

def parse_filename(filename):
    # Updated pattern to optionally include dropout_prob
    pattern = r"F(\d+\.\d+)_model_(\d+)_(\d+)_(\d+)_(\d+\.\d+)_(\d+\.\d+)_(\d+\.\d+)(?:_(\d+\.\d+))?\.png"
    match = re.match(pattern, filename)
    if match:
        # if float(match.group(1)) > 0.2:
        # if int(match.group(3)) == 1024 and int(match.group(4))==1 and float(match.group(5))==0.97 and float(match.group(7))==0.0001:
            return {
                "F1": float(match.group(1)),
                "Batch Size": int(match.group(2)),
                "Hidden Size": int(match.group(3)),
                "Layers": int(match.group(4)),
                "Alpha": float(match.group(5)),
                "Gamma": float(match.group(6)),
                "Learning Rate": float(match.group(7)),
                "Dropout Prob": float(match.group(8)) if match.group(8) else None  # Uncomment if you decide to use dropout_prob later
            }
    else:
        return None

data = []

# Walk through the image directory
for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith(".png"):
            params = parse_filename(file)
            if params:
                params['filename'] = file  # Optional: keep the filename if needed
                data.append(params)

# Convert the list of dictionaries to a DataFrame
df = pd.DataFrame(data)

# print(df['filename'])  # Print the first few rows of the DataFrame
# print(df.describe())  # Get a statistical summary of the DataFrame

# Example: Correlation matrix
# print(df.corr())

# sns.pairplot(df, vars=["F1", "batch_size", "num_layers", "alpha", "gamma", "learning_rate"])
# plt.show()

# Create a parallel coordinates plot
# fig = px.parallel_coordinates(df, 
#                               dimensions=['F1', 'batch_size', 'hidden_size', 'num_layers', 'alpha', 'gamma', 'learning_rate', "dropout_prob"],
#                               color='F1',  # Assuming 'accuracy' is a metric you want to visualize
#                               color_continuous_scale=px.colors.diverging.Tealrose,
#                               labels={"batch_size": "Batch Size",
#                                       "hidden_feature_size": "Hidden Features",
#                                       'num_layers': "Number of Layers",
#                                       "learning_rate": "Learning Rate",
#                                       "alpha": "Alpha",
#                                       "gamma": "Gamma",
#                                       "dropout_probability": "Dropout Prob.",
#                                       "F1": "F1"},
#                               title="Parallel Coordinates Plot for Model Hyperparameters")

# # Show plot
# fig.show()

# Drop 'dropout_probability' from the DataFrame
filtered_df = df.drop('filename', axis=1)


top_performers = filtered_df.sort_values(by='F1', ascending=False).head(11)  # Adjust the number for top N results

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(12, 2))  # Adjust size as necessary
ax.axis('tight')
ax.axis('off')

# Create the table
the_table = table(ax, top_performers, loc='center', cellLoc='center', rowLoc='center')

# Style the table
the_table.auto_set_font_size(False)
the_table.set_fontsize(12)
the_table.scale(1.2, 1.2)  # You can adjust the scale to fit your specific needs

# Save the table as an image
plt.savefig("top_performers.png", dpi=200, bbox_inches='tight')  # You can adjust DPI for resolution

plt.show()  # Display the plot in the notebook or script output