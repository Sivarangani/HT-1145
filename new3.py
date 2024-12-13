import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set font properties
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 23

# Create the data for comparison with very short model names
data = {
    "Model Name": [
        "Chi-Square PSO",
        "CNN + GCAB",
        "Res-BiANet",
        "Proposed"
    ],
    "Training Time (s)": [1500, 1300, 1100, 800],
    "Memory Usage (MB)": [1024, 900, 850, 800],
    "Inference Time (ms)": [100, 80, 60, 30]
}

# Convert the data into a pandas DataFrame
df = pd.DataFrame(data)

# Set up the positions for the bars
bar_width = 0.25
index = np.arange(len(df))

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the bars for each metric
bar1 = ax.barh(index, df['Training Time (s)'], bar_width, label='Training Time (s)', color='skyblue')
bar2 = ax.barh(index + bar_width, df['Memory Usage (MB)'], bar_width, label='Memory Usage (MB)', color='salmon')
bar3 = ax.barh(index + 2*bar_width, df['Inference Time (ms)'], bar_width, label='Inference Time (ms)', color='lightgreen')

# Labeling the plot
ax.set_xlabel('Values')
ax.set_ylabel('Model Name')
ax.set_title('Computational Efficiency')
ax.set_yticks(index + bar_width)
ax.set_yticklabels(df['Model Name'])
ax.legend()

# Adjust layout for better display
plt.tight_layout()

# Show the plot
plt.show()
