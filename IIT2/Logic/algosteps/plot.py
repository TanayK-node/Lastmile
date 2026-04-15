import matplotlib.pyplot as plt
import pandas as pd

# 1. The data from your sensitivity analysis results
data = {
    "Box Size": [
        "Ultra-Strict\n(Platforms Only)", 
        "Strict\n(+ Rickshaw Stand)", 
        "Optimal\n(+ Bus Depots)", 
        "Loose\n(+ Cafes/Shops)", 
        "Too Large\n(Bleeds out)"
    ],
    "Commuters Captured": [32, 90, 200, 385, 709],
    "Virtual Hubs": [0, 2, 9, 15, 22]
}

df = pd.DataFrame(data)

# 2. Setup the figure and the first axis
fig, ax1 = plt.subplots(figsize=(10, 6))

# 3. Plot Virtual Hubs (The main metric for the 'Elbow')
color1 = '#d62728' # Red
ax1.set_xlabel('Bounding Box Size (Hyperparameter)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Virtual Hubs Generated (Signal)', color=color1, fontsize=12, fontweight='bold')
ax1.plot(df['Box Size'], df['Virtual Hubs'], color=color1, marker='o', linewidth=3, markersize=8, label="Virtual Hubs")
ax1.tick_params(axis='y', labelcolor=color1)
ax1.grid(True, linestyle='--', alpha=0.5)

# 4. Highlight the actual "Elbow" Point mathematically
ax1.annotate('The "Elbow" (Optimal Sweet Spot)', 
             xy=(2, 9),  # Index 2 is "Optimal", y-value is 9
             xytext=(1.2, 14), # Text position
             arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=8),
             fontsize=12, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1))

# 5. Create a secondary y-axis for Commuters Captured
ax2 = ax1.twinx()  
color2 = '#1f77b4' # Blue
ax2.set_ylabel('Commuters Captured (Raw Volume)', color=color2, fontsize=12, fontweight='bold')  
ax2.plot(df['Box Size'], df['Commuters Captured'], color=color2, marker='s', linestyle='dashed', linewidth=2, markersize=8, label="Commuters")
ax2.tick_params(axis='y', labelcolor=color2)

# 6. Formatting and Titles
plt.title('Catchment Sensitivity Analysis: Finding the Elbow Point', fontsize=15, fontweight='bold', pad=15)
fig.tight_layout()  

# 7. Save the plot as a high-res image
output_file = 'elbow_plot_catchment.png'
plt.savefig(output_file, dpi=300)
print(f"Success! High-resolution elbow plot saved as '{output_file}'")