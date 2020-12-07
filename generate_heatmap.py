# Run in python 3 with matplotlib version 3.3.2
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

# Calculate attention values
# ATT_MULTIPLIER = 0.001
# att_vals = [
#     [ 192.67068,   183.30688,   407.51514,   107.820114,  353.74738, 177.43607,   174.16016 ],
#     [ 544.59485,   494.74646,  1083.9738,    232.24953,   837.9962, 641.4215,    435.9056  ],
#     [ 436.97632,   400.5496,    909.81464,   199.37619,   728.4785, 440.2516,    369.90973 ],
#     [ 858.16077,   755.7981,   1645.7383,    316.84985,  1224.3372, 1089.8124,    641.3263  ]]

ATT_MULTIPLIER = 1
att_vals = [
    [ 0.62153,   0.5549, 0.69456, 1.04123, 0.53569, 0.55287, 0.9502 ],
    [0.51261, 0.57777, 1.03169, 1.01187, 0.44399, 1.11346, 0.76232],
    [0.99259, 1.05616, 0.74374, 0.6382, 0.596, 0.62307, 1.16556],
    [1.17071, 1.13594, 0.6497, 0.94856, 1.02123, 0.632, 0.85519]
]

att_vals = [[round(val * ATT_MULTIPLIER, 2) for val in row] for row in att_vals]

# Create heatmap
fig, ax = plt.subplots()
im = ax.imshow(att_vals)

# Set axis labels
ax.set_xticks(np.arange(len(att_vals[0])))
ax.set_yticks(np.arange(len(att_vals)))
ax.set_xticklabels([i+1 for i in range(len(att_vals[0]))])
ax.set_yticklabels([i+1 for i in range(len(att_vals))])

# Turn spines off and create white grid for better contrast
for edge, spine in ax.spines.items():
    spine.set_visible(False)
ax.set_xticks(np.arange(len(att_vals[0]) + 1) - 0.5, minor=True)
ax.set_yticks(np.arange(len(att_vals) + 1) - 0.5, minor=True)
ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
ax.tick_params(which='minor', bottom=False, left=False)

# Annotate heatmap, use color threshold
# avg = np.mean(att_vals)
# for i in range(len(att_vals)):
#     for j in range(len(att_vals[i])):
#         text_color = ['black', 'white'][int(att_vals[i][j] < avg)]
#         text = ax.text(j, i, att_vals[i][j], ha='center', va='center', color=text_color)

# Set titles
ax.set_title('Actual: Non-novel    Predicted: Non-Novel')
ax.set_xlabel('Source document sentences')
ax.set_ylabel('Target document sentences')

# Add colorbar
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = ax.figure.colorbar(im, cax=cax)

# Save image
plt.savefig('/mnt/f/DMN/heatmaps/heatmap_dmn.png')
