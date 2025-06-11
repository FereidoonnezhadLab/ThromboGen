import h5py
import numpy as np
import pyvista as pv
from matplotlib.colors import ListedColormap

# Load the .mat file
with h5py.File('voxel-type-2.mat', 'r') as file:
    data = file['c'][:]
    matrix = np.array(data).transpose()

# Values you want to plot (e.g., just 1, 2, 3 — you can filter this list)
values_to_plot = [1,2,3]

# Create mask for selected values
mask = np.isin(matrix, values_to_plot)

# Get coordinates and values
x, y, z = np.where(mask)
values = matrix[x, y, z]
points = np.column_stack((x, y, z))

# Create the PyVista point cloud
point_cloud = pv.PolyData(points)
point_cloud['values'] = values

# Define custom colormap (red for 1, green for 2, yellow for 3)
color_list = ['red', 'green', 'yellow']
custom_cmap = ListedColormap(color_list)

# Normalize values to 0, 1, 2 (for colormap indexing)
normalized_values = values - 1  # because our colormap starts at 0

# Create plot
plotter = pv.Plotter()
plotter.add_points(
    point_cloud,
    scalars=normalized_values,
    cmap=custom_cmap,
    render_points_as_spheres=True,
    point_size=4,
    clim=[0, 3]  # tells the plotter how to map 0,1,2 to colors
)

# Add legend manually
legend_entries = [
    ['1: RBCs', 'red'],
    ['2: Fibrin', 'green'],
    ['3: Platelets', 'yellow']
]
plotter.add_legend(legend_entries, bcolor='white')

plotter.show()