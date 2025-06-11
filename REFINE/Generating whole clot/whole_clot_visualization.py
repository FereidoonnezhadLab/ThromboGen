import h5py
import numpy as np
import pyvista as pv

# Load the .mat file using h5py
with h5py.File('whole_clot.mat', 'r') as f:
    coords = np.array(f['FinalCoordinates']).T  # (N,3)
    atom_type = np.array(f['atom_type']).T.squeeze()  # (N,)

# Create a PyVista plotter
plotter = pv.Plotter(window_size=[1000, 1000])

# Define colors for each type
color_map = {
    1: 'green',      # Fibrin
    2: 'blue',    # Platelets
    3: 'red'       # RBCs
}

# Define opacity for each atom type
opacity_map = {
    1: 0.5,  # Fibrin
    2: 0.5,  # Platelets
    3: 1   # RBCs
}

# Define different point size for each atom type
point_size_map = {
    1: 2,  # Fibrin very small points
    2: 2,  # Platelets larger
    3: 2   # RBCs larger
}

# Select which atom types to plot
selected_atom_types = [1,2,3]

# Add each selected group separately
for atom in selected_atom_types:
    points = coords[atom_type == atom]
    if points.size > 0:
        cloud = pv.PolyData(points)
        plotter.add_mesh(
            cloud,
            color=color_map.get(atom, 'gray'),
            point_size=point_size_map.get(atom, 1),  # <- point size based on type
            opacity=opacity_map.get(atom, 0.5),
            render_points_as_spheres=False
        )

# Show and Save
plotter.show(auto_close=False)
plotter.screenshot('fibrin_network.png')
plotter.close()
