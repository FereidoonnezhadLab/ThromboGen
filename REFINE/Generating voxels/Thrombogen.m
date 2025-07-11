% ==========================================================
% Fibrin Network Simulation Script
% ==========================================================
function [porosity, composition, ClotMatrix] = Thrombogen(counter)
% ==========================================================
% Section 0: Initialization
% ==========================================================
seed = 2000 + counter;
go_forward = false;

rng(seed); % Set random seed for reproducibility
% Define physical and simulation parameters
numSpheres = randi(500);
max_diameter_factor = rand();
clot_volume = 5e7;
Window_size = 100; % Simulation domain size (µm)
Window_size_crop = 72; % Cropping size (µm)
Croped_clot_dim = 144; % Matrix resolution per dimension
Resolution = Window_size_crop / Croped_clot_dim; % µm per voxel
platelet_ratio = rand();
fibrin_concentration = 0.1+2.5*rand(); % g/L
voxel_volume = Window_size_crop^3;
density_threshold = 0.001;
platelet_radius = 1.5; %(um)
rbc_filling_factor = 0.5+0.5*rand();
angle_balancing_weight = 1;

% Interpolation constants for physical properties
C1 = 0.4; L1 = 4.87; Z1 = 3.19; v1 = 0.339;
C2 = 1.6; L2 = 2.99; Z2 = 3.33; v2 = 0.341;

% Directional preference setup
preferred_direction = [1, 1, 1] / norm([1, 1, 1]);
direction_weight = 0; % Set to >0 to add directional alignment

% ==========================================================
% Section 1: Generate Inclusions
% ==========================================================
% Generate non-overlapping spherical inclusions within domain
max_dimension = Window_size;
max_diameter = max_dimension * max_diameter_factor;
sphere_centers = [];
sphere_radii = [];
Points = [];
max_steps = 200; % Max number of tries to create an sphere
% Generate spheres iteratively, ensuring no overlap
for i = 1:numSpheres
    counter = 0;
    stop_condition = 0;
    while true
        counter = counter+1;
        % Calculate the average center of existing spheres, if any
        if isempty(sphere_centers)
            avg_center = [0, 0, 0]; % Start at the origin if no spheres exist
        else
            avg_center = mean(sphere_centers, 1);
        end

        % Generate random offset for sphere placement, relative to the average center
        distance_factor = rand(); % A random factor to control distance from the average center
        center_offset = (2 * rand(1, 3) - 1) * (1 - distance_factor) * (Window_size / 2);
        center = avg_center + center_offset;
        radius = rand()*max_diameter/2;
        % Check if this sphere intersects any previously added spheres
        intersects = false;
        for j = 1:length(sphere_radii)
            distance = norm(center - sphere_centers(j, :));
            if distance < (radius + sphere_radii(j))
                intersects = true;
                break;
            end
        end

        % If no intersection, add the sphere to the list
        if ~intersects
            sphere_centers = [sphere_centers; center]; %#ok<*AGROW>
            sphere_radii = [sphere_radii; radius];
            break;
        end
        if counter > max_steps
            numSpheres = i-1;
            stop_condition = 1;
            break;
        end
    end
    if stop_condition
        break;
    end
end

% ==========================================================
% Section 2: Effective Volume Calculation
% ==========================================================
% Estimate total clot volume and effective fibrin volume
clot_volume_um3 = max((4/3)*pi*Window_size_crop^3,calculate_clot_volume(sphere_centers, sphere_radii));
total_inclusion_volume_um3 = sum((4 / 3) * pi * (sphere_radii.^3));
% Calculate effective fibrin volume (excluding inclusions)
effective_fibrin_volume_um3 = 1.5*(clot_volume_um3 - total_inclusion_volume_um3);
% Convert fibrin concentration from g/L to g/um^3 and estimate fibrin volume
fibrin_concentration_um3 = fibrin_concentration * 1e-15; % Convert to g/um^3
fibrin_density = 1.395 * 1e-12; % Assume fibrin density in g/um^3
fibrin_volume_um3 = (fibrin_concentration_um3 / fibrin_density) * effective_fibrin_volume_um3;

% ==========================================================
% Section 3: Estimate Bond Length and Node Count
% ==========================================================
% Estimate number of points from volume and mean bond length
fibrin_radius = 0.15 / 2; % um
% Total bond length required
total_bond_length = fibrin_volume_um3 / (pi * fibrin_radius^2);
% Calculate target mean bond length and estimate the number of required nodes
target_mean_length = interp1([C1, C2], [L1, L2], fibrin_concentration, 'linear', 'extrap');
numPoints = ceil(total_bond_length / target_mean_length);
if numPoints < 1000
    go_forward = false;
end

% ==========================================================
% Section 4: Generate Clustered Points
% ==========================================================
% Generate point clouds around each inclusion with KDE
for i = 1:numSpheres
    cluster_center = sphere_centers(i, :);
    radius = sphere_radii(i);
    num_cluster_points = round(numPoints / numSpheres);
    count = 0;
    cluster_points = [];
    bandwidth_std = eye(3) * (clot_volume_um3 / numSpheres) / 100;  % Ensure connectivity
    max_attempts  = 5000;          % <- new safety valve
    attempts      = 0;

    while count < num_cluster_points && attempts < max_attempts
        attempts = attempts + 1;

        % 1. draw a candidate batch
        new_points = mvnrnd(cluster_center, bandwidth_std, num_cluster_points);

        % 2. keep only points outside *all* spheres
        distances       = vecnorm(new_points - cluster_center, 2, 2);
        valid_points    = new_points(distances > radius, :);

        if ~isempty(sphere_centers)
            D = pdist2(valid_points, sphere_centers);
            inside_any  = any(D < sphere_radii', 2);
            valid_points(inside_any, :) = [];
        end

        % 3. append as many as we still need
        n_new   = min(num_cluster_points - count, size(valid_points,1));
        if n_new > 0
            cluster_points = [cluster_points; valid_points(1:n_new , :)];
            count          = count + n_new;
        end

        % 4. if we are making no progress, widen the search
        if n_new == 0 && mod(attempts,50)==0
            bandwidth_std = bandwidth_std * 1.5;   % expand kernel
        end
    end

    % fall-back: if we still lack points, relax the demand
    if count < num_cluster_points
        warning('Could not place the full %d points for sphere %d (got %d).', ...
            num_cluster_points, i, count);
    end


    % Append to overall points list
    Points = [Points; cluster_points];
end

% ==========================================================
% Section 5: Bond Generation
% ==========================================================
% Create bonds between nodes following valency distribution
target_valency = interp1([C1, C2], [Z1, Z2], fibrin_concentration, 'linear', 'extrap');
[bonds, ~] = generate_bonds_with_valency(Points, target_valency, preferred_direction,direction_weight);

% Calculate the number of connections for each point
numPoints = size(Points, 1); % Update to reflect the current number of points
all_connections = bonds(:);  % Combines both columns of bonds into one long list
connection_counts = accumarray(all_connections, 1, [numPoints, 1]);

% ==========================================================
% Section 6: Network Squeezing
% ==========================================================
% Displace nodes near inclusions to simulate compression
displacement_total = zeros(size(Points));

% Compute distances between all points and sphere centers
D = pdist2(Points, sphere_centers);  % Size: [num_points, num_spheres]

for j = 1:length(sphere_radii)
    r_influence = sphere_radii(j) * 1.2;

    % Find points within the influence zone of the j-th sphere
    affected = D(:, j) < r_influence;

    if any(affected)
        % Compute direction vectors (normalized) from sphere center to points
        directions = Points(affected, :) - sphere_centers(j, :);
        dists = sqrt(sum(directions.^2, 2));
        unit_dirs = directions ./ dists;

        % Displacement magnitudes
        disp_mags = (r_influence - dists);

        % Final displacements
        displacements = unit_dirs .* disp_mags;

        % Accumulate displacements
        displacement_total(affected, :) = displacement_total(affected, :) + displacements;
    end
end
% Apply total displacement
Points = Points + displacement_total;

% ==========================================================
% Section 7: Relaxation and Optimization
% ==========================================================
% Apply relaxation forces to optimize bond distribution
max_iterations = 2000;
energy_threshold = 0.07;
k_fibrin = 0.1;
initial_step_size = 0.0001;
step_size = initial_step_size;
max_step_size = 0.05;
min_step_size = 1e-5;

% New parameters for convergence stability
kl_stable_threshold = 1e-2;  % Threshold for considering J-S Divergence as stable
kl_stable_max_count = 5;    % Number of successive iterations with small changes to consider convergence stable

% Initialize previous J-S Divergence for comparison in the iteration loop
prev_Jensen_Shannon_Divergence = Inf; % Start with a high initial value

% Pre-calculate reference bond length distribution for comparison
nu = interp1([C1, C2], [v1, v2], fibrin_concentration, 'linear', 'extrap');
L = target_mean_length; % Mean length
% Calculate s^2 from nu
s_squared = nu * L^2;
% Calculate zeta^2
zeta_squared = log((s_squared / L^2) + 1);
% Calculate lambda
lambda = log(L) - (zeta_squared / 2);
% Define the range of x values for the distribution
bond_lengths = compute_bond_lengths(bonds, Points);
x_values = linspace(1e-10, max(bond_lengths), length(bond_lengths)); % Avoid zero for stability
% Calculate the log-normal PDF based on the paper formula
log_normal_pdf = (1 ./ (x_values * sqrt(2 * pi * zeta_squared))) .* ...
    exp(-((lambda - log(x_values)).^2) / (2 * zeta_squared));

% ==========================================================
% Modified Relaxation Process to Include Directional Force
% ==========================================================
for iteration = 1:max_iterations
    % Compute current bond lengths
    bond_lengths = compute_bond_lengths(bonds, Points);

    % Fit Log-normal distributions and compute PDFs
    pd1 = fitdist(bond_lengths, 'Lognormal');
    dx = mean(diff(x_values));
    pdf1 = pdf(pd1, x_values);

    % Normalize PDFs
    pdf1_normalized = pdf1 / sum(pdf1 * dx);
    log_normal_pdf_normalized = log_normal_pdf / sum(log_normal_pdf * dx);

    % Calculate average distribution
    average_pdf = 0.5 * (pdf1_normalized + log_normal_pdf_normalized);

    % Add epsilon to avoid log(0)
    epsilon = 1e-10;
    pdf1_normalized = pdf1_normalized + epsilon;
    log_normal_pdf_normalized = log_normal_pdf_normalized + epsilon;
    average_pdf = average_pdf + epsilon;

    % Compute J-S Divergence
    KL_P1_avg = sum(pdf1_normalized .* log(pdf1_normalized ./ average_pdf)) * dx;
    KL_P2_avg = sum(log_normal_pdf_normalized .* log(log_normal_pdf_normalized ./ average_pdf)) * dx;
    Jensen_Shannon_Divergence = 0.5 * (KL_P1_avg + KL_P2_avg);

    if isnan(Jensen_Shannon_Divergence)
        %disp('NaN detected in Jensen-Shannon Divergence. Stopping iteration.');
        break;
    end

    %disp(['Jensen-Shannon Divergence: ', num2str(Jensen_Shannon_Divergence)]);

    % Compute forces considering the current bond length distribution and move points
    forces = compute_forces_log_normal(bonds, bond_lengths, lambda, zeta_squared, k_fibrin, Points);

    % Add directional alignment force
    directional_forces = zeros(size(forces));
    for i = 1:size(bonds, 1)
        p1 = bonds(i, 1);
        p2 = bonds(i, 2);
        bond_vector = Points(p2, :) - Points(p1, :);
        bond_vector_normalized = bond_vector / norm(bond_vector);

        % Calculate directional force as a projection onto the preferred direction
        alignment_component = dot(bond_vector_normalized, preferred_direction);
        alignment_force = direction_weight * alignment_component * preferred_direction;

        % Apply directional force
        directional_forces(p1, :) = directional_forces(p1, :) + alignment_force;
        directional_forces(p2, :) = directional_forces(p2, :) - alignment_force;
    end

    % Balancing force for angles
    angle_forces = zeros(size(forces));

    % For each point, try to equalize angles between bonds
    parfor i = 1:size(Points, 1)
        connected_bonds = find(bonds(:,1) == i | bonds(:,2) == i);
        if numel(connected_bonds) < 2
            continue; % Not enough bonds to calculate angles
        end

        vectors = zeros(numel(connected_bonds), 3);
        for j = 1:numel(connected_bonds)
            b = bonds(connected_bonds(j), :);
            other = b(b ~= i);
            vectors(j, :) = Points(other, :) - Points(i, :);
            vectors(j, :) = vectors(j, :) / norm(vectors(j, :));
        end

        % Compute pairwise angles
        for j = 1:size(vectors, 1)-1
            for k = j+1:size(vectors,1)
                v1 = vectors(j, :);
                v2 = vectors(k, :);
                angle = acos(dot(v1, v2));
                ideal_angle = 2*pi / size(vectors, 1);
                angle_diff = angle - ideal_angle;
                torque = cross(v1, v2);
                if norm(torque) > 0
                    torque_dir = torque / norm(torque);
                    % Small corrective force
                    correction = angle_balancing_weight * angle_diff * torque_dir;
                    angle_forces(i, :) = angle_forces(i, :) + correction;
                end
            end
        end
    end


    % Total force = original force + directional force + angle_force
    total_forces = forces + directional_forces + angle_forces;

    % Move points based on computed total forces
    Points = move_points(Points, total_forces, step_size);

    % Adjust step size and k_fibrin
    if iteration > 1 && Jensen_Shannon_Divergence >= prev_Jensen_Shannon_Divergence
        step_size = max(step_size * 0.8, min_step_size);
        k_fibrin = min(k_fibrin * 2, 20); % Increase k_fibrin significantly for more exploration
        %disp(['Step size reduced to ', num2str(step_size), ', k_fibrin increased to ', num2str(k_fibrin), ' at iteration ', num2str(iteration)]);
    else
        step_size = min(step_size * 1.1, max_step_size);
        k_fibrin = max(k_fibrin * 0.9, 1); % Gradually reduce k_fibrin if improving
    end

    % Check for convergence and update kl_stable_count
    if abs(Jensen_Shannon_Divergence - prev_Jensen_Shannon_Divergence) < kl_stable_threshold
        kl_stable_count = kl_stable_count + 1;
    else
        kl_stable_count = 0;
    end

    % Apply random perturbation if the system is stable for too many iterations
    if kl_stable_count > kl_stable_max_count
        % Apply a small random perturbation to all points
        perturbation_strength = 0.05 * target_mean_length; % Relative to target mean length
        Points = Points + perturbation_strength * (rand(size(Points)) - 0.5);
        %disp(['Applied random perturbations to points at iteration ', num2str(iteration)]);
        kl_stable_count = 0; % Reset the stable count
    end


    % Update Jensen-Shannon Divergence for the next iteration
    prev_Jensen_Shannon_Divergence = Jensen_Shannon_Divergence;

    % Check if the system has converged
    if Jensen_Shannon_Divergence < energy_threshold
        %disp(['Converged at iteration ', num2str(iteration)]);
        break;
    end
end

% ==========================================================
% Section 8: Cleanup - Remove Long Bonds
% ==========================================================
% Remove nodes with excessively long connections
max_allowed_length = 5 * target_mean_length; % Maximum allowed bond length (adjust as needed)

% Identify and remove nodes with many super-long bonds
long_bond_threshold = max_allowed_length;
to_remove = false(numPoints, 1); % Logical array to mark nodes for removal

% Loop over each node to identify problematic ones
for i = 1:numPoints
    % Find bonds connected to the current node
    connected_bonds = bonds(:, 1) == i | bonds(:, 2) == i;
    connected_lengths = bond_lengths(connected_bonds);

    % Check if there are multiple super-long bonds connected to this node
    if sum(connected_lengths > long_bond_threshold) > 0
        % Mark the node for removal
        to_remove(i) = true;
    end
end

% Remove marked nodes from Points and update bonds
Points(to_remove, :) = [];
remaining_indices = find(~to_remove);

% Update bonds to reflect the removal of nodes
new_bonds = [];
for i = 1:size(bonds, 1)
    % Get the bond nodes and check if both are still valid
    node1 = bonds(i, 1);
    node2 = bonds(i, 2);
    if ~to_remove(node1) && ~to_remove(node2)
        % Map old indices to new indices after removal
        new_node1 = find(remaining_indices == node1);
        new_node2 = find(remaining_indices == node2);
        new_bonds = [new_bonds; new_node1, new_node2];
    end
end

% Update bonds and recalculate bond lengths
bonds = new_bonds;
bond_lengths = compute_bond_lengths(bonds, Points);

% ==========================================================
% Section 9: Retain Largest Connected Component
% ==========================================================
% Keep only the largest subgraph of the network
numPoints = size(Points, 1);
adjacency_list = cell(numPoints, 1);
for i = 1:size(bonds, 1)
    adjacency_list{bonds(i, 1)} = [adjacency_list{bonds(i, 1)}, bonds(i, 2)];
    adjacency_list{bonds(i, 2)} = [adjacency_list{bonds(i, 2)}, bonds(i, 1)];
end

% Find connected components using BFS
visited = false(numPoints, 1);
components = {};
for i = 1:numPoints
    if ~visited(i)
        % Start a new BFS from the unvisited node
        queue = i;
        component = [];
        while ~isempty(queue)
            current = queue(1);
            queue(1) = [];
            if ~visited(current)
                visited(current) = true;
                component = [component, current];
                queue = [queue, adjacency_list{current}]; % Add all neighbors to the queue
            end
        end
        components{end + 1} = component; % Store the connected component
    end
end

% Find the largest connected component
component_sizes = cellfun(@length, components);
[~, largest_component_idx] = max(component_sizes);
largest_component = components{largest_component_idx};

% Keep only the points and bonds that are part of the largest connected component
Points = Points(largest_component, :);

% Update the bonds to only include bonds within the largest connected component
% Create a mapping from old indices to new indices
new_index_map = zeros(numPoints, 1);
new_index_map(largest_component) = 1:length(largest_component);

% Update the bonds array
bonds = bonds(ismember(bonds(:, 1), largest_component) & ismember(bonds(:, 2), largest_component), :);
bonds = [new_index_map(bonds(:, 1)), new_index_map(bonds(:, 2))];

% ==========================================================
% Section 10: Adjust Bonds Inside Spheres
% ==========================================================
% Shift bonds out of sphere interiors for realism
bonds = move_bonds_outside_spheres(bonds, Points, sphere_centers, sphere_radii);

% Calculate the number of connections for each point
numPoints = size(Points, 1); % Update to reflect the current number of points
connection_counts = zeros(numPoints, 1); % Initialize connection countsg
for i = 1:size(bonds, 1)
    connection_counts(bonds(i, 1)) = connection_counts(bonds(i, 1)) + 1;
    connection_counts(bonds(i, 2)) = connection_counts(bonds(i, 2)) + 1;
end

% ==========================================================
% Section 11: 3D Matrix Setup
% ==========================================================
% Prepare empty voxel matrix and coordinate mapping
Window_center = mean(Points,1);

% Define the bounds of the cropping window
crop_min = Window_center-Window_size_crop/2; % Minimum bound in um
crop_max = Window_center+Window_size_crop/2;  % Maximum bound in um

% Initialize the 3D matrix
ClotMatrix = single(zeros(Croped_clot_dim, Croped_clot_dim, Croped_clot_dim));

% Function to map coordinates into the matrix indices
map_to_matrix = @(coords) round((coords - crop_min) / Resolution) + 1;

% ==========================================================
% Section 12: Insert Fibrin Bonds
% ==========================================================
% Use Bezier curves to draw and map fibrin fibers
[ClotMatrix, bond_length_total] = draw_fibrin_bonds(bonds, Points, crop_min, crop_max, Croped_clot_dim, map_to_matrix);
% ==========================================================
% Section 13: Insert RBCs
% ==========================================================
[ClotMatrix, rbc_points_all,rbc_indices] = place_rbc_in_spheres(ClotMatrix, sphere_centers, sphere_radii, Croped_clot_dim, map_to_matrix,rbc_filling_factor);
% ==========================================================
% Section 14: Insert Platelets
% ==========================================================
[ClotMatrix, num_platelets_output, platelet_centers] = insert_platelets(ClotMatrix, Points, connection_counts, platelet_radius, platelet_ratio, crop_min, crop_max, Croped_clot_dim, map_to_matrix, Resolution);
% ==========================================================
% Section 15: Compute Porosity and Composition
% ==========================================================
platelet_diamter = 2 * platelet_radius / (Window_size_crop / Croped_clot_dim);
Fibrin_volume = bond_length_total * pi * (fibrin_radius)^2;
Platelet_volume = num_platelets_output * (4/3) * pi * (platelet_diamter / 2)^3;
RBC_volume = nnz(ClotMatrix(:) == 1);
composition = RBC_volume / (RBC_volume + Fibrin_volume + Platelet_volume);
porosity = 1 - ((RBC_volume + Fibrin_volume + Platelet_volume) / Croped_clot_dim^3);

% % % Section 13: Export clot geometry for Blender
% % % Bezier fibrin strands, RBC alpha surfaces, platelet spheres
% % % ==========================================================
% % % Crop points to export only within crop_min and crop_max ---
in_crop_Points = all(Points >= crop_min & Points <= crop_max, 2);
Points_cropped = Points(in_crop_Points, :);
index_map = zeros(size(Points,1),1);
index_map(in_crop_Points) = 1:nnz(in_crop_Points);

% Filter bonds to retain only those with both nodes inside cropped region
valid_bonds = all(in_crop_Points(bonds), 2);
bonds_cropped = [index_map(bonds(valid_bonds,1)), index_map(bonds(valid_bonds,2))];

% Crop RBC points
rbc_mask = all(rbc_points_all >= crop_min & rbc_points_all <= crop_max, 2);
rbc_points_cropped = rbc_points_all(rbc_mask, :);

% Crop platelet centers
platelet_mask = all(platelet_centers >= crop_min & platelet_centers <= crop_max, 2);
platelet_centers_cropped = platelet_centers(platelet_mask, :);

% Recenter cropped objects
all_coords = [Points_cropped; rbc_points_cropped; platelet_centers_cropped];
center_shift = mean([crop_min; crop_max]);
Points_cropped = Points_cropped - center_shift;
rbc_points_cropped = rbc_points_cropped - center_shift;
platelet_centers_cropped = platelet_centers_cropped - center_shift;


% --- Prepare RBC clusters ---
rbc_clusters = cell(size(rbc_indices));
valid_mask = all(rbc_points_all >= crop_min & rbc_points_all <= crop_max, 2);
cropped_idx_map = find(valid_mask); % maps index in cropped to full

for i = 1:length(rbc_indices)
    cluster_indices = rbc_indices{i};
    % Keep only the points that were included in the cropped RBC list
    valid_in_cluster = ismember(cluster_indices, cropped_idx_map);
    if any(valid_in_cluster)
        actual_indices = cluster_indices(valid_in_cluster);
        rbc_clusters{i} = rbc_points_all(actual_indices, :) - center_shift;
    else
        rbc_clusters{i} = []; % Skip completely excluded clusters
    end
end


% --- Parallelize Fibrin Bezier tubes ---
tube_radius = 0.1;
n_segments = 8;
n_samples = 20;
num_bonds = size(bonds_cropped, 1);  % Use cropped bonds

tube_vertices = cell(num_bonds, 1);
tube_faces    = cell(num_bonds, 1);
tube_colors   = cell(num_bonds, 1);

parfor i = 1:num_bonds
    p0 = Points_cropped(bonds_cropped(i,1), :);
    p2 = Points_cropped(bonds_cropped(i,2), :);
    control = (p0 + p2)/2 + (rand(1,3)-0.5)*1;
    t = linspace(0,1,n_samples)';
    curve = (1 - t).^2 .* p0 + 2*(1 - t).*t .* control + t.^2 .* p2;
    [v, f] = tubeAlongCurve(curve, tube_radius, n_segments);
    tube_vertices{i} = v;
    tube_faces{i}    = f;
    tube_colors{i}   = repmat([255 255 255], size(v,1), 1);
end

% --- Parallelize RBC alpha surfaces ---
rbc_vertices = cell(length(rbc_clusters), 1);
rbc_faces    = cell(length(rbc_clusters), 1);
rbc_colors   = cell(length(rbc_clusters), 1);

parfor i = 1:length(rbc_clusters)
    pts = rbc_clusters{i};
    if size(pts,1) < 4
        continue;
    end
    try
        shp = alphaShape(pts, 1.5);
        [f, v] = boundaryFacets(shp);
    catch
        [~, f, v] = convhull(pts);
    end
    rbc_vertices{i} = v;
    rbc_faces{i}    = f;
    rbc_colors{i}   = repmat([255 0 0], size(v,1), 1);
end

% --- Parallelize Platelet spheres ---
[sp_v, sp_f] = icosphere(2);
sp_v = sp_v * 0.75;
num_platelets = size(platelet_centers_cropped, 1);

platelet_vertices = cell(num_platelets, 1);
platelet_faces    = cell(num_platelets, 1);
platelet_colors   = cell(num_platelets, 1);

parfor i = 1:num_platelets
    v = sp_v + platelet_centers_cropped(i, :);  % <-- use _cropped version
    f = sp_f;
    platelet_vertices{i} = v;
    platelet_faces{i}    = f;
    platelet_colors{i}   = repmat([0 0 255], size(v,1), 1);
end

% --- Merge all meshes together efficiently ---

% Fibrin
all_v = vertcat(tube_vertices{:});
offsets = cumsum([0; cellfun(@(v) size(v,1), tube_vertices(1:end-1))]);
all_f = cell2mat(cellfun(@(f, o) f + o, tube_faces, num2cell(offsets), 'UniformOutput', false));
all_c = vertcat(tube_colors{:});

% RBCs
rbc_v = vertcat(rbc_vertices{:});
offsets = cumsum([0; cellfun(@(v) size(v,1), rbc_vertices(1:end-1))]);
rbc_f = cell2mat(cellfun(@(f, o) f + o, rbc_faces, num2cell(offsets), 'UniformOutput', false));
rbc_c = vertcat(rbc_colors{:});

% Platelets
pl_v = vertcat(platelet_vertices{:});
offsets = cumsum([0; cellfun(@(v) size(v,1), platelet_vertices(1:end-1))]);
pl_f = cell2mat(cellfun(@(f, o) f + o, platelet_faces, num2cell(offsets), 'UniformOutput', false));
pl_c = vertcat(platelet_colors{:});

% Final mesh
fv_all.vertices = [all_v; rbc_v; pl_v];
fv_all.faces    = [all_f; rbc_f + size(all_v,1); pl_f + size(all_v,1) + size(rbc_v,1)];
fv_all.colors   = [all_c; rbc_c; pl_c];

% --- Write to PLY ---
ply_filename = sprintf('clot_cropped_%s.ply', datestr(now,'yyyymmdd_HHMM')); %#ok<TNOW1>
writePLYMesh(fv_all.vertices, fv_all.faces, fv_all.colors, ply_filename);
fprintf('✅ Cropped Blender mesh written to %s\n', ply_filename);

end

% ==========================================================
% Helper Function: generate_bonds_with_valency
% ==========================================================
function [bonds, bond_lengths] = generate_bonds_with_valency(Points, target_valency, preferred_direction, direction_weight)
numPoints = size(Points, 1);
bond_buffer = zeros(numPoints * 10, 2); % Preallocate
bond_idx = 1;
connection_counts = zeros(numPoints, 1);
target_connections = round(geornd(1 / (target_valency - 2), numPoints, 1) + target_valency);

for i = 1:numPoints
    vectors = Points - Points(i, :);
    distances = sqrt(sum(vectors.^2, 2));
    direction_scores = vectors * preferred_direction';
    weighted_distances = distances - direction_weight * direction_scores;

    [~, idx] = sort(weighted_distances);
    nearest_neighbors = idx(2:50);

    for j = 1:length(nearest_neighbors)
        ni = nearest_neighbors(j);
        if connection_counts(i) < target_connections(i) && connection_counts(ni) < target_connections(ni)
            bond_buffer(bond_idx, :) = [i, ni];
            bond_idx = bond_idx + 1;
            connection_counts(i) = connection_counts(i) + 1;
            connection_counts(ni) = connection_counts(ni) + 1;
        end
        if connection_counts(i) >= target_connections(i)
            break;
        end
    end
end

bonds = bond_buffer(1:bond_idx-1, :);
bond_lengths = compute_bond_lengths(bonds, Points);
end

% ==========================================================
% Helper Function: compute_forces_log_normal
% ==========================================================
function forces = compute_forces_log_normal(bonds, bond_lengths, lambda, zeta_squared, k_fibrin, Points)
numPoints = size(Points, 1);
forces = zeros(numPoints, 3);
target_length = exp(lambda + zeta_squared / 2);
target_adjustment = 0.5 * mean(bond_lengths);

for i = 1:size(bonds, 1)
    p1 = bonds(i, 1);
    p2 = bonds(i, 2);
    vec = Points(p2, :) - Points(p1, :);
    bond_len = bond_lengths(i);
    adjusted_target = 0.5 * (target_length + target_adjustment);
    force_mag = -k_fibrin * (bond_len - adjusted_target);
    force_vec = (force_mag / bond_len) * vec;

    forces(p1, :) = forces(p1, :) - force_vec;
    forces(p2, :) = forces(p2, :) + force_vec;
end
end

% ==========================================================
% Helper Function: compute_bond_lengths
% ==========================================================
function bond_lengths = compute_bond_lengths(bonds, Points)
p1 = Points(bonds(:,1), :);
p2 = Points(bonds(:,2), :);
bond_lengths = sqrt(sum((p1 - p2).^2, 2)) + 1e-6; % Add epsilon for stability
end

% ==========================================================
% Helper Function: move_points
% ==========================================================
function Points = move_points(Points, forces, step_size)
Points = Points + step_size * forces;
end

% ==========================================================
% Helper Function: move_bonds_outside_spheres
% ==========================================================
function bonds = move_bonds_outside_spheres(bonds, Points, sphere_centers, sphere_radii)
midpoints = (Points(bonds(:,1), :) + Points(bonds(:,2), :)) / 2;
D = pdist2(midpoints, sphere_centers);

for j = 1:length(sphere_radii)
    inside = D(:, j) < sphere_radii(j);
    if any(inside)
        mids = midpoints(inside, :);
        center = sphere_centers(j, :);
        dir = mids - center;
        normed = sqrt(sum(dir.^2, 2));
        unit_dir = dir ./ normed;
        new_mids = center + unit_dir * sphere_radii(j) * 1.1;

        for k = 1:sum(inside)
            i = find(inside);
            p1 = bonds(i(k), 1);
            p2 = bonds(i(k), 2);
            Points(p1,:) = (Points(p1,:) + new_mids(k,:)) / 2;
            Points(p2,:) = (Points(p2,:) + new_mids(k,:)) / 2;
        end
    end
end
end

% ==========================================================
% Helper Function: calculate_clot_volume
% ==========================================================
function total_volume = calculate_clot_volume(sphere_centers, sphere_radii)
total_points = 1000;
if size(sphere_centers, 2) ~= 3 || size(sphere_centers, 1) ~= size(sphere_radii, 1)
    error('Mismatch in center and radius dimensions');
end

surface_areas = 4 * pi * (sphere_radii.^2);
points_per_sphere = round((surface_areas / sum(surface_areas)) * total_points);
all_points = [];

for i = 1:length(sphere_radii)
    theta = 2 * pi * rand(points_per_sphere(i), 1);
    phi = acos(2 * rand(points_per_sphere(i), 1) - 1);
    r = sphere_radii(i);
    x = r * sin(phi) .* cos(theta) + sphere_centers(i,1);
    y = r * sin(phi) .* sin(theta) + sphere_centers(i,2);
    z = r * cos(phi) + sphere_centers(i,3);
    all_points = [all_points; x y z];
end

dt = delaunayTriangulation(all_points);
[~, volume] = convexHull(dt);
total_volume = volume * 1.1;
end
% ==========================================================
% Helper Function: place_rbc_in_spheres
% ==========================================================
function [ClotMatrix, rbc_points_all, rbc_indices] = place_rbc_in_spheres(ClotMatrix, sphere_centers, sphere_radii, Croped_clot_dim, map_to_matrix, rbc_filling_factor)

% Constants
RBC_diameter_noncompacted = 8; % µm
voxel_spacing = 0.5;
Resolution = 72 / Croped_clot_dim;

% Output initialization
rbc_points_all = [];
rbc_indices = {};

num_restarts = 5;        % Number of restart attempts per sphere
max_steps = 100;         % Optimization steps per restart

for s = 1:length(sphere_radii)
    radius_sphere = sphere_radii(s);
    center_sphere = sphere_centers(s, :);

    % Scale RBCs based on sphere size
    scaling = 0.6 * (radius_sphere < 2 * RBC_diameter_noncompacted) + ...
              0.9 * (radius_sphere >= 2 * RBC_diameter_noncompacted);
    rbc_d = RBC_diameter_noncompacted * scaling;
    [P, Q, R] = rbc_shape_constants(scaling);

    % Target RBC volume in voxels
    sphere_volume_voxels = (4/3) * pi * (radius_sphere / Resolution)^3;
    target_voxel_volume = rbc_filling_factor * sphere_volume_voxels;

    % Keep best result over all restarts
    best_voxel_count = 0;
    best_voxel_mask = false(Croped_clot_dim, Croped_clot_dim, Croped_clot_dim);
    best_rbc_pts = [];
    best_rbc_idx = {};

    for trial = 1:num_restarts
        temp_voxel_mask = false(Croped_clot_dim, Croped_clot_dim, Croped_clot_dim);
        temp_rbc_pts = [];
        temp_rbc_idx = {};
        voxel_count = 0;

        for step = 1:max_steps
            candidate_center = generate_random_point_in_sphere(radius_sphere - rbc_d/2, center_sphere);
            rot = rand(1, 3) * pi;
            rbc_pts = generate_rbc_voxels(candidate_center, rot, P, Q, R, voxel_spacing);

            if size(rbc_pts,1) < 4
                continue;
            end

            rbc_idx_array = map_to_matrix(rbc_pts);
            valid_mask = all(rbc_idx_array > 0, 2) & all(rbc_idx_array <= Croped_clot_dim, 2);
            valid_vox = rbc_idx_array(valid_mask, :);

            lin_idx = sub2ind([Croped_clot_dim, Croped_clot_dim, Croped_clot_dim], ...
                              valid_vox(:,1), valid_vox(:,2), valid_vox(:,3));

            % Voxel-level hard constraint
            if all(ClotMatrix(lin_idx) == 0 & ~temp_voxel_mask(lin_idx))
                temp_voxel_mask(lin_idx) = true;
                offset = size(temp_rbc_pts, 1);
                temp_rbc_pts = [temp_rbc_pts; rbc_pts];
                temp_rbc_idx{end+1} = offset + (1:size(rbc_pts,1));
                voxel_count = voxel_count + numel(lin_idx);

                if voxel_count >= target_voxel_volume
                    break;
                end
            end
        end

        % Keep best trial
        if voxel_count > best_voxel_count
            best_voxel_mask = temp_voxel_mask;
            best_rbc_pts = temp_rbc_pts;
            best_rbc_idx = temp_rbc_idx;
            best_voxel_count = voxel_count;
        end
    end

    % Add best result to final matrix
    ClotMatrix(best_voxel_mask) = 1;
    offset = size(rbc_points_all, 1);
    rbc_points_all = [rbc_points_all; best_rbc_pts];
    for k = 1:length(best_rbc_idx)
        rbc_indices{end+1} = offset + best_rbc_idx{k};
    end
end
end

% ==========================================================
% Helper Function: generate_random_point_in_sphere
% ==========================================================
function pt = generate_random_point_in_sphere(R, center)
theta = 2 * pi * rand(); phi = acos(2 * rand() - 1); r = R * rand()^(1/3);
pt = center + [r * sin(phi) * cos(theta), r * sin(phi) * sin(theta), r * cos(phi)];
end
% ==========================================================
% Helper Function: rbc_shape_constants
% ==========================================================
function [P, Q, R] = rbc_shape_constants(scaling)
d = 8 / scaling; br = 1 / scaling; h = 2.12 / scaling;
P = -(d^2 / 2) + (h^2 / 2) * ((d^2 / br^2) - 1) - h^2 / 2 * ((d^2 / br^2) - 1) * sqrt(1 - (br^2 / h^2));
Q = P * (d^2 / br^2) + (br^2 / 4) * (d^4 / br^4 - 1);
R = -P * (d^2 / 4) - d^4 / 16;
end
% ==========================================================
% Helper Function: generate_rbc_voxels
% ==========================================================
function pts = generate_rbc_voxels(center, rot, P, Q, R, voxel_spacing)
    % Create a local grid around the center
    range = 10; % µm from center to define bounding box
    [x, y, z] = meshgrid(center(1)-range:voxel_spacing:center(1)+range, ...
                         center(2)-range:voxel_spacing:center(2)+range, ...
                         center(3)-range:voxel_spacing:center(3)+range);

    % Apply rotation
    [xr, yr, zr] = rotate_grid(x, y, z, center, rot);

    % Apply RBC implicit equation
    eq = ((xr).^2 + (yr).^2 + (zr).^2).^2 + P * ((xr).^2 + (yr).^2) + Q * (zr).^2 + R;

    % Keep only voxels inside the shape
    pts = [x(eq <= 0), y(eq <= 0), z(eq <= 0)];
end

% ==========================================================
% Helper Function: rotate_grid
% ==========================================================
function [xr, yr, zr] = rotate_grid(x, y, z, center, rot)
x = x - center(1); y = y - center(2); z = z - center(3);
% X-rotation
xt = x; zt = z;
x = xt * cos(rot(1)) - zt * sin(rot(1));
z = xt * sin(rot(1)) + zt * cos(rot(1));
% Y-rotation
yt = y; zt = z;
y = yt * cos(rot(2)) + zt * sin(rot(2));
z = -yt * sin(rot(2)) + zt * cos(rot(2));
% Z-rotation
xt = x; yt = y;
x = xt * cos(rot(3)) - yt * sin(rot(3));
y = xt * sin(rot(3)) + yt * cos(rot(3));
xr = x; yr = y; zr = z;
end
% ==========================================================
% Helper Function: draw_fibrin_bonds
% ==========================================================
function [ClotMatrix, bond_length_total] = draw_fibrin_bonds(bonds, Points, crop_min, crop_max, Croped_clot_dim, map_to_matrix)
%DRAW_FIBRIN_BONDS Initializes and draws curved fibrin bonds using quadratic Bezier curves
%   bonds: [N x 2] index pairs of connected points
%   Points: [M x 3] coordinates of the nodes
%   crop_min, crop_max: 1x3 bounds for cropping
%   Croped_clot_dim: scalar value indicating the size of a cubic 3D ClotMatrix (e.g., 240)
%   map_to_matrix: function to convert physical coordinates to matrix indices

% Initialize ClotMatrix as a cubic 3D uint8 matrix
ClotMatrix = zeros(Croped_clot_dim, Croped_clot_dim, Croped_clot_dim, 'uint8');
bond_length_total = 0;
num_bond_points = 50;

% Precompute Bezier parameter t
t = linspace(0, 1, num_bond_points)';
T0 = (1 - t).^2;
T1 = 2 * (1 - t) .* t;
T2 = t.^2;

for i = 1:size(bonds, 1)
    P0 = Points(bonds(i, 1), :);
    P2 = Points(bonds(i, 2), :);

    % Randomized control point for Bezier curve
    P1 = (P0 + P2) / 2 + (rand(1, 3) - 0.5) * 2;

    % Estimate bond length in voxel space
    P_idx_0 = map_to_matrix(P0);
    P_idx_1 = map_to_matrix(P1);
    if all(P_idx_0 > 0 & P_idx_0 <= Croped_clot_dim) && all(P_idx_1 > 0 & P_idx_1 <= Croped_clot_dim)
        bond_length_total = bond_length_total + sqrt(sum((P_idx_1 - P_idx_0).^2));
    end

    % Bezier curve points (vectorized)
    bezier_curve = T0 * P0 + T1 * P1 + T2 * P2;

    % Filter out-of-crop points
    inside_crop = all(bezier_curve >= crop_min, 2) & all(bezier_curve <= crop_max, 2);
    voxel_points = bezier_curve(inside_crop, :);

    % Map to voxel indices and write to ClotMatrix
    idx_vox = map_to_matrix(voxel_points);
    valid_vox = all(idx_vox > 0, 2) & all(idx_vox <= Croped_clot_dim, 2);
    idx_vox = idx_vox(valid_vox, :);
    lin_idx = sub2ind([Croped_clot_dim, Croped_clot_dim, Croped_clot_dim], idx_vox(:,1), idx_vox(:,2), idx_vox(:,3));
    empty_voxel_mask = ClotMatrix(lin_idx) == 0;
    ClotMatrix(lin_idx(empty_voxel_mask)) = 2;
end
end
% ==========================================================
% Helper Function: insert_platelets
% ==========================================================
function [ClotMatrix, num_platelets_output, platelet_centers] = insert_platelets(ClotMatrix, Points, connection_counts, platelet_radius, platelet_ratio, crop_min, crop_max, Croped_clot_dim, map_to_matrix, Resolution)
%INSERT_PLATELETS Inserts spherical platelet voxels at cross-link points (valency = 4)

% Identify eligible points
platelet_indices = find(connection_counts == 4);
num_platelets_to_plot = round(platelet_ratio * length(platelet_indices));
platelet_indices = randsample(platelet_indices, num_platelets_to_plot);
platelet_centers = Points(platelet_indices,:);
num_platelets_output = 0;

% Precompute grid offsets
grid_range = -platelet_radius:Resolution:platelet_radius;
[xg, yg, zg] = ndgrid(grid_range, grid_range, grid_range);
offsets = [xg(:), yg(:), zg(:)];
distances = sqrt(sum(offsets.^2, 2));
unit_sphere_offsets = offsets(distances <= platelet_radius, :);

% Loop through each platelet index
for i = 1:num_platelets_to_plot
    center = Points(platelet_indices(i), :);

    if all(center >= crop_min & center <= crop_max)
        center_idx = map_to_matrix(center);
        if all(center_idx > 0 & center_idx <= Croped_clot_dim)
            num_platelets_output = num_platelets_output + 1;
            voxel_coords = unit_sphere_offsets + center;

            % Map to matrix indices
            idx_vox = map_to_matrix(voxel_coords);
            valid_vox = all(idx_vox > 0, 2) & all(idx_vox <= Croped_clot_dim, 2);
            idx_vox = idx_vox(valid_vox, :);
            lin_idx = sub2ind([Croped_clot_dim, Croped_clot_dim, Croped_clot_dim], idx_vox(:,1), idx_vox(:,2), idx_vox(:,3));
            empty_voxel_mask = ClotMatrix(lin_idx) == 0;
            ClotMatrix(lin_idx(empty_voxel_mask)) = 3;
        end
    end
end
end

% ==========================================================
% Helper Function: tubeAlongCurve
% ==========================================================
function [V, F] = tubeAlongCurve(P, radius, seg)
N = size(P,1);
V = []; F = [];
for i = 1:N
    t = max(i-1,1); u = min(i+1,N);
    dir = P(u,:) - P(t,:);
    dir = dir / norm(dir + 1e-6);
    [cx, cy, ~] = cylinder(radius, seg);
    ring = [cx(1, 1:end-1)', cy(1, 1:end-1)', zeros(seg,1)];
    R = vrrotvec2mat(vrrotvec([0 0 1], dir));
    ring = (R * ring')' + P(i,:);
    V = [V; ring];
    if i > 1
        idx = size(V,1) - 2*seg;
        for j = 1:seg
            j2 = mod(j,seg)+1;
            F = [F; idx+j, idx+seg+j, idx+seg+j2; idx+j, idx+seg+j2, idx+j2];
        end
    end
end
end
% ==========================================================
% Helper Function: writePLYMesh
% ==========================================================
function writePLYMesh(vertices, faces, colors, filename)

% Start file and write header
fid = fopen(filename, 'w');
fprintf(fid, 'ply\nformat ascii 1.0\n');
fprintf(fid, 'element vertex %d\n', size(vertices,1));
fprintf(fid, 'property float x\nproperty float y\nproperty float z\n');
fprintf(fid, 'property uchar red\nproperty uchar green\nproperty uchar blue\n');
fprintf(fid, 'element face %d\n', size(faces,1));
fprintf(fid, 'property list uchar int vertex_indices\n');
fprintf(fid, 'end_header\n');
fclose(fid); % close for now; we'll append below

% === Build vertex lines in parallel
vertex_lines = strings(size(vertices,1), 1);
parfor i = 1:size(vertices,1)
    vertex_lines(i) = sprintf('%.4f %.4f %.4f %d %d %d\n', ...
        vertices(i,1), vertices(i,2), vertices(i,3), ...
        colors(i,1), colors(i,2), colors(i,3)); %#ok<PFBNS>
end

% === Build face lines in parallel
face_lines = strings(size(faces,1), 1);
parfor i = 1:size(faces,1)
    face_lines(i) = sprintf('3 %d %d %d\n', ...
        faces(i,1)-1, faces(i,2)-1, faces(i,3)-1);
end

% === Concatenate all lines and write to file
fid = fopen(filename, 'a'); % append mode
fprintf(fid, '%s', join(vertex_lines, ''));
fprintf(fid, '%s', join(face_lines, ''));
fclose(fid);

end


%% icoshpehre
function [V,F] = icosphere(subdivisions)
% Create an icosphere mesh (triangulated sphere)
t = (1 + sqrt(5)) / 2;

verts = [-1,  t,  0;
    1,  t,  0;
    -1, -t,  0;
    1, -t,  0;
    0, -1,  t;
    0,  1,  t;
    0, -1, -t;
    0,  1, -t;
    t,  0, -1;
    t,  0,  1;
    -t,  0, -1;
    -t,  0,  1];

faces = [1,12,6; 1,6,2; 1,2,8; 1,8,11; 1,11,12;
    2,6,10; 6,12,5; 12,11,3; 11,8,7; 8,2,9;
    4,10,5; 4,5,3; 4,3,7; 4,7,9; 4,9,10;
    5,10,6; 3,5,12; 7,3,11; 9,7,8; 10,9,2];

V = verts ./ vecnorm(verts, 2, 2); % normalize
F = faces;

for i = 1:subdivisions
    [F, V] = subdivide(F, V);
    V = V ./ vecnorm(V, 2, 2);  % re-normalize to unit sphere
end
end

function [F2, V2] = subdivide(F, V)
midpoint = containers.Map('KeyType','char','ValueType','int32');
V2 = V;
F2 = zeros(size(F,1)*4, 3);
vidx = size(V,1);

for i = 1:size(F,1)
    tri = F(i,:);
    a = getMid(tri(1), tri(2), V, midpoint, vidx); vidx = max(vidx, a);
    b = getMid(tri(2), tri(3), V, midpoint, vidx); vidx = max(vidx, b);
    c = getMid(tri(3), tri(1), V, midpoint, vidx); vidx = max(vidx, c);
    F2((i-1)*4 + (1:4), :) = [
        tri(1) a c;
        tri(2) b a;
        tri(3) c b;
        a b c];
end

    function idx = getMid(i1, i2, V, map, ~)
        key = sprintf('%d-%d', min(i1,i2), max(i1,i2));
        if map.isKey(key)
            idx = map(key);
        else
            newV = (V(i1,:) + V(i2,:))/2;
            V2(end+1,:) = newV;
            idx = size(V2,1);
            map(key) = idx;
        end
    end
end