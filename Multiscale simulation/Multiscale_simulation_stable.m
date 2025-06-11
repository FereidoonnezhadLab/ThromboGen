function sensor_data = Multiscale_simulation(averaged_responses_downsampled,ClotMatrix)
addpath(genpath('C:\GitHub\Software_packages'))
seed = 15;
rng(seed);
num_variations = size(averaged_responses_downsampled,1);
averaged_responses_downsampled = single(averaged_responses_downsampled);

%----------------------------------------------------------------------
% 4) Create the cylindrical mask
%----------------------------------------------------------------------
voxel_size        = 120e-6;       % in meters
cylinder_diameter = 5e-3;         % in meters
cylinder_height   = 1e-3;         % in meters
ROI_size = cylinder_diameter+4e-3;
matrix_size = round(ROI_size/voxel_size);
rotation_matrix   = [0 0 0];   % rotation vector
% Create a 3D mask for the clot
clot_mask = makeCylinder( ...
    matrix_size, matrix_size, matrix_size, ...
    matrix_size/2, matrix_size/2, matrix_size/2, ...
    (cylinder_diameter/2) / voxel_size, ...
    cylinder_height / voxel_size, ...
    rotation_matrix);

% If the mask is large, convert to single
clot_mask = single(clot_mask);

% Find all voxels that belong to the cylinder
[idx_x, idx_y, idx_z] = ind2sub(size(clot_mask), find(clot_mask == 1));
num_coordinates       = numel(idx_x);
voxelPlot(clot_mask)

% % Preallocate assigned_matrices in single
assigned_matrices = zeros(num_coordinates, 6, 20001, 'single');

%----------------------------------------------------------------------
% 5) Set up the mesh for ValoMC
%----------------------------------------------------------------------
x_arr = linspace(-matrix_size/2, matrix_size/2, matrix_size) * voxel_size * 1e3; % mm
y_arr = x_arr;  % symmetric
z_arr = x_arr;  % symmetric

% ValoMC typically works with double-precision for geometry
vmcmesh   = createGridMesh(x_arr, y_arr, z_arr);
vmcmedium = createMedium(vmcmesh);

% Preallocate optical parameters in single
scattering_coefficients = single(zeros(size(clot_mask)));
absorption_coefficients = single(zeros(size(clot_mask)));
scattering_anisotropies = single(0.9678 * ones(size(clot_mask), 'single'));
refractive_indexes      = single(1.0    * ones(size(clot_mask), 'single'));

%----------------------------------------------------------------------
% 6) Set up the k-Wave Grid
%    (We use Nx = matrix_size*6, etc., per your code)
%----------------------------------------------------------------------
dx = voxel_size / 6;
Nx = matrix_size * 6;
Ny = Nx;
Nz = Nx;
kgrid = kWaveGrid(Nx, dx, Ny, dx, Nz, dx);

% Medium in single
medium.sound_speed = single(1540 * ones(Nx, Ny, Nz, 'single'));   % [m/s]
medium.density     = single(1025 * ones(Nx, Ny, Nz, 'single'));   % [kg/m^3]

%----------------------------------------------------------------------
% 7) Resample clot indices to the finer grid
%----------------------------------------------------------------------
[idx_new_x, idx_new_y, idx_new_z] = resample_indices( ...
    idx_x, idx_y, idx_z, matrix_size, 6*matrix_size);

%----------------------------------------------------------------------
h = waitbar(0, 'Progress'); % Create a progress bar window
for i = 1:num_coordinates
    random_index = randi(num_variations);
    clot_matrix = squeeze(ClotMatrix(random_index,:,:,:));
    waitbar((i-1) / num_coordinates, h, sprintf('clot construction: %d%%', round((i/ num_coordinates) * 100)));
    pause(0.01); % Optional: Add a delay to slow down the progress for demonstration purposes
    % Apply Random 90-degree Rotations
    [ax, angle, clot_matrix] = apply_random_rotation(clot_matrix);
    % Rotate response matrix accordingly
    assigned_matrices(i, :, :) = rotate_response_matrix(averaged_responses_downsampled(random_index, :, :),ax, angle);
  
    % Evaluate optical parameters in single
    absorption_coefficients(idx_x(i), idx_y(i), idx_z(i)) = ...
        down_sample_apply_parameter(clot_matrix, 0, 22.48, 5, 5, ...
        2*floor(voxel_size / 1e-6));
    scattering_coefficients(idx_x(i), idx_y(i), idx_z(i)) = ...
        down_sample_apply_parameter(clot_matrix, 0, 72.78, 72.78, 72.78, ...
        2*floor(voxel_size / 1e-6));
    refractive_indexes(idx_x(i), idx_y(i), idx_z(i)) = ...
        down_sample_apply_parameter(clot_matrix, 1, 1.4, 1.4, 1.4, ...
        2*floor(voxel_size / 1e-6));

    sx = 6; sy = sx;  sz = sy;

    % Insert smaller blocks into the large medium arrays
    base_x = idx_new_x(i) - round(sx/2);
    base_y = idx_new_y(i) - round(sy/2);
    base_z = idx_new_z(i) - round(sz/2);

    medium.sound_speed(base_x + (1:sx), ...
        base_y + (1:sy), ...
        base_z + (1:sz)) = down_sample_apply_parameter(clot_matrix, 1540, 1800, 1800, 1800, ...
        2*floor((voxel_size/6)/1e-6));

    medium.density(base_x + (1:sx), ...
        base_y + (1:sy), ...
        base_z + (1:sz)) = down_sample_apply_parameter(clot_matrix, 1025, 2000, 2000, 2000, ...
        2*floor((voxel_size/6)/1e-6));
end
close(h);
%----------------------------------------------------------------------
% 9) Assign optical properties to vmcmedium
%----------------------------------------------------------------------
% ValoMC expects these as double, so we might convert them back if needed
% For safety, use double(...) for vmcmedium if ValoMC requires it:
vmcmedium.scattering_anisotropy   = repmat(double(scattering_anisotropies(:)), 6, 1);
vmcmedium.absorption_coefficient  = repmat(double(absorption_coefficients(:)), 6, 1);
vmcmedium.scattering_coefficient  = repmat(double(scattering_coefficients(:)), 6, 1);
vmcmedium.refractive_index        = repmat(double(refractive_indexes(:)), 6, 1);

vmcboundary = createBoundary(vmcmesh, vmcmedium);
lightsource = findBoundaries(vmcmesh, 'direction', [0 0 0], [0 0 -10], 5);
vmcboundary.lightsource(lightsource) = {'cosinic'};

% Solve the light distribution
solution = ValoMC(vmcmesh, vmcmedium, vmcboundary);

%----------------------------------------------------------------------
% 10) Map the fluence onto a regular grid
%----------------------------------------------------------------------
TR = triangulation(double(vmcmesh.H), vmcmesh.r);  % Must be double for triangulation
[X, Y, Z] = meshgrid(y_arr, x_arr, z_arr);         % Usually double

locations = [X(:), Y(:), Z(:)];
indices = pointLocation(TR, locations);
indices(isnan(indices)) = 1;

elem_flu = single(solution.element_fluence);   % Convert to single if large
grid_fluence = reshape(elem_flu(indices), size(X));
grid_fluence = single(grid_fluence);

absorbed_energy = absorption_coefficients .* grid_fluence * 1e3;
absorbed_energy_per_voxel = absorbed_energy .* clot_mask;

%----------------------------------------------------------------------
% 11) Define time array for k-Wave
%----------------------------------------------------------------------
T       = 1e-5;
dt      = 5e-10;
T_array = 0:dt:T;
kgrid.t_array = T_array;
%----------------------------------------------------------------------
% 12) Create the source mask
%----------------------------------------------------------------------
source.u_mask = zeros(Nx, Ny, Nz, 'single');
offsets = [ 1,  0,  0;
    -1,  0,  0;
    0,  1,  0;
    0, -1,  0;
    0,  0,  1;
    0,  0, -1];

num_points = num_coordinates;
for i = 1:num_points
    x = idx_new_x(i);
    y = idx_new_y(i);
    z = idx_new_z(i);

    for j = 1:6
        xn = x + offsets(j,1);
        yn = y + offsets(j,2);
        zn = z + offsets(j,3);
        if xn>=1 && xn<=Nx && yn>=1 && yn<=Ny && zn>=1 && zn<=Nz
            source.u_mask(xn, yn, zn) = 1;
        end
    end
end

%----------------------------------------------------------------------
% 13) Assign the time responses to source
%----------------------------------------------------------------------
clearvars -except kgrid medium source sensor absorbed_energy_per_voxel assigned_matrices num_points ...
    Nx Ny Nz idx_x idx_y idx_z idx_new_x idx_new_y idx_new_z dx
num_timesteps = length(kgrid.t_array);
source.ux = zeros(6 * num_points, num_timesteps, 'single');
source.uy = zeros(6 * num_points, num_timesteps, 'single');
source.uz = zeros(6 * num_points, num_timesteps, 'single');

for i = 1:num_points
    % For each voxel, multiply the assigned 6x(time) matrix by the
    % local absorbed energy
    response_per_voxel = absorbed_energy_per_voxel(idx_x(i), idx_y(i), idx_z(i)) ...
        .* squeeze(assigned_matrices(i, :, :));  % 6 x time

    row_idx = 6*(i-1);
    Ro = single(medium.density(idx_new_x(i),idx_new_y(i),idx_new_z(i)));
    c = single(medium.sound_speed(idx_new_x(i),idx_new_y(i),idx_new_z(i)));

    source.ux(row_idx + 1, :) = +(1/(Ro*c))*single(response_per_voxel(1, :));
    source.ux(row_idx + 2, :) = -(1/(Ro*c))*single(response_per_voxel(2, :));
    source.uy(row_idx + 3, :) = +(1/(Ro*c))*single(response_per_voxel(3, :));
    source.uy(row_idx + 4, :) = -(1/(Ro*c))*single(response_per_voxel(4, :));
    source.uz(row_idx + 5, :) = +(1/(Ro*c))*single(response_per_voxel(5, :));
    source.uz(row_idx + 6, :) = -(1/(Ro*c))*single(response_per_voxel(6, :));
end

%----------------------------------------------------------------------
% 14) Define the sensor (at 0 degrees)
%----------------------------------------------------------------------
sensor.mask = single(zeros(Nx, Ny, Nz, 'single'));

sensor_angle = 0;
center_sensor = [Nx/2, Ny/2, 0.9*Nz];

sensor_angle_rad = deg2rad(sensor_angle);
R_y = [cos(sensor_angle_rad), 0, sin(sensor_angle_rad);
    0,                    1, 0;
    -sin(sensor_angle_rad), 0, cos(sensor_angle_rad)];

Num_sensors_y = Ny;
Num_sensors_x = Nx;
[y_sensor, x_sensor] = meshgrid( ...
    linspace(-Num_sensors_y/2, Num_sensors_y/2, Num_sensors_y), ...
    linspace(-Num_sensors_x/2, Num_sensors_x/2, Num_sensors_x));
z_sensor = zeros(size(x_sensor));
sensor_grid = [x_sensor(:), y_sensor(:), z_sensor(:)]';

rotated_sensor_grid = R_y * sensor_grid;
rotated_sensor_grid(1,:) = rotated_sensor_grid(1,:) + center_sensor(1);
rotated_sensor_grid(2,:) = rotated_sensor_grid(2,:) + center_sensor(2);
rotated_sensor_grid(3,:) = rotated_sensor_grid(3,:) + center_sensor(3);

rotated_x = round(rotated_sensor_grid(1,:));
rotated_y = round(rotated_sensor_grid(2,:));
rotated_z = round(rotated_sensor_grid(3,:));

valid_idx = rotated_x>0 & rotated_x<=Nx & ...
    rotated_y>0 & rotated_y<=Ny & ...
    rotated_z>0 & rotated_z<=Nz;

lin_idx = sub2ind(size(sensor.mask), ...
    rotated_x(valid_idx), ...
    rotated_y(valid_idx), ...
    rotated_z(valid_idx));
sensor.mask(lin_idx) = 1;

%----------------------------------------------------------------------
% 15) Run the k-Wave 3D simulation
%----------------------------------------------------------------------
clearvars -except kgrid medium source sensor
sensor_data = kspaceFirstOrder3DC(kgrid, medium, source, sensor, ...
    'PMLInside', false);
sensor_data = single(sensor_data);  % If large, convert to single
end
%=========================================================================%
%  Subfunction: resample_indices
%=========================================================================%
function [idx_new_x, idx_new_y, idx_new_z] = resample_indices(idx_x, idx_y, idx_z, old_size, new_size)
% Resample indices from a coarse grid to a finer grid.

scale_factor = new_size / old_size;
idx_new_x = round((idx_x - 0.5) * scale_factor + 0.5);
idx_new_y = round((idx_y - 0.5) * scale_factor + 0.5);
idx_new_z = round((idx_z - 0.5) * scale_factor + 0.5);

% Ensure indices remain within bounds
idx_new_x = max(min(idx_new_x, new_size), 1);
idx_new_y = max(min(idx_new_y, new_size), 1);
idx_new_z = max(min(idx_new_z, new_size), 1);
end

%=========================================================================%
%  Subfunction: down_sample_apply_parameter
%=========================================================================%
function DownsampledMatrix = down_sample_apply_parameter(ClotMatrix, ...
    param_empty, param_rbc, param_fibrin, param_platelet, downsample_factor)
% 1) Replace codes in ClotMatrix with corresponding parameter values
% 2) Downsample by averaging blocks of size (downsample_factor^3)

ParameterMatrix = zeros(size(ClotMatrix), 'single');
ParameterMatrix(ClotMatrix == 0) = param_empty;
ParameterMatrix(ClotMatrix == 1) = param_rbc;
ParameterMatrix(ClotMatrix == 2) = param_fibrin;
ParameterMatrix(ClotMatrix == 3) = param_platelet;

[rows, cols, slices] = size(ParameterMatrix);
rows_ds   = floor(rows   / downsample_factor);
cols_ds   = floor(cols   / downsample_factor);
slices_ds = floor(slices / downsample_factor);

DownsampledMatrix = zeros(rows_ds, cols_ds, slices_ds, 'single');

for i = 1:rows_ds
    for j = 1:cols_ds
        for k = 1:slices_ds
            block = ParameterMatrix( ...
                (i-1)*downsample_factor+1 : i*downsample_factor, ...
                (j-1)*downsample_factor+1 : j*downsample_factor, ...
                (k-1)*downsample_factor+1 : k*downsample_factor);

            DownsampledMatrix(i,j,k) = mean(block(:));
        end
    end
end

end

%% Helper Function: Apply Random 90-degree Rotations
function [ax, angle, rotated_matrix] = apply_random_rotation(clot_matrix)
    rotations = [0, 90, 180, 270];
    ax = randi(3); % Random axis (1=x, 2=y, 3=z)
    angle = rotations(randi(4));

    switch ax
        case 1  % Rotate around X-axis
            for i = 1:(angle / 90)
                clot_matrix = permute(clot_matrix, [1, 3, 2]); % Swap Y and Z
                clot_matrix = flip(clot_matrix, 2); % Flip along Y
            end
        case 2  % Rotate around Y-axis
            for i = 1:(angle / 90)
                clot_matrix = permute(clot_matrix, [3, 2, 1]); % Swap X and Z
                clot_matrix = flip(clot_matrix, 1); % Flip along X
            end
        case 3  % Rotate around Z-axis
            for i = 1:(angle / 90)
                clot_matrix = permute(clot_matrix, [2, 1, 3]); % Swap X and Y
                clot_matrix = flip(clot_matrix, 2); % Flip along Y
            end
    end
    rotated_matrix = clot_matrix;
end


function rotated_response = rotate_response_matrix(response_matrix, ax, angle)
    response_matrix = squeeze(response_matrix); % Ensure it's 6xN

    % Ensure the input is 6 rows corresponding to [x, -x, y, -y, z, -z]
    if size(response_matrix, 1) ~= 6
        error('Input response_matrix must have 6 rows.');
    end

    % Define lookup table for row permutations
    rotation_map = struct(...
        'x_90',  [1, 2, 5, 6, 4, 3], ...
        'x_180', [1, 2, 4, 3, 6, 5], ...
        'x_270', [1, 2, 6, 5, 3, 4], ...
        'y_90',  [6, 5, 3, 4, 1, 2], ...
        'y_180', [2, 1, 3, 4, 6, 5], ...
        'y_270', [5, 6, 3, 4, 2, 1], ...
        'z_90',  [3, 4, 2, 1, 5, 6], ...
        'z_180', [2, 1, 4, 3, 5, 6], ...
        'z_270', [4, 3, 1, 2, 5, 6] ...
    );

    % Create key for lookup
    key = sprintf('%c_%d', 'x' + (ax - 1), angle);

    % Apply rotation if valid, otherwise keep original order
    if isfield(rotation_map, key)
        rotated_response = response_matrix(rotation_map.(key), :);
    else
        rotated_response = response_matrix; % No rotation if angle is not 90, 180, or 270
    end
end

