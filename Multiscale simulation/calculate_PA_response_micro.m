function sensor_data = calculate_PA_response_micro(matrix)
    % Setup optical properties
    Fluence_rate = ones(size(matrix));
    absorption_coefficients = zeros(size(matrix));
   
    % Assign properties based on matrix labels
    absorption_coefficients(matrix == 1) = 23;
    absorption_coefficients(matrix == 2) = 0.005;
    absorption_coefficients(matrix == 3) = 0.5;


    % Calculate absorbed energy
    absorbed_energy = Fluence_rate .* absorption_coefficients;

    % Use 72 um block, no interpolation needed (already 144^3 at 0.5 um)
    Voxel_size_um = 72;  % um
    Nx = size(absorbed_energy, 1);
    Ny = size(absorbed_energy, 2);
    Nz = size(absorbed_energy, 3);
    
    dx = Voxel_size_um * 1e-6 / Nx;  % Convert to meters
    dy = dx;
    dz = dx;

    % Create k-Wave grid
    kgrid = kWaveGrid(Nx, dx, Ny, dy, Nz, dz);

    % Initial pressure source
    gruneisen_parameter = 0.16;
    source.p0 = gruneisen_parameter .* absorbed_energy;

    % Medium properties
    medium.sound_speed = 1540 * ones(Nx, Ny, Nz);  % background
    medium.sound_speed(matrix == 1 | matrix == 2 | matrix == 3) = 2380;
    medium.density = 1025 * ones(Nx, Ny, Nz);      % background
    medium.density(matrix == 1 | matrix == 2 | matrix == 3) = 2000;

    % Time array
    T = 5e-7;        % total time
    dt = 5e-11;      % time step
    kgrid.t_array = 0:dt:T;

    % Define sensor mask: all 6 faces
    sensor.mask = zeros(Nx, Ny, Nz);
    sensor.mask(1, :, :)     = 1;  % front
    sensor.mask(end, :, :)   = 1;  % back
    sensor.mask(:, 1, :)     = 1;  % left
    sensor.mask(:, end, :)   = 1;  % right
    sensor.mask(:, :, 1)     = 1;  % bottom
    sensor.mask(:, :, end)   = 1;  % top

    % Run simulation
    sensor_data = kspaceFirstOrder3DG(kgrid, medium, source, sensor, ...
        'PMLInside', false, ...
        'PlotSim', false);  % Add 'DataCast','single' for speed if needed
end
