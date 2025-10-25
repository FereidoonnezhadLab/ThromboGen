function Voxel_response = calculate_PA_response_micro_gpu(matrix, Voxel_size_um, cuda_dir)
% CALCULATE_PA_RESPONSE_MICRO_GPU
% GPU-based k-Wave PA simulation for one voxel block and returns 6×N response
%
% Inputs:
%   matrix        : 3D uint8 matrix (0=void, 1=RBC, 2=Fibrin, 3=Platelet)
%   Voxel_size_um : physical voxel size in micrometers
%   cuda_dir      : path to k-Wave CUDA executable
%
% Output:
%   Voxel_response: 6×N matrix (six boundary faces × time series)

    %% --- Add k-Wave path ---
    addpath(genpath('/home/hghodsi/Software_packages/k-wave-toolbox-version-1.4/k-Wave'));
    exe_path = fullfile(cuda_dir, 'kspaceFirstOrder-CUDA');
    if ~isfile(exe_path)
        error('kspaceFirstOrder-CUDA not found in %s', cuda_dir);
    end

    %% --- Simulation grid ---
    Nx = size(matrix,1);
    Ny = Nx; Nz = Nx;
    dx = Voxel_size_um * 1e-6 / Nx;
    kgrid = kWaveGrid(Nx, dx, Ny, dx, Nz, dx);

    %% --- Optical absorption model ---
    Fluence_rate = ones(size(matrix), 'single');
    absorption_coefficients = 0.49*ones(size(matrix), 'single');
    absorption_coefficients(matrix==1)=23;      % RBC
    absorption_coefficients(matrix==2)=0.005;   % Fibrin
    absorption_coefficients(matrix==3)=0.5;     % Platelet
    absorbed_energy = Fluence_rate .* absorption_coefficients;
    source.p0 = 0.16 * absorbed_energy;

    %% --- Acoustic parameters ---
    medium.sound_speed = 1525 * ones(Nx,Ny,Nz,'single');
    medium.sound_speed(matrix==1) = 1570;
    medium.sound_speed(matrix==2) = 1567.5;
    medium.sound_speed(matrix=3) = 1540;
    medium.density = 1027.5 * ones(Nx,Ny,Nz,'single');
    medium.density(matrix==1) = 1125;
    medium.density(matrix==2) = 1060;
    medium.density(matrix==3) = 1080;

    %% --- Time parameters ---
    T = 1e-5;      % total simulation time (10 µs)
    dt = 5e-11;    % base step (as in your batch file)
    new_dt = 5e-10;
    downsample_factor = new_dt / dt;
    T_micro = 5e-7;        % total time microscale
    kgrid.t_array = 0:dt:T_micro;
    T_array = 0:new_dt:T;

    %% --- Sensor mask (6 faces) ---
    sensor.mask = zeros(Nx,Ny,Nz,'logical');
    sensor.mask(1,:,:)   = 1;
    sensor.mask(end,:,:) = 1;
    sensor.mask(:,1,:)   = 1;
    sensor.mask(:,end,:) = 1;
    sensor.mask(:,:,1)   = 1;
    sensor.mask(:,:,end) = 1;

    %% --- Temporary I/O paths ---
    timestamp = datestr(now,'yyyymmdd_HHMMSS_FFF');
    input_file  = fullfile(cuda_dir, ['input_' timestamp '.h5']);
    tmp_dir     = '/beegfs/scratch/hghodsi/kWave_temp';
    if ~isfolder(tmp_dir), mkdir(tmp_dir); end
    output_file = fullfile(tmp_dir, ['output_' timestamp '.h5']);

    %% --- Save k-Wave input ---
    kspaceFirstOrder3D(kgrid, medium, source, sensor, ...
        'PMLInside', false, 'PlotSim', false, 'SaveToDisk', input_file);

    %% --- Run CUDA solver ---
    timeout_sec = 600;
    cmd = sprintf('timeout %d bash -c "cd %s && ./kspaceFirstOrder-CUDA -i %s -o %s --p_raw"', ...
        timeout_sec, cuda_dir, input_file, output_file);
    fprintf('Running CUDA command:\n%s\n', cmd);
    [status, cmdout] = system(cmd);

    if status ~= 0
        warning('CUDA simulation failed (status %d): %s', status, cmdout);
        Voxel_response = [];
        if isfile(input_file), delete(input_file); end
        return;
    end

    %% --- Read result ---
    try
        sensor_data = h5read(output_file, '/p');  % raw boundary data
    catch ME
        warning('Failed to read %s: %s', output_file, ME.message);
        sensor_data = [];
    end

    %% --- Cleanup ---
    if isfile(input_file), delete(input_file); end
    if isfile(output_file), delete(output_file); end
    try gpuDevice([]); end %#ok<TRYNC>

    %% --- Integration, reshape, and downsampling ---
    if isempty(sensor_data)
        Voxel_response = [];
        return;
    end

    % sensor_data: (#boundary_points × time_points)
    time_points = size(sensor_data, 2);
    spatial_length = size(sensor_data, 1);
    grid_length = sqrt(spatial_length / 6);
    if abs(grid_length - round(grid_length)) > 1e-6
        warning('Unexpected sensor data size, skipping voxel');
        Voxel_response = [];
        return;
    end

    reshaped = reshape(sensor_data, [grid_length, grid_length, 6, time_points]);
    avg_responses = squeeze(mean(mean(reshaped, 1), 2));   % average per face
    padded = padarray(avg_responses, [0, length(T_array) - size(avg_responses, 2)], 0, 'post');
    Voxel_response = downsample(padded', downsample_factor)'; % final 6×N

end
