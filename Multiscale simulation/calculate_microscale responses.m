% =====================================
% Process PA responses for 144³ clots
% =====================================

% Constants
T = 1e-5;                  % Total time duration (10 µs)
dt = 5e-11;                % Original time step (50 ps)
new_dt = 5e-10;            % New desired time step (500 ps)
downsample_factor = new_dt / dt;
T_array = 0:dt:T;
new_T_array = 0:new_dt:T;

addpath(genpath('C:\GitHub\Software_packages'));

% Group file names (each contains 50 clots max)
groupFiles = {
    'Group_1_VeryStiff.mat', ...
    'Group_2_Stiff.mat', ...
    'Group_3_Normal.mat', ...
    'Group_4_VeryLoose.mat'
};

% Loop through each group
for g = 1:numel(groupFiles)
    % Load the group
    load(groupFiles{g});  % Loads: Micro (144x144x144xN), Por, Comp
    N = size(Micro, 4);   % Number of clots
    
    fprintf('Processing %s with %d samples...\n', groupFiles{g}, N);

    % Preallocate output: [N, 6 faces, new_time_points]
    averaged_responses_downsampled = zeros(N, 6, numel(new_T_array));
    
    % Progress bar
    h = waitbar(0, sprintf('Processing %s...', groupFiles{g}));
    
    for i = 1:N
        waitbar(i / N, h, sprintf('%s - %d%%', groupFiles{g}, round(i/N*100)));

        % Extract one clot
        clot_volume = squeeze(Micro(:, :, :, i));  % size: 144 x 144 x 144

        % Calculate PA response
        Voxel_response = calculate_PA_response_micro(clot_volume);  % Expected: [grid^2 * 6, time]
        Voxel_response = Voxel_response(1:end-2, :);  % Drop trailing empty lines if needed

        % Detect grid size dynamically
        time_points = size(Voxel_response, 2);
        spatial_length = size(Voxel_response, 1);
        grid_length = sqrt(spatial_length / 6);  % Assuming square grid per face
        if mod(grid_length, 1) ~= 0
            error('Voxel response size is incompatible with 6 face reshaping.');
        end
        grid_size = [grid_length, grid_length];  % E.g., [119, 119] or smaller

        % Reshape and average per face
        reshaped = reshape(Voxel_response, [grid_size(1), grid_size(2), 6, time_points]);
        avg_responses = squeeze(mean(mean(reshaped, 1), 2));  % [6 x time]

        % Pad to full T_array length if needed
        padded = padarray(avg_responses, [0, length(T_array) - size(avg_responses, 2)], 0, 'post');

        % Downsample
        averaged_responses_downsampled(i,:,:) = downsample(padded', downsample_factor)';  % [6 x downsampled_time]
    end
    
    close(h);

    % Save results
    [~, baseName, ~] = fileparts(groupFiles{g});
    save([baseName '_Responses.mat'], 'averaged_responses_downsampled', 'Por', 'Comp', '-v7.3');
end

disp('All groups processed and saved.');
