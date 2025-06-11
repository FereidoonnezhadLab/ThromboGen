% load filtered_outputs_II.mat
T = 1e-5;              % Total time duration
dt = 5e-11;            % Original time step
new_dt = 5e-10;        % New desired time step (downsample factor = 10)
downsample_factor = new_dt / dt;
T_array = 0:dt:T;      % Original time array
new_T_array = 0:new_dt:T; % Downsampled time array
ClotMatrix = ClotMatrix_valid;
addpath(genpath('C:\GitHub\kwave'));
N = size(ClotMatrix,1); % Number of iterations
h = waitbar(0, 'Progress'); % Create a progress bar window
for i = 1:N
    waitbar(i / N, h, sprintf('Overall Progress: %d%%', round((i / N) * 100)));
    pause(0.1); % Optional: Add a delay to slow down the progress for demonstration purposes
    Voxel_response = calculate_PA_response_micro(squeeze(ClotMatrix(i,:,:,:)));
    Voxel_response = Voxel_response(1:end-2,:);
    % Define the grid size
    grid_size = [119, 119];
    num_faces = 6;
    num_time_points = size(Voxel_response, 2);
    % Reshape the data: each face as a separate grid (119x119) over time
    Voxel_response_reshaped = reshape(Voxel_response, [grid_size(1), grid_size(2), num_faces, num_time_points]);

    % Preallocate for averaged responses
    averaged_responses = zeros(num_faces, num_time_points);

    % Loop over each face and compute the average over the grid
    for face = 1:num_faces
        % Average over the 119x119 grid for the current face
        averaged_responses(face, :) = mean(mean(Voxel_response_reshaped(:, :, face, :), 1), 2);
    end

    % Reshape averaged_responses to 6 time responses
    averaged_responses = squeeze(averaged_responses); % Final size: 6 x time points

    % Zero-padding to match the desired duration
    averaged_responses_padded = padarray(averaged_responses, [0, length(T_array) - size(averaged_responses, 2)], 0, 'post');

    % Downsample the data to the new time step
    averaged_responses_downsampled(i,:,:) = downsample(averaged_responses_padded', downsample_factor)';
end
close (h);
save Clot_responses.mat averaged_responses_downsampled
