function [freq_axis, PCA] = Simulation_post_process(sensor_data, mode)
addpath(genpath('C:\GitHub\Software_packages'))

if nargin < 2
    mode = '2D';  % Default to 2D mode
end

%% Parameters
Fs = 1 / (5e-10);                % Sampling frequency (Hz)
dt = 1 / Fs;                     % Time step (s)
c = 1540;                        % Speed of sound (m/s)
pixel_size_um = 36;             % µm per pixel
detector_size_um = 36;         % Detector diameter (µm)
step_size_um = 36;             % Detector step size (µm)

%% Reshape sensor data into 3D matrix
N = size(sensor_data,2);
dimension = round(sqrt(size(sensor_data,1)));
sensor_data_3D = reshape(sensor_data(1:dimension^2,:), [dimension, dimension, N]);
sensor_data_3D = single(sensor_data_3D);
%% Flat-top Bandpass Filter
fc = 5e6; bw = 4e6;
down_factor = 10;
Fs = Fs/down_factor;
[b, a] = butter(2, [fc-bw/2 fc+bw/2]/(Fs/2), 'bandpass');

[x_size, y_size, t_size] = size(sensor_data_3D);
sensor_data_3D_ds = zeros(x_size, y_size, ceil(t_size / down_factor), 'like', sensor_data_3D);

for x = 1:x_size
    parfor y = 1:y_size
        signal = squeeze(sensor_data_3D(x, y, :));
        sensor_data_3D_ds(x, y, :) = resample(double(signal), 1, down_factor);
    end
end
sensor_data_3D = sensor_data_3D_ds;
for x = 1:size(sensor_data_3D,1)
    parfor y = 1:size(sensor_data_3D,2)
        signal = squeeze(sensor_data_3D(x,y,:));
        sensor_data_3D(x,y,:) = single(filtfilt(b, a, double(signal)));
    end
end

%% Integration (if applicable)
detector_size_pix = round(detector_size_um / pixel_size_um);
step_size_pix = round(step_size_um / pixel_size_um);
Nx = size(sensor_data_3D,1); Ny = size(sensor_data_3D,2);

if detector_size_um > pixel_size_um
    x_positions = 1:step_size_pix:(Nx - detector_size_pix + 1);
    y_positions = 1:step_size_pix:(Ny - detector_size_pix + 1);
    down_Nx = length(x_positions);
    down_Ny = length(y_positions);
    sensor_data_integrated = zeros(down_Nx, down_Ny, N, 'single');

    sigma = detector_size_pix / 3;
    gaussian_kernel = fspecial('gaussian', [detector_size_pix detector_size_pix], sigma);
    gaussian_kernel = gaussian_kernel / sum(gaussian_kernel(:));

    for t = 1:N
        frame = sensor_data_3D(:,:,t);
        for ix = 1:down_Nx
            for iy = 1:down_Ny
                x_start = x_positions(ix);
                y_start = y_positions(iy);
                x_end = x_start + detector_size_pix - 1;
                y_end = y_start + detector_size_pix - 1;
                block = frame(x_start:x_end, y_start:y_end);
                weighted_sum = sum(block(:) .* gaussian_kernel(:));
                sensor_data_integrated(ix, iy, t) = weighted_sum;
            end
        end
    end
    sensor_data_down_sampled = sensor_data_integrated;
else
    sensor_data_down_sampled = sensor_data_3D;
end

%% Slice for 2D or full data for 3D
if strcmpi(mode, '2D')
    sensor_data_input = squeeze(sensor_data_down_sampled(round(end/2),:,:));
else
    sensor_data_input = sensor_data_down_sampled;
end

%% PCA Spectral Analysis
sensor_data_flat = reshape(sensor_data_input, [], ceil(N/down_factor));  % [sensors x time]
S_fft = fft(sensor_data_flat, [], 2)';                 % [time x sensors]
N_half = floor(ceil(N/down_factor)/2) + 1;
S_fft = S_fft(1:N_half, :);                            % keep positive freq
freq_axis = linspace(0, Fs/2, N_half)';                % frequency axis (Hz)
spectra = abs(S_fft);                                  % magnitude spectra

[coeff, ~, ~, ~, ~] = pca(spectra');
filtered_signal = abs(coeff(:,1)) / max(abs(coeff(:,1)));
PCA = filtered_signal;
% % % Plot PCA spectrum
% % % hold on
% % % plot(freq_axis, PCA, 'LineWidth',2);
% % % xlabel('Frequency (Hz)');
% % % ylabel('PCA 1st Component');
% % % title('PCA Spectral Signature');
% % % grid on;

%% k-Wave Reconstruction
% dx = pixel_size_um * 1e-6;

if strcmpi(mode, '2D')
    % recon = kspaceLineRecon(squeeze(sensor_data_input)', dx, dt, c, ...
    %     'Interp', '*nearest', 'Plot', false, 'PosCond', false);
    % hilbert_amp = abs(hilbert(recon));
    % hilbert_amp = hilbert_amp / max(hilbert_amp(:));
    % 
    % %% Visualization (2D)
    % x_vec = (0:size(hilbert_amp,2)-1) * dx * 1e3;
    % z_vec = (0:size(hilbert_amp,1)-1) * dt * c * 1e3;
    % figure;
    % imagesc(x_vec, z_vec, 20*log10(hilbert_amp));
    % colormap(viridis);
    % axis equal tight off;
    % clim([-45 0]);
    % xlabel('X (mm)'); ylabel('Z (mm)');

else
    % % % %% Dimensions
    % % % Nx = size(sensor_data_input, 1);
    % % % Ny = size(sensor_data_input, 2);
    % % % Nt = size(sensor_data_input, 3);
    % % % dx = pixel_size_um * 1e-6;   % m
    % % % dz_phys = dt * c;            % m (based on time-of-flight)
    % % % z_mm = (0:Nt-1) * dz_phys * 1e3;  % mm
    % % % 
    % % % 
    % % % %% Reconstruct full volume (slice-by-slice through Y)
    % % % recon3D = zeros(Nt, Nx, Ny, 'single');  % [Z, X, Y]
    % % % 
    % % % for iy = 1:Ny
    % % %     sensor_slice = squeeze(sensor_data_input(:, iy, :));  % [X, T]
    % % %     recon_slice = kspaceLineRecon(sensor_slice', dx, dt, c, ...
    % % %         'Interp', '*nearest', 'Plot', false, 'PosCond', false);  % [Z x X]
    % % % 
    % % %     recon3D(:,:,iy) = abs(hilbert(recon_slice));  % [Z, X, Y]
    % % % end
    % % % 
    % % % recon3D = recon3D / max(recon3D(:));
    % % % recon3D = permute(recon3D, [2 3 1]);  % [X, Y, Z]
    % % % 
    % % % %% Spatial grid in mm
    % % % threshold = 0.1 * max(recon3D(:));  % Noise suppression
    % % % [x_idx, y_idx, z_idx] = ind2sub(size(recon3D), find(recon3D > threshold));
    % % % x_mm = (x_idx - 1) * dx * 1e3;
    % % % y_mm = (y_idx - 1) * dx * 1e3;
    % % % z_mm = (z_idx - 1) * dt * c * 1e3;  % Physical depth
    % % % 
    % % % 
    % % % %% Plot cylindrical isosurface
    % % % figure;
    % % % % Define cylinder mask (centered at 0,0 in mm coords)
    % % % x_c = linspace(min(x_mm), max(x_mm), 100);
    % % % y_c = linspace(min(y_mm), max(y_mm), 100);
    % % % z_c = linspace(4.0, 5.0, 20);  % Thin slice at known cylinder location
    % % % [Xc, Yc, Zc] = meshgrid(x_c, y_c, z_c);
    % % % 
    % % % Rc = sqrt((Xc - mean(x_mm)).^2 + (Yc - mean(y_mm)).^2);
    % % % cylinder_mask = Rc <= 3;  % 3 mm radius
    % % % cylinder_mask = smooth3(single(cylinder_mask), 'box', 3);
    % % % 
    % % % p1 = patch(isosurface(Xc, Yc, Zc, cylinder_mask, 0.5));
    % % % set(p1, 'FaceColor', [0.5 0.5 0.5], 'EdgeColor', 'none', 'FaceAlpha', 0.2);
    % % % hold on;
    % % % 
    % % % %% Scatter plot of reconstructed volume
    % % % mask = recon3D > threshold;
    % % % 
    % % % % Scatter colored by amplitude
    % % % scatter3(x_mm, y_mm, z_mm, 10, recon3D(recon3D > threshold), 'filled');
    % % % 
    % % % 
    % % % %% Plot settings
    % % % axis equal tight;
    % % % xlabel('X (mm)');
    % % % ylabel('Y (mm)');
    % % % zlabel('Z (mm)');
    % % % title('3D Reconstructed Volume + Cylinder');
    % % % colormap(viridis(256));
    % % % colorbar;
    % % % view(3);
    % % % lighting gouraud;
    % % % camlight;
end


end
