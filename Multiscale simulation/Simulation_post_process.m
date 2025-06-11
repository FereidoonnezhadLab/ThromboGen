function [] = Simulation_post_process(sensor_data)
addpath(genpath('C:\GitHub\Software_packages'))
%% load sensor_data
% load sensor_data_90_comp.mat sensor_data
% load sensor_data_70_comp.mat sensor_data
Fs = 1 / (5e-10);
% Fs = 62.5e6;
% [p, q] = rat(Fs / Fs_init);  % get integer ratio
% % Resample each row
% for i = 1:size(sensor_data,1)
%     sensor_data_resampled(i,:) = resample(sensor_data(i,:), p, q);
%     disp(i);
% end
N = size(sensor_data,2);          % Length of each time sequence
dimension = round(sqrt(size(sensor_data,1)));
% sensor_data_3D = reshape(sensor_data(1:dimension*dimension,:), [dimension,dimension,N]);
% sensor_data_3D = single(sensor_data_3D);

%% INegration Over axa Regions Instead of Simple Downsampling
% % pixel_size_um = 20;        % micrometers per pixel
% % detector_size_um = 280;    % physical detector diameter (µm)
% % step_size_um = 300;        % physical spacing between detectors (µm)
% % 
% % detector_size_pix = round(detector_size_um / pixel_size_um);  %  14
% % step_size_pix = round(step_size_um / pixel_size_um);          %  15
% % 
% % Nx = size(sensor_data_3D,1);
% % Ny = size(sensor_data_3D,2);
% % N = size(sensor_data_3D,3);
% % 
% % x_positions = 1:step_size_pix:(Nx - detector_size_pix + 1);
% % y_positions = 1:step_size_pix:(Ny - detector_size_pix + 1);
% % 
% % down_Nx = length(x_positions);
% % down_Ny = length(y_positions);
% % 
% % sensor_data_integrated = zeros(down_Nx, down_Ny, N, 'single');
% % 
% % % Create Gaussian weighting kernel (PSF)
% % sigma = detector_size_pix / 3;  % rule of thumb
% % gaussian_kernel = fspecial('gaussian', [detector_size_pix detector_size_pix], sigma);
% % gaussian_kernel = gaussian_kernel / sum(gaussian_kernel(:));  % normalize
% % 
% % % Loop over time frames and integrate with Gaussian weighting
% % for t = 1:N
% %     frame = sensor_data_3D(:,:,t);
% %     for ix = 1:down_Nx
% %         for iy = 1:down_Ny
% %             x_start = x_positions(ix);
% %             y_start = y_positions(iy);
% %             x_end = x_start + detector_size_pix - 1;
% %             y_end = y_start + detector_size_pix - 1;
% %             block = frame(x_start:x_end, y_start:y_end);
% %             weighted_sum = sum(block(:) .* gaussian_kernel(:));
% %             sensor_data_integrated(ix, iy, t) = weighted_sum;
% %         end
% %     end
% % end


% choosing the middle row in a 3D matrix to have a 2D image
% sensor_data_down_sampled = sensor_data_integrated(round(end/2),:,:);

%% PCA Spectral Response

% sensor_data = reshape(sensor_data_down_sampled,[],N);

S_fft = fft(sensor_data, [], 2)';

% Take only the positive frequency part.
N_half = floor(N/2) + 1;
S_fft = S_fft(1:N_half, :);

% Create the positive frequency axis.
freq_axis = linspace(0, Fs/2, N_half)';

% Compute the magnitude spectrum.
spectra = abs(S_fft);


%% Apply Frequency Domain Filtering (filter from datasheet)
% % frequency_hz = 1e6*[0.000, 0.408, 0.816, 1.224, 1.633, 2.041, 2.449, 2.857, 3.265, 3.673, ...
% %                  4.082, 4.490, 4.898, 5.306, 5.714, 6.122, 6.531, 6.939, 7.347, 7.755, ...
% %                  8.163, 8.571, 8.980, 9.388, 9.796, 10.204, 10.612, 11.020, 11.429, 11.837, ...
% %                  12.245, 12.653, 13.061, 13.469, 13.878, 14.286, 14.694, 15.102, 15.510, 15.918, ...
% %                  16.327, 16.735, 17.143, 17.551, 17.959, 18.367, 18.776, 19.184, 19.592, 20.000];
% %
% % magnitude_db = [-100.000, -100.000, -100.000, -100.000, -100.000, -57.718, -46.293, -42.879, -41.708, -42.825,  ...
% %                 -40.751, -32.577, -24.366, -20.433, -19.644, -19.793, -19.964, -19.916, -19.914, -20.259, ...
% %                 -20.726, -21.121, -21.559, -21.968, -22.502, -23.552, -24.787, -27.261, -30.426, -33.788, ...
% %                 -37.457, -40.926, -44.037, -46.681, -49.503, -51.460, -53.564, -55.725, -100.000, -100.000, ...
% %                 -100.000, -100.000, -100.000, -100.000, -100.000, -100.000, -100.000, -100.000, -100.000, -100.000];
% % % Convert to linear scale
% % magnitude_linear = 10.^(magnitude_db / 20);
% load filter_response.mat
% Define frequency vector for positive side only
% % % f_positive = Fs * (0:(N/2)) / N;
% % % 
% % % % INerpolate the filter
% % % filter_half = interp1(frequency_hz-1e6, magnitude_linear, f_positive, 'linear', 0);
% % % 
% % % for i = 1:size(spectra,2)
% % %     signal = squeeze(spectra(:,i));
% % %     spectra_filtered(i,:) =  signal .*filter_half(:);
% % % end

% Perform PCA on the spectra.
% Each sensor (column) is treated as an observation, and each frequency bin (row) as a variable.
% Transpose the spectra so that rows correspond to sensors.
[coeff, score, latent, tsquared, explained] = pca(spectra');
filtered_signal = abs(coeff(:,1))/max(abs(coeff(:,1)));

% Plot the first principal componeN (represeNative spectral profile) vs. frequency.
% figure
hold on
plot(freq_axis, movmean(filtered_signal ,5)/max(movmean(filtered_signal ,5)), 'LineWidth',2);
xlabel('Frequency (Hz)');
ylabel('PCA Coefficient (1st Component)');
grid on;

end

%% KWave reconstruction
% Update Grid Parameters
% % % % dx = 20e-6 * block_size;  % Updated Grid spacing (m) after integration
% % % % c = 1540;  % Speed of sound (m/s)
% % % % dt = 5e-10;  % Time step (s)
% % % %
% % % % % Run k-Wave Reconstruction
% % % % recon = kspaceLineRecon(squeeze(sensor_data_down_sampled)', dx, dt, c,'Interp', '*nearest', 'Plot', false, 'PosCond', false);
% % % % hilbert_amp = abs(hilbert(recon));
% % % % hilbert_amp = hilbert_amp / max(hilbert_amp(:));
% % % %
% % % % % Define Real-World Coordinates (in mm)
% % % % x_vec = (0:down_Nx-1) * dx * 1e3;
% % % % y_vec = (0:down_Ny-1) * dx * 1e3;
% % % % z_vec = (0:N-1) * dt * c * 1e3;
% % % %
% % % % figure
% % % % axis equal;
% % % % axis tight
% % % % axis off
% % % % imagesc(x_vec, z_vec, 20*log10(hilbert_amp));
% % % % colormap hot;
% % % % xlabel('X (mm)'); ylabel('Z (mm)');
% % % % clim([-25 0]);
