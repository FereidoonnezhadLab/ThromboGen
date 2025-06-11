addpath(genpath('C:\GitHub\Software_packages'))
% clear
% load('catagorical_data.mat','averaged_responses_downsampled_III','ClotMatrix_III');
% sensor_data_III = Multiscale_simulation(averaged_responses_downsampled_III,ClotMatrix_III);
% save catagorical_data.mat sensor_data_III -append
% clear
% load('catagorical_data.mat','ClotMatrix_IV','averaged_responses_downsampled_IV');
% sensor_data_IV_2  = Multiscale_simulation(averaged_responses_downsampled_IV,ClotMatrix_IV);
% save catagorical_data.mat sensor_data_IV_2 -append
% clear
% load('catagorical_data.mat','averaged_responses_downsampled_V','ClotMatrix_V');
% sensor_data_V = Multiscale_simulation(averaged_responses_downsampled_V,ClotMatrix_V);
% save catagorical_data.mat sensor_data_V -append
% clear
% load('catagorical_data.mat','averaged_responses_downsampled_VI','ClotMatrix_VI');
% sensor_data_VI = Multiscale_simulation(averaged_responses_downsampled_VI,ClotMatrix_VI);
% save catagorical_data.mat sensor_data_VI -append
clear
load('catagorical_data.mat','ClotMatrix_VII','averaged_responses_downsampled_VII');
sensor_data_VII_3 = Multiscale_simulation(averaged_responses_downsampled_VII,ClotMatrix_VII);
save catagorical_data.mat sensor_data_VII_3 -append


