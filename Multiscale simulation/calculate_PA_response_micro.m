function sensor_data = calculate_PA_response_micro(matrix)
Fluence_rate = ones(size(matrix));
scattering_coefficients = 0*ones(size(matrix));
absorption_coefficients = 0*ones(size(matrix));
scattering_anisotropies = 0.9678*ones(size(matrix));
refractive_indexes = 1.0*ones(size(matrix));
absorption_coefficients(matrix == 1) = 22.48;
scattering_coefficients(matrix == 1) = 72.78;
absorption_coefficients(matrix == 2) = 5; %% dummmy value
scattering_coefficients(matrix == 2) = 72.78; %% dummmy value
absorption_coefficients(matrix == 3) = 5; %% dummmy value
scattering_coefficients(matrix == 3) = 72.78; %% dummmy value
refractive_indexes(matrix == 1) = 1.4;
refractive_indexes(matrix == 2) = 1.4;
refractive_indexes(matrix == 3) = 1.4;
absorbed_energy = Fluence_rate.*absorption_coefficients;
Voxel_size = 240; %in um
x_arr = linspace(round(-Voxel_size/2),round(Voxel_size/2),size(absorbed_energy,1));
y_arr = linspace(round(-Voxel_size/2),round(Voxel_size/2),size(absorbed_energy,2));
z_arr = linspace(round(-Voxel_size/2),round(Voxel_size/2),size(absorbed_energy,3));
[X,Y,Z] = meshgrid(y_arr,x_arr,z_arr); % Matlab function
xarr_q = linspace(min(X(:)),max(X(:)),size(X,1)/2);
Yarr_q = linspace(min(Y(:)),max(Y(:)),size(Y,1)/2);
Zarr_q = linspace(min(Z(:)),max(Z(:)),size(Z,1)/2);
[Xq,Yq,Zq]=meshgrid(xarr_q,Yarr_q,Zarr_q);
absorbed_energy = interp3(X,Y,Z,double(absorbed_energy),Xq,Yq,Zq,"linear");
matrix = interp3(X,Y,Z,double(matrix),Xq,Yq,Zq,"linear");
Voxel_size = 120; %in um
x_arr = linspace(round(-Voxel_size/2),round(Voxel_size/2),size(absorbed_energy,1));
y_arr = linspace(round(-Voxel_size/2),round(Voxel_size/2),size(absorbed_energy,2));
z_arr = linspace(round(-Voxel_size/2),round(Voxel_size/2),size(absorbed_energy,3));
[X,Y,Z] = meshgrid(y_arr,x_arr,z_arr); % Matlab function
dx = 1e-6*Voxel_size/size(absorbed_energy,1);
dy = 1e-6*Voxel_size/size(absorbed_energy,2);
dz = 1e-6*Voxel_size/size(absorbed_energy,3);
Nx = size(absorbed_energy,1);
Ny = size(absorbed_energy,2);
Nz = size(absorbed_energy,3);
kgrid = kWaveGrid(Nx, dx, Ny, dy,Nz, dz);
gruneisen_parameter = 0.16;
source.p0 = gruneisen_parameter .* absorbed_energy;  % [Pa]
medium.sound_speed = 1540*ones(Nx,Ny,Nz);    % [m/s]
medium.sound_speed(matrix == 1) = 2380;   % [m/s]
medium.sound_speed(matrix == 2) = 2380;   % [m/s]
medium.sound_speed(matrix == 3) = 2380;   % [m/s]
medium.density = 1025*ones(Nx, Ny,Nz);        % [kg/m^3]
medium.density(matrix == 1) = 2000;
medium.density(matrix == 2) = 2000;
medium.density(matrix == 3) = 2000;
T = 5e-7;
dt = 5e-11;
T_array = 0:dt:T;
kgrid.t_array = T_array;
sensor.mask = zeros(Nx, Ny, Nz);
sensor.mask(1, :, :) = 1;             % Front face
sensor.mask(end, :, :) = 1;           % Back face
sensor.mask(:, 1, :) = 1;             % Left face
sensor.mask(:, end, :) = 1;           % Right face
sensor.mask(:, :, 1) = 1;             % Bottom face
sensor.mask(:, :, end) = 1;           % Top face
sensor_data = kspaceFirstOrder3DG(kgrid, medium, source, sensor, 'PMLInside', false);
end




