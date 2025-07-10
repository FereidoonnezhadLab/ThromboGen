function sensor_data = Multiscale_simulation(Voxel_responses, ClotMatrix, coordinate_types, seed, mode)
addpath(genpath('C:\GitHub\Software_packages'));

% Set default if coordinate_types not provided
if isempty(coordinate_types)
    if ndims(Voxel_responses) == 3
        Nx = size(ClotMatrix, 2); Ny = size(ClotMatrix, 3); Nz = size(ClotMatrix, 4);
        coordinate_types = ones(Nx * Ny * Nz, 1, 'uint8');
    else
        error('coordinate_types is required when multiple regions exist.');
    end
end

rng(seed,'twister'); % for reproducibility

% Expand input dimensions if needed
if ndims(Voxel_responses) == 3
    Voxel_responses = reshape(Voxel_responses, 1, size(Voxel_responses,1), size(Voxel_responses,2), size(Voxel_responses,3));
end
if ndims(ClotMatrix) == 4
    ClotMatrix = reshape(ClotMatrix, 1, size(ClotMatrix,1), size(ClotMatrix,2), size(ClotMatrix,3), size(ClotMatrix,4));
end
Voxel_responses = single(Voxel_responses);

% Initialization
nVars = size(Voxel_responses,2);
nTime = size(Voxel_responses,4);
cut_idx = 1:round(2*nTime/3);  % keep only the first 2/3
nTime = numel(cut_idx);

% --- Geometry setup
voxel_sz = 72e-6;
cyl_diam = 6e-3; cyl_h = 1e-3;
ROI_sz = cyl_diam + 4e-3;  %4e-3
mat_sz = round(ROI_sz/voxel_sz);
clot_mask = uint8(makeCylinder(mat_sz,mat_sz,mat_sz, ...
    mat_sz/2,mat_sz/2,mat_sz/2, (cyl_diam/2)/voxel_sz, cyl_h/voxel_sz,[0 0 0]));

[idx_x, idx_y, idx_z] = ind2sub(size(clot_mask), find(clot_mask==1));
Ncoords = numel(idx_x);

% --- Preallocate matrices
assigned_mat = zeros(Ncoords,6,nTime,'single');
abs_coeff = zeros(size(clot_mask));
scat_coeff = abs_coeff; refrac_idx = ones(size(clot_mask));

% Optical mesh and acoustic grid
mm = 1e3; coords = linspace(-mat_sz/2,mat_sz/2,mat_sz)*voxel_sz*mm;
vmcmesh = createGridMesh(coords,coords,coords);
vmcmed = createMedium(vmcmesh);
anisotropy = single(0.9678).*ones(size(clot_mask),'single');
kwave_mesh_ratio = 2;
dx = voxel_sz/kwave_mesh_ratio; Nx = mat_sz*kwave_mesh_ratio; Ny = Nx; Nz = Nx;
kgrid = kWaveGrid(Nx,dx,Ny,dx,Nz,dx);

% Shared time array
T = 1e-5; dt = 5e-10; t_array = 0:dt:T;
nt_full = numel(t_array);
cut_idx = 1:round(2*nt_full/3);  % keep only the first 2/3
nt = numel(cut_idx);
kgrid.t_array = t_array(cut_idx);

medium.sound_speed = single(1540).*ones(Nx,Ny,Nz,'single');
medium.density = single(1025).*ones(Nx,Ny,Nz,'single');

[idx_fx,idx_fy,idx_fz] = resample_indices(idx_x,idx_y,idx_z,mat_sz,kwave_mesh_ratio*mat_sz);
subs_lin = uint32(sub2ind(size(clot_mask),idx_x,idx_y,idx_z));

% --- PARFOR loop
abs_vals = zeros(Ncoords,1,'single'); scat_vals = abs_vals; refr_vals = abs_vals;
bx_list = zeros(Ncoords,1); by_list = bx_list; bz_list = bx_list;
block_ss = cell(Ncoords,1); block_rho = cell(Ncoords,1);
rnd = randi(nVars);

fprintf('PARFOR: building clot (%d voxels) …\n',Ncoords);

progress = 0;
q = parallel.pool.DataQueue;
afterEach(q, @(~) updateProgress());

    function updateProgress()
        progress = progress + 1;
        if mod(progress, round(Ncoords/100)) == 0 || progress == Ncoords
            fprintf('  voxel %d / %d (%.0f%%)\n', progress, Ncoords, 100 * progress / Ncoords);
        end
    end

parfor ii = 1:Ncoords
    C = squeeze(ClotMatrix(coordinate_types(ii),rnd,:,:,:));
    assigned_mat(ii,:,:) = squeeze(Voxel_responses(coordinate_types(ii),rnd,:,cut_idx)); %#ok<*PFBNS>
    abs_vals(ii) = down_sample_apply_parameter(C,0.06,23,0.005,0.5,2*floor(voxel_sz/1e-6));
    scat_vals(ii) = down_sample_apply_parameter(C,0.125,85,0.06,0.125,2*floor(voxel_sz/1e-6));
    refr_vals(ii) = down_sample_apply_parameter(C,1.345,1.395,1.575,1.375,2*floor(voxel_sz/1e-6));

    bx = idx_fx(ii)-floor(kwave_mesh_ratio/2); by = idx_fy(ii)-floor(kwave_mesh_ratio/2); bz = idx_fz(ii)-floor(kwave_mesh_ratio/2);
    bx_list(ii)=bx; by_list(ii)=by; bz_list(ii)=bz;

    block_ss{ii} = down_sample_apply_parameter(C,1525,1570,1565.5,1540,2*floor(dx/1e-6));
    block_rho{ii}= down_sample_apply_parameter(C,1027.5,1125,1060,1080,2*floor(dx/1e-6));
    % progress update
    send(q, ii);
end
delete(gcp('nocreate'))  % if using parpool and memory isn’t released
fprintf('PARFOR done.  Stitching results …\n');
abs_coeff(subs_lin) = abs_vals; scat_coeff(subs_lin) = scat_vals; refrac_idx(subs_lin) = refr_vals;

for ii = 1:Ncoords
    bx = bx_list(ii); by = by_list(ii); bz = bz_list(ii);
    medium.sound_speed(bx+(1:kwave_mesh_ratio),by+(1:kwave_mesh_ratio),bz+(1:kwave_mesh_ratio)) = block_ss{ii};
    medium.density    (bx+(1:kwave_mesh_ratio),by+(1:kwave_mesh_ratio),bz+(1:kwave_mesh_ratio)) = block_rho{ii};
    if mod(ii,round(Ncoords/100))==0 || ii==Ncoords
        fprintf('  stitched %3.0f %%\n',100*ii/Ncoords);
    end
end

% --- Fluence and gradients
vmcmed.scattering_anisotropy = repmat(double(anisotropy(:)),6,1);
vmcmed.absorption_coefficient = repmat(double(abs_coeff(:)),6,1);
vmcmed.scattering_coefficient = repmat(double(scat_coeff(:)),6,1);
vmcmed.refractive_index = repmat(double(refrac_idx(:)),6,1);
vmcboundary = createBoundary(vmcmesh,vmcmed);
lsrc = findBoundaries(vmcmesh,'direction',[0 0 -10],[0 0 0],5);
vmcboundary.lightsource(lsrc) = {'cosinic'};

solution = ValoMC(vmcmesh,vmcmed,vmcboundary);
TR = triangulation(double(vmcmesh.H), vmcmesh.r);
[X,Y,Z] = meshgrid(coords,coords,coords);
idx = pointLocation(TR,[X(:),Y(:),Z(:)]); idx(isnan(idx)) = 1;
fluGrid = reshape(single(solution.element_fluence(idx)),size(X));
abs_E = abs_coeff .* fluGrid * 1e3;
abs_Evoxel = abs_E .* single(clot_mask);

% Select simulation mode
if strcmpi(mode, 'simplistic')
    %% ---------------- SIMPLISTIC MODE ONLY ----------------

    % Generate initial pressure field
    source_s.p0 = zeros(Nx,Ny,Nz,'single');
    for ii = 1:Ncoords
        source_s.p0(idx_fx(ii), idx_fy(ii), idx_fz(ii)) = abs_Evoxel(idx_x(ii), idx_y(ii), idx_z(ii));
    end
    source_s.p_mask = source_s.p0 > 0;

    % Define sensor
    sensor.mask = zeros(Nx,Ny,Nz,'single');
    cent = [Nx/2,Ny/2,0.9*Nz];
    [y_s,x_s] = meshgrid(linspace(-Ny/2,Ny/2,Ny), linspace(-Nx/2,Nx/2,Nx));
    pts = [x_s(:), y_s(:), zeros(numel(x_s),1)]';
    pts = eye(3)*pts + cent';
    rx=round(pts(1,:)); ry=round(pts(2,:)); rz=round(pts(3,:));
    valid = rx>0 & rx<=Nx & ry>0 & ry<=Ny & rz>0 & rz<=Nz;
    idx = sub2ind(size(sensor.mask),rx(valid),ry(valid),rz(valid));
    sensor.mask(idx) = 1;

    % Run simulation
    sensor_data = single(kspaceFirstOrder3DG(kgrid,medium,source_s,sensor,'PMLInside',false));

    return  % Exit function early if only simplistic
end

if strcmpi(mode, 'multiscale')
    % ---------------- Multiscale Mode ----------------
    [grad_x, grad_y, grad_z] = gradient(single(abs_Evoxel), dx, dx, dx);
    grad_mag = sqrt(grad_x.^2 + grad_y.^2 + grad_z.^2) + eps;
    grad_x = grad_x ./ grad_mag;
    grad_y = grad_y ./ grad_mag;
    grad_z = grad_z ./ grad_mag;

    % Allocate space for 2 sources per voxel
    source_m.ux = zeros(2*Ncoords, nt, 'single');
    source_m.uy = zeros(2*Ncoords, nt, 'single');
    source_m.uz = zeros(2*Ncoords, nt, 'single');
    source_m.u_mask = zeros(Nx, Ny, Nz, 'single');

    for ii = 1:Ncoords
        % Gradient vector at voxel
        gx = grad_x(idx_x(ii), idx_y(ii), idx_z(ii));
        gy = grad_y(idx_x(ii), idx_y(ii), idx_z(ii));
        gz = grad_z(idx_x(ii), idx_y(ii), idx_z(ii));

        % Weight 6-directional response based on gradient
        dir_weight = [max(0,+gx); max(0,-gx); max(0,+gy); max(0,-gy); max(0,+gz); max(0,-gz)];
        dir_weight = dir_weight / sum(dir_weight + eps);
        raw_resp = squeeze(assigned_mat(ii,:,:));
        weighted_resp = dir_weight .* raw_resp;
        resp = abs_Evoxel(idx_x(ii), idx_y(ii), idx_z(ii)) .* weighted_resp;

        % Acoustic scaling
        rho = medium.density(idx_fx(ii), idx_fy(ii), idx_fz(ii));
        c   = medium.sound_speed(idx_fx(ii), idx_fy(ii), idx_fz(ii));
        s   = single(1) / (rho * c);

        % Assign +direction composite to source 1, -direction composite to source 2
        row1 = 2*(ii-1) + 1;
        row2 = row1 + 1;

        % Source 1: positive direction components
        source_m.ux(row1,:) = +s * resp(1,:);
        source_m.uy(row1,:) = +s * resp(3,:);
        source_m.uz(row1,:) = +s * resp(5,:);

        % Source 2: negative direction components (with negative sign)
        source_m.ux(row2,:) = -s * resp(2,:);
        source_m.uy(row2,:) = -s * resp(4,:);
        source_m.uz(row2,:) = -s * resp(6,:);

        % Set two spatially distinct points in the mask
        cx = idx_fx(ii);
        cy = idx_fy(ii);
        cz = idx_fz(ii);

        % Source 1 at current voxel
        source_m.u_mask(cx, cy, cz) = 1;

        % Source 2 at offset voxel (+1,+1,+1), bounded
        cx2 = min(cx+1, Nx);
        cy2 = min(cy+1, Ny);
        cz2 = min(cz+1, Nz);
        source_m.u_mask(cx2, cy2, cz2) = 1;
    end

    % Define sensor on a horizontal 2D grid near the top
    sensor.mask = zeros(Nx, Ny, Nz, 'single');
    cent = [Nx/2, Ny/2, 0.9 * Nz];
    [y_s, x_s] = meshgrid(linspace(-Ny/2, Ny/2, Ny), linspace(-Nx/2, Nx/2, Nx));
    pts = [x_s(:), y_s(:), zeros(numel(x_s), 1)]';
    pts = eye(3) * pts + cent';
    rx = round(pts(1,:));
    ry = round(pts(2,:));
    rz = round(pts(3,:));
    valid = rx > 0 & rx <= Nx & ry > 0 & ry <= Ny & rz > 0 & rz <= Nz;
    idx = sub2ind(size(sensor.mask), rx(valid), ry(valid), rz(valid));
    sensor.mask(idx) = 1;

    % Run simulation
    sensor_data = single(kspaceFirstOrder3DG(kgrid, medium, source_m, sensor, 'PMLInside', false));
    return
end

error('Invalid mode. Use ''simplistic'' or ''multiscale''.');
end

%====================== HELPER FUNCTIONS ===============================
function [fx,fy,fz] = resample_indices(x,y,z,oldSz,newSz)
s = newSz/oldSz;
fx = max(min(round((x-0.5)*s+0.5),newSz),1);
fy = max(min(round((y-0.5)*s+0.5),newSz),1);
fz = max(min(round((z-0.5)*s+0.5),newSz),1);
end

function DownsampledMatrix = down_sample_apply_parameter(ClotMatrix, ...
    param_empty, param_rbc, param_fibrin, param_platelet, downsample_factor)
% 1) Replace codes in ClotMatrix with corresponding parameter values
% 2) Downsample by averaging blocks of size (downsample_factor^3)

% Step 1: Map codes to values using lookup
lut = single(zeros(1, max(ClotMatrix(:))+1));
lut(1) = param_empty;
lut(2) = param_rbc;
lut(3) = param_fibrin;
lut(4) = param_platelet;
ParameterMatrix = lut(double(ClotMatrix)+1);

% Step 2: Crop to size divisible by downsample_factor
[rows, cols, slices] = size(ParameterMatrix);
rows_ds   = floor(rows   / downsample_factor);
cols_ds   = floor(cols   / downsample_factor);
slices_ds = floor(slices / downsample_factor);

ParameterMatrix = ParameterMatrix( ...
    1 : rows_ds * downsample_factor, ...
    1 : cols_ds * downsample_factor, ...
    1 : slices_ds * downsample_factor);

% Step 3: Reshape into 5D for fast mean over blocks
ParameterMatrix = reshape(ParameterMatrix, ...
    downsample_factor, rows_ds, ...
    downsample_factor, cols_ds, ...
    downsample_factor, slices_ds);

% Permute to [down_x, down_y, down_z, rows_ds, cols_ds, slices_ds]
ParameterMatrix = permute(ParameterMatrix, [1 3 5 2 4 6]);

% Step 4: Compute mean over the first 3 dims (block content)
DownsampledMatrix = squeeze(mean(ParameterMatrix, [1 2 3]));

end

