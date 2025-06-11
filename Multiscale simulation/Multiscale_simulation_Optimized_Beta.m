function sensor_data = Multiscale_simulation(Voxel_responses, ClotMatrix, coordinate_types)
%MULTISCALE_SIMULATION  Multiscale photo‑acoustic simulation pipeline (PARFOR version)
%
%   sensor_data = Multiscale_simulation(Voxel_responses, ClotMatrix, coordinate_types)
%
%   ------------------------------------------------------------------
%   INPUTS
%     Voxel_responses  [Nregions × Nvars × 6 × Ntime]  single/double
%     ClotMatrix       [Nregions × Nvars × Nx × Ny × Nz]  uint8
%     coordinate_types [Ncoords × 1]  uint8
%   OUTPUT
%     sensor_data      k‑Wave pressure recordings (single)
%   ------------------------------------------------------------------

%#ok<*NASGU,*AGROW>

addpath(genpath('C:\GitHub\Software_packages'));
rng(15,'twister');                     % reproducibility
Voxel_responses = single(Voxel_responses);

nVars   = size(Voxel_responses,2);     % Monte‑Carlo realisations
nTime   = size(Voxel_responses,4);

% --------------------------- CYLINDER ROI -----------------------------
voxel_sz     = 120e-6;                 % [m]
cyl_diam     = 6e-3; cyl_h = 1e-3;     % [m]
ROI_sz       = cyl_diam + 4e-3;        % lateral margin
mat_sz       = round(ROI_sz/voxel_sz); % coarse grid size (cube)

clot_mask = single(makeCylinder(mat_sz,mat_sz,mat_sz, ...
    mat_sz/2,mat_sz/2,mat_sz/2, (cyl_diam/2)/voxel_sz, cyl_h/voxel_sz,[0 0 0]));

[idx_x, idx_y, idx_z] = ind2sub(size(clot_mask), find(clot_mask==1));
Ncoords               = numel(idx_x);

% --------------------------- PRE‑ALLOCATE -----------------------------
assigned_mat = zeros(Ncoords,6,nTime,'single');          % rotated responses

abs_coeff   = single.zeros(size(clot_mask));
scat_coeff  = abs_coeff;   refrac_idx = abs_coeff;

% --------------------------- VALOMC & K‑WAVE SETUP --------------------
mm      = 1e3;                                     % m→mm scale
coords  = linspace(-mat_sz/2,mat_sz/2,mat_sz)*voxel_sz*mm;
vmcmesh = createGridMesh(coords,coords,coords);
vmcmed  = createMedium(vmcmesh);

anisotropy = single(0.9678).*ones(size(clot_mask),'single');

% fine acoustic grid (×6 in each dim)
dx = voxel_sz/6;  Nx = mat_sz*6; Ny = Nx; Nz = Nx;
kgrid = kWaveGrid(Nx,dx,Ny,dx,Nz,dx);

medium.sound_speed = single(1540).*ones(Nx,Ny,Nz,'single');
medium.density     = single(1025).*ones(Nx,Ny,Nz,'single');

% map coarse voxel indices to fine grid
[idx_fx,idx_fy,idx_fz] = resample_indices(idx_x,idx_y,idx_z,mat_sz,6*mat_sz);
subs_lin = uint32(sub2ind(size(clot_mask),idx_x,idx_y,idx_z));  % 1‑D indices

% --------------------------- LOOP CONSTANTS ---------------------------
ds_big  = 2*floor(voxel_sz     /1e-6);   % coarse → scalar
ds_fine = 2*floor((voxel_sz/6)/1e-6);    % fine   → 6³ block
sx = 6; sy = 6; sz = 6;

fprintf('PARFOR: building clot (%d voxels) …\n',Ncoords);

% ----------------------------------------------------------------------
%   PARALLEL VOXEL LOOP   (worker‑local accumulation)
% ----------------------------------------------------------------------
abs_vals   = zeros(Ncoords,1,'single');
scat_vals  = abs_vals;  refr_vals = abs_vals;

bx_list = uint16.zeros(Ncoords,1);  by_list = bx_list; bz_list = bx_list;
block_ss = cell(Ncoords,1);         block_rho = cell(Ncoords,1);

parfor ii = 1:Ncoords
    % random clot realisation & rotation -------------------------------
    rnd = randi(nVars);
    C   = squeeze(ClotMatrix(coordinate_types(ii),rnd,:,:,:));
    [ax,ang,C] = apply_random_rotation(C);

    assigned_mat(ii,:,:) = rotate_response_matrix( ...
        squeeze(Voxel_responses(coordinate_types(ii),rnd,:,:)),ax,ang); %#ok<*PFBNS>

    % optical parameters (scalar) --------------------------------------
    abs_vals (ii) = down_sample_apply_parameter(C,[0 22.48 3 3],ds_big);
    scat_vals(ii) = down_sample_apply_parameter(C,[0 72.78 72.78 72.78],ds_big);
    refr_vals(ii) = down_sample_apply_parameter(C,[1 1.4 1.4 1.4],ds_big);

    % acoustic 6×6×6 blocks -------------------------------------------
    bx = idx_fx(ii)-sx/2;  by = idx_fy(ii)-sy/2;  bz = idx_fz(ii)-sz/2;
    bx_list(ii)=bx; by_list(ii)=by; bz_list(ii)=bz;

    block_ss{ii} = down_sample_apply_parameter(C,[1540 1800 1800 1800],ds_fine);
    block_rho{ii}= down_sample_apply_parameter(C,[1025 2000 2000 2000],ds_fine);
end
fprintf('PARFOR done.  Stitching results …\n');

% ----------------------------------------------------------------------
%            MERGE WORKER OUTPUTS ON CLIENT (serial)
% ----------------------------------------------------------------------
abs_coeff  (subs_lin) = abs_vals;
scat_coeff (subs_lin) = scat_vals;
refrac_idx (subs_lin) = refr_vals;

for ii = 1:Ncoords
    bx = bx_list(ii); by = by_list(ii); bz = bz_list(ii);
    medium.sound_speed(bx+(1:6),by+(1:6),bz+(1:6)) = block_ss {ii};
    medium.density    (bx+(1:6),by+(1:6),bz+(1:6)) = block_rho{ii};
    if mod(ii,round(Ncoords/100))==0 || ii==Ncoords
        fprintf('  stitched %3.0f %%\n',100*ii/Ncoords);
    end
end
fprintf('Clot assembly complete.\n');

% --------------------------- VALOMC SOLVER ----------------------------
vmcmed.scattering_anisotropy  = repmat(double(anisotropy(:)),6,1);
vmcmed.absorption_coefficient = repmat(double(abs_coeff(:)),6,1);
vmcmed.scattering_coefficient = repmat(double(scat_coeff(:)),6,1);
vmcmed.refractive_index       = repmat(double(refrac_idx(:)),6,1);

vmcboundary = createBoundary(vmcmesh,vmcmed);
lsrc        = findBoundaries(vmcmesh,'direction',[0 0 0],[0 0 -10],5);
vmcboundary.lightsource(lsrc) = {'cosinic'};

solution = ValoMC(vmcmesh,vmcmed,vmcboundary);

% --------------------------- FLUENCE MAP ------------------------------
TR = triangulation(double(vmcmesh.H), vmcmesh.r);
[X,Y,Z] = meshgrid(coords,coords,coords);
locs    = [X(:),Y(:),Z(:)];
idx     = pointLocation(TR,locs); idx(isnan(idx)) = 1;
fluGrid = reshape(single(solution.element_fluence(idx)),size(X));
abs_E   = abs_coeff .* fluGrid * 1e3;
abs_Evoxel = abs_E .* clot_mask;

% --------------------------- K‑WAVE PREP ------------------------------
T  = 1e-5;  dt = 5e-10;
kgrid.t_array = 0:dt:T; nt = numel(kgrid.t_array);

source.u_mask = zeros(Nx,Ny,Nz,'single');
nbr = [1 0 0;-1 0 0;0 1 0;0 -1 0;0 0 1;0 0 -1];
for ii = 1:Ncoords
    cx = idx_fx(ii); cy = idx_fy(ii); cz = idx_fz(ii);
    for jj = 1:6
        xn=cx+nbr(jj,1); yn=cy+nbr(jj,2); zn=cz+nbr(jj,3);
        if xn>=1&&xn<=Nx&&yn>=1&&yn<=Ny&&zn>=1&&zn<=Nz
            source.u_mask(xn,yn,zn)=1; end
    end
end

source.ux = zeros(6*Ncoords,nt,'single');
source.uy = source.ux; source.uz = source.ux;
for ii = 1:Ncoords
    resp = abs_Evoxel(idx_x(ii),idx_y(ii),idx_z(ii)) .* squeeze(assigned_mat(ii,:,:));
    row  = 6*(ii-1);
    rho  = medium.density    (idx_fx(ii),idx_fy(ii),idx_fz(ii));
    c    = medium.sound_speed(idx_fx(ii),idx_fy(ii),idx_fz(ii));
    s    = single(1)/(rho*c);
    source.ux(row+1,:) = +s*resp(1,:);  source.ux(row+2,:) = -s*resp(2,:);
    source.uy(row+3,:) = +s*resp(3,:);  source.uy(row+4,:) = -s*resp(4,:);
    source.uz(row+5,:) = +s*resp(5,:);  source.uz(row+6,:) = -s*resp(6,:);
end

% --------------------------- SENSOR PLANE -----------------------------
sensor.mask = single.zeros(Nx,Ny,Nz,'single');
cent = [Nx/2,Ny/2,0.9*Nz];
[y_s,x_s] = meshgrid(linspace(-Ny/2,Ny/2,Ny), linspace(-Nx/2,Nx/2,Nx));
pts = [x_s(:), y_s(:), zeros(numel(x_s),1)]';   % plane @ z = 0

R_y = eye(3);             % sensor_angle = 0°, so no rotation
pts = R_y*pts + cent';
rx=round(pts(1,:)); ry=round(pts(2,:)); rz=round(pts(3,:));
valid = rx>0 & rx<=Nx & ry>0 & ry<=Ny & rz>0 & rz<=Nz;
idx   = sub2ind(size(sensor.mask),rx(valid),ry(valid),rz(valid));
sensor.mask(idx) = 1;

% --------------------------- K‑WAVE RUN -------------------------------
clearvars -except kgrid medium source sensor
sensor_data = single(kspaceFirstOrder3DG(kgrid,medium,source,sensor,'PMLInside',false));
end

%====================== HELPER FUNCTIONS ===============================
function [fx,fy,fz] = resample_indices(x,y,z,oldSz,newSz)
    s = newSz/oldSz;
    fx = max(min(round((x-0.5)*s+0.5),newSz),1);
    fy = max(min(round((y-0.5)*s+0.5),newSz),1);
    fz = max(min(round((z-0.5)*s+0.5),newSz),1);
end

function [ax,angle,R] = apply_random_rotation(A)
    ax = randi(3); k = randi(4)-1; angle = 90*k;
    switch ax
        case 1, R = rot90(A,k,[2 3]);
        case 2, R = rot90(A,k,[1 3]);
        case 3, R = rot90(A,k,[1 2]);
    end
end

function M = down_sample_apply_parameter(C,vals,ds)
    Mparam = single(vals); Mparam = Mparam(C+1);
    sz  = size(Mparam)./ds;
    tmp = reshape(Mparam,ds,sz(1),ds,sz(2),ds,sz(3));
    M   = squeeze(mean(mean(mean(tmp,1),3),5));
    M   = permute(M,[2 4 6]);
end

function R = rotate_response_matrix(resp,ax,angle)
    map = struct('x_90',[1 2 5 6 4 3],'x_180',[1 2 4 3 6 5],'x_270',[1 2 6 5 3 4], ...
                 'y_90',[6 5 3 4 1 2],'y_180',[2 1 3 4 6 5],'y_270',[5 6 3 4 2 1], ...
                 'z_90',[3 4 2 1 5 6],'z_180',[2 1 4 3 5 6],'z_270',[4 3 1 2 5 6]);
    key = sprintf('%c_%d','x'+(ax-1),angle);
    R   = resp(map.(key),:);
end
