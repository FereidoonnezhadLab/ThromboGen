function generate_clot_microstructure_voxels(shell_index, voxel_size, resolution, P)
% Multiscale clot generator → per-macro-voxel fine 3D matrices (.mat)
% Labels: 0 background, 1 RBC, 2 fibrin, 3 platelet
try
    delete(gcp('nocreate'));
    clust = parcluster('Processes');
    delete(clust.Jobs);
catch
end
maxNumCompThreads(1);
fprintf('✓ Parallel cache cleared and MATLAB set to single-thread mode.\n');
%% Inputs
% shell_index : which shell OBJ to use (as before)
% voxel_size : macro voxel edge [µm], default 100
% resolution : fine voxel step [µm], default 0.5
% P : struct of overrides (same as your script)
%% ---------- HARD REQS ----------
req = {'mex_inpolyhedron','knn_grid_cuda','mex_forces_log_normal','mex_angle_forces','rbc_pack_cuda'};
if gpuDeviceCount==0, error('GPURequired:NoGPU','CUDA GPU required.'); end
missing = req(~cellfun(@(f) exist(f,'file')==3, req));
if ~isempty(missing), error('GPURequired:MissingMEX','Missing MEX: %s', strjoin(missing,', ')); end
try
    g = gpuDevice;
    fprintf('Using GPU: %s (%0.1f GB total)\n', g.Name, g.TotalMemory/1e9);
catch ME
    error('No usable CUDA GPU found or driver issue: %s', ME.message);
end
%% ---------- INPUTS / DEFAULTS ----------
if nargin<1 || isempty(shell_index), shell_index= 1; end
if nargin<2 || isempty(voxel_size), voxel_size = 100; end % µm
if nargin<3 || isempty(resolution), resolution = 0.5; end % µm
if nargin<4, P = struct; end
rng(shell_index,'twister');
%% ---- PARAMETERS (overridable via P) ----
numSpheres = getP(P,'numSpheres',randi(10000));
platelet_ratio = getP(P,'platelet_ratio',rand());
fibrin_concentration = getP(P,'fibrin_concentration',0.2+2.3*rand()); % g/L
max_r = getP(P,'max_inclusion_radius',60); % largest inclusion radius (~8–10 RBCs)
min_r = getP(P,'min_inclusion_radius',10); % % smallest inclusion radius (~1–2 RBCs)
% Empirical mappings
C1=0.4; L1=4.87; Z1=3.19; v1=0.339;
C2=1.6; L2=2.99; Z2=3.33; v2=0.341;
% Geometry details
fibrin_radius_um = 0.2/2;
tube_segments = 8;
curve_samples = 20;
platelet_radius_um = 1.5;
% Directional preference setup
preferred_direction = [1, 1, 1] / norm([1, 1, 1]);
direction_weight = 0; % Set to >0 to add directional alignment
%% ---------- LOAD SHELL + UTILS (same as your current script) ----------
shell_dir = '/home/hghodsi/Matlab_codes/Clot_generator/Thrombi_surface_meshes';
filesList = dir(fullfile(shell_dir,'mesh_*.obj'));
assert(~isempty(filesList),'No OBJ files found.');
assert(shell_index>=1 && shell_index<=numel(filesList),'shell_index out of range');
obj_path = fullfile(shell_dir, filesList(shell_index).name);
[SV,SF] = readOBJ_local(obj_path);
SV = SV * 1000; % mm -> µm
mesh_name = erase(filesList(shell_index).name,'.obj');
% In-shell test
isInsideShell = @(Q) logical(mex_inpolyhedron(SF,SV,ensureNx3double(Q)));
%% ---- BONDING BOX ----
bbmin = min(SV,[],1);
bbmax = max(SV,[],1);
extent = bbmax - bbmin;
%% ---- SECTION 1: Generate inclusions (heterogeneous distribution) ----
% Define physical inclusion radius limits (µm)
% Log-normal sampling for inclusion sizes
r_median = median([min_r max_r]); % µm
r_sigma = 0.35; % log-normal spread (dimensionless)
r_samples = lognrnd(log(r_median), r_sigma, [numSpheres,1]);
r_samples = min(max(r_samples, min_r), max_r); % clamp to limits
sphere_centers = []; sphere_radii = []; max_tries_per = 300;
% Choose heterogeneity mode: 'radial' or 'axial'
hetero_mode = 'radial'; % options: 'radial', 'axial', or 'none'
% Control gradient intensity (0=uniform, 1=strong bias)
hetero_strength = 0.8;
% Precompute shell geometry
center_shell = mean([bbmin; bbmax],1);
extent = bbmax - bbmin;
radius_shell = norm(extent)/2;
for i = 1:numSpheres
    ok = false; t = 0;
    while ~ok && t < max_tries_per
        t = t + 1;
        % --- Sample position according to heterogeneity type ---
        switch hetero_mode
            case 'radial'
                % bias inclusions toward shell center or edge
                u = rand();
                % use power-law bias: smaller exponent → near center, larger → periphery
                r_bias = radius_shell * (u^(1 - hetero_strength));
                dirr = randn(1,3); dirr = dirr / norm(dirr);
                c = center_shell + dirr * r_bias;
            case 'axial'
                % bias inclusions along one axis (e.g., z)
                c = bbmin + rand(1,3).*extent;
                z_rel = (c(3) - bbmin(3)) / extent(3); % normalized [0,1]
                if rand() > (1 - hetero_strength*z_rel)
                    % cluster toward high-z side
                    c(3) = bbmin(3) + extent(3)*z_rel.^0.5;
                else
                    % some uniform spread
                    c(3) = bbmin(3) + rand()*extent(3);
                end
            otherwise
                % uniform fallback
                c = bbmin + rand(1,3).*extent;
        end % --- Continue with standard validity checks ---
        r = r_samples(i);
        if ~isInsideShell(c), continue; end
        if ~isempty(sphere_centers)
            if any(vecnorm(sphere_centers - c, 2, 2) < (sphere_radii + r)), continue; end
        end
        ax = eye(3);
        if ~all(isInsideShell(c + r*ax)) || ~all(isInsideShell(c - r*ax)), continue; end
        % Accept inclusion
        sphere_centers = [sphere_centers; c];
        sphere_radii = [sphere_radii; r];
        ok = true;
    end
    if ~ok, break; end
end
fprintf("section-1 done\n")
%% ---- SECTION 2: Effective volume & length budget ----
shell_vol_est = polyVolume(SV,SF);
inc_vol = sum((4/3)*pi*(sphere_radii.^3));
eff_mult = 0.8 + 1.2*rand();
fibrin_density_g_um3 = 1.395e-12; % fibrin density
fibrin_conc_g_um3 = fibrin_concentration*1e-15; % g/um^3
fibrin_volume_um3 = (fibrin_conc_g_um3 / fibrin_density_g_um3) * eff_mult * max(shell_vol_est - inc_vol,1);
fprintf("section-2 done\n")
%% ---- SECTION 3: Node count ----
total_bond_len = fibrin_volume_um3 / (pi * fibrin_radius_um^2);
target_mean_length = interp1([C1,C2],[L1,L2],fibrin_concentration,'linear','extrap');
numPoints = max(ceil(total_bond_len / max(target_mean_length,1e-3)), 40);
fprintf("section-3 done\n")
%% ---- SECTION 4: Clustered points (GPU-accelerated batched sampling) ----
% Goal: draw ~numPoints points inside shell, outside inclusions, with clustered Gaussians
clusters = max(1, size(sphere_centers,1));
base_band = (min(extent)/8)^2; % covariance scale (um^2)
cand_factor = 2; % initial oversampling
chunk_inpoly = 1e5; % mex_inpolyhedron chunk size
batch_size = 1e7; % max GPU batch
haveInclusions = ~isempty(sphere_centers);
SC = single(sphere_centers);
SR = single(sphere_radii);
% Precompute per-cluster sigma (isotropic) proportional to inclusion radius
if haveInclusions
    sig_per = single(max((SR*0.35).^2, base_band)); % variance ~ (0.35 R)^2
else
    sig_per = single(base_band)*ones(1,'single');
end
need = numPoints;
taken = 0;
accum = zeros(0,3,'single');
attempt = 0; max_attempts = 25;
while taken < need && attempt < max_attempts
    attempt = attempt + 1;
    M_total = ceil(cand_factor * (need - taken));
    n_batches = ceil(M_total / batch_size);
    accepted_batch = 0; % for adaptive update
    for b = 1:n_batches
        M = min(batch_size, M_total - (b-1)*batch_size);
        %% --- Step 1: Generate candidates on GPU ---
        if haveInclusions
            ridx = randi(clusters, M, 1, 'single');
            mu = SC(ridx, :);
            sigv = sig_per(max(1, min(clusters, round(ridx))));
            stdv = sqrt(sigv);
            Z = randn(M, 3, 'single', 'gpuArray');
            Cand = gather(mu + Z .* stdv); % Mx3 in µm
        else
            center = single(bbmin + 0.5 * extent);
            std0 = sqrt(single(base_band));
            Z = randn(M, 3, 'single', 'gpuArray');
            Cand = gather(center + Z * std0);
        end
        %% --- Step 2: Bounding box pre-filter (fast CPU) ---
        in_bb = Cand(:,1)>=bbmin(1) & Cand(:,1)<=bbmax(1) & ...
            Cand(:,2)>=bbmin(2) & Cand(:,2)<=bbmax(2) & ...
            Cand(:,3)>=bbmin(3) & Cand(:,3)<=bbmax(3); Cand = Cand(in_bb,:);
        %% --- Step 3: Exclude points inside any inclusion (GPU KNN) ---
        if haveInclusions && ~isempty(Cand)
            [nn_idx, nn_d] = knn_grid_cuda(SC, single(Cand), int32(1));
            nn_idx = gather(nn_idx);
            nn_d = gather(nn_d);
            keep = nn_d > SR(double(nn_idx)); % outside nearest inclusion
            Cand = Cand(keep,:);
        end
        %% --- Step 4: Keep only points inside shell mesh (CPU chunks) ---
        if isempty(Cand), continue; end
        numCand = size(Cand,1);
        nChunks = ceil(numCand / chunk_inpoly);
        ok_chunks = cell(nChunks,1);
        parfor c = 1:nChunks
            j1 = (c-1)*chunk_inpoly + 1;
            j2 = min(c*chunk_inpoly, numCand);
            ok_chunks{c} = mex_inpolyhedron(double(SF), double(SV), double(Cand(j1:j2,:)));
        end
        ok_shell = vertcat(ok_chunks{:});
        Cand = Cand(ok_shell,:);
        %% --- Step 5: Deduplicate on a 0.05 µm grid ---
        if isempty(Cand), continue; end
        Q = round(Cand / 0.05);
        [~, ia] = unique(Q, 'rows', 'stable');
        Cand = Cand(ia,:);
        %% --- Step 6: Accumulate accepted points ---
        take = min(size(Cand,1), need - taken);
        accum = [accum; Cand(1:take,:)]; %#ok<AGROW>
        taken = size(accum,1);
        accepted_batch = accepted_batch + take;
        fprintf("progress : %f \n",100*length(accum)/numPoints);
        if size(accum,1) == numPoints
            break;
        end
    end
    if size(accum,1) == numPoints
        break;
    end
    %% --- Adaptive oversampling adjustment ---
    kept_rate = max(accepted_batch / max(M_total,1), 1e-4);
    if kept_rate < 0.05
        cand_factor = min(cand_factor * 1.5, 12); % too strict -> widen
    elseif kept_rate > 0.3
        cand_factor = max(cand_factor * 0.8, 3); % too lenient -> tighten
    end
    fprintf('Pass %d: kept %d/%d (%.2f%%) -> cand_factor=%.2f\n', ...
        attempt, accepted_batch, M_total, kept_rate*100, cand_factor);
end
if size(accum,1) < numPoints
    warning('Accepted %d / %d points; increase max_attempts or cand_factor.', ...
        size(accum,1), numPoints);
end
Points = single(accum(1:min(end,numPoints), :));
Window_center = mean(Points,1,'omitnan');
fprintf("section-4 done\n");
%% ---- SECTION 5: Bonds via GPU KNN ----
target_valency = interp1([C1,C2],[Z1,Z2],fibrin_concentration,'linear','extrap');
k = 8; % neighbors to consider
[idx, dists] = knn_grid_call(Points, k);
[bonds, ~, ~] = generate_bonds_from_knn( ...
    Points, idx, dists, target_valency, preferred_direction, direction_weight, true); % last arg = useGPU
fprintf("section-5 done\n");
try
    g = gpuDevice;
    wait(g);
    reset(g);
catch
    warning('GPU reset skipped (device busy)');
end
%% ---- SECTION 6: Gentle push off inclusions (vector) ----
if ~isempty(sphere_centers)
    % nearest inclusion center per point (K=1)
    [nn_idx, nn_dist] = knn_grid_cuda(single(sphere_centers), single(Points), int32(1));
    nn_idx = double(nn_idx); % M×1 (1-based)
    nn_dist = double(nn_dist); % M×1
    rinfl_pt = 1.2 * sphere_radii(nn_idx); % per-point influence radius
    A = nn_dist < rinfl_pt; % points to push
    if any(A)
        vec = Points(A,:) - sphere_centers(nn_idx(A),:);
        nv = sqrt(sum(vec.^2,2)) + 1e-9;
        u = vec ./ nv;
        disp = rinfl_pt(A) - nn_dist(A);
        Points(A,:) = Points(A,:) + u .* disp;
    end
end
fprintf("section-6 done\n")
%% ---- SECTION 7: Relaxation (SDF + adaptive bounding box + moved-only check) ----
max_iterations = 1000;
energy_threshold = 0.1;
k_fibrin = 1;
step_size = 0.00001;
max_step_size = 1; min_step_size = 1e-8;
prevJS = Inf;
nu = interp1([C1,C2],[v1,v2],fibrin_concentration,'linear','extrap');
L = target_mean_length;
s2 = nu*L^2;
z2 = log((s2/L^2)+1);
lam = log(L) - 0.5*z2;
Pg = gpuArray(single(Points));
Lg = compute_bond_lengths_gpu(bonds, Pg, true);
xv = gpuArray(linspace(1e-10, max(gather(Lg)), max(10,numel(bonds))));
lnpdf = (1 ./ (xv * sqrt(2*pi*z2))) .* exp(-((lam - log(xv)).^2)/(2*z2));
angle_w = 0.1;
grow_js_grid_every = 25;
last_xv_max = -Inf;
%% --- Safe localized SDF grid creation ---
grid_res = 2.0; % base resolution in µm
pad = 20.0; % padding around geometry (µm)
max_vox = 2e7; % safety cap (~80 MB single precision)
bbmin_local = min(SV) - pad;
bbmax_local = max(SV) + pad;
approx_vox = prod((bbmax_local - bbmin_local) / grid_res);
if approx_vox > max_vox
    scale = (approx_vox / max_vox)^(1/3);
    grid_res = grid_res * scale;
    fprintf('⚠️ Grid too large (%.1f M voxels). Increasing grid_res to %.2f µm.\n', ...
        approx_vox/1e6, grid_res);
end
xv_sdf = bbmin_local(1):grid_res:bbmax_local(1);
yv_sdf = bbmin_local(2):grid_res:bbmax_local(2);
zv_sdf = bbmin_local(3):grid_res:bbmax_local(3);
fprintf('Creating local SDF grid %dx%dx%d (%.1f M voxels, %.2f µm spacing)...\n', ...
    numel(xv_sdf), numel(yv_sdf), numel(zv_sdf), ...
    numel(xv_sdf)*numel(yv_sdf)*numel(zv_sdf)/1e6, grid_res);
[gx, gy, gz] = meshgrid(xv_sdf, yv_sdf, zv_sdf);
queryPts = [gx(:), gy(:), gz(:)];
% Compute distance to the nearest vertex (approximation of point-to-surface)
fprintf('Computing nearest vertex distances via GPU...\n');
% Compute nearest vertex distance using GPU KNN (k = 1)
[~, dist_nn] = knn_grid_cuda(single(SV), single(queryPts), int32(1));
% dist_nn is already the Euclidean distance to the nearest vertex
D_flat = gather(dist_nn(:));
fprintf('GPU nearest neighbor distances computed (%.1f MB)\n', numel(D_flat)*4/1e6);
In_flat = mex_inpolyhedron(SF,SV,[gx(:),gy(:),gz(:)]);
sdf = reshape(D_flat .* (1 - 2*In_flat), size(gx));
fprintf('SDF grid ready (%.1f MB in memory)\n', numel(sdf)*4/1e6);
%% --- Initialize state ---
Points_prev = Points;
inside_ok = interp3(gx,gy,gz,sdf,Points(:,1),Points(:,2),Points(:,3),'linear',Inf) < 0;
osc_win = 30; osc_band = 0.015; osc_cross_min = 6;
osc_min_iters = 80; osc_trig_level = 0.15;
JS_buf = nan(1,osc_win); buf_i = 0;
for it = 1:max_iterations
    Lg = compute_bond_lengths_gpu(bonds, Pg, true);
    L = gather(Lg);
    Lmax = max(L(~isinf(L) & ~isnan(L)));
    if ~isfinite(Lmax) || Lmax <= 0
        Lmax = max(1e-6, target_mean_length);
    end
    if it==1 || it % grow_js_grid_every == 0 || Lmax > 0.95*last_xv_max xv = gpuArray(linspace(1e-10, Lmax, max(256, min(4096, numel(bonds)))));
        lnpdf = (1 ./ (xv * sqrt(2*pi*z2))) .* exp(-((lam - log(xv)).^2)/(2*z2));
        last_xv_max = Lmax;
    end
    finiteMask = isfinite(L) & (L > 0);
    maxReasonable = 1e3 * max(1e-9, target_mean_length);
    finiteMask = finiteMask & (L < maxReasonable);
    L_fit = L(finiteMask);
    if numel(L_fit) < 10, L_fit = max(1e-9, L(finiteMask | isfinite(L))); end
    pd1 = fitdist(L_fit, 'Lognormal');
    dx = max(1e-12, mean(diff(gather(xv))));
    pdf1 = pdf(pd1, gather(xv)); pdf1 = pdf1 / sum(pdf1*dx);
    lnp = gather(lnpdf); lnp = lnp / sum(lnp*dx);
    avg = 0.5*(pdf1 + lnp);
    epsz = 1e-10;
    JS = 0.5*( sum((pdf1+epsz).*log((pdf1+epsz)./(avg+epsz)))*dx + ...
        sum((lnp +epsz).*log((lnp +epsz)./(avg+epsz)))*dx );
    if JS < energy_threshold, break; end
    % --- Compute forces and integrate ---
    Fg = mex_forces_log_normal(int32(bonds), Lg, lam, z2, k_fibrin, Pg);
    ang_cpu = mex_angle_forces(double(bonds), double(gather(Pg)),1.0);
    Fg = Fg + angle_w * gpuArray(single(ang_cpu));
    Pg = Pg + step_size * Fg;
    Points = gather(Pg);
    % --- Fast SDF constraint ---
    moved = vecnorm(Points - Points_prev,2,2) > 0.1;
    if any(moved)
        sdf_vals = interp3(gx,gy,gz,sdf,Points(moved,1),Points(moved,2),Points(moved,3),'linear',Inf);
        inside_ok(moved) = sdf_vals < 0;
    end
    Points_prev = Points;
    out_idx = find(~inside_ok);
    if ~isempty(out_idx)
        dir_to_center = (mean(Points,1) - Points(out_idx,:));
        dir_to_center = dir_to_center ./ (vecnorm(dir_to_center,2,2)+1e-9);
        Points(out_idx,:) = Points(out_idx,:) + 0.1 * dir_to_center;
    end
    % --- Repulsion from inclusions ---
    if ~isempty(sphere_centers)
        [nn_idx, nn_dist] = knn_grid_cuda(single(sphere_centers), single(Points), int32(1));
        nn_idx = double(nn_idx); nn_dist = double(nn_dist);
        rinfl_pt = 1.2 * sphere_radii(nn_idx);
        A = nn_dist < rinfl_pt;
        if any(A)
            vec = Points(A,:) - sphere_centers(nn_idx(A),:);
            nv = sqrt(sum(vec.^2,2)) + 1e-9;
            u = vec ./ nv; disp = rinfl_pt(A) - nn_dist(A);
            Points(A,:) = Points(A,:) + u .* disp;
        end
    end
    Pg = gpuArray(single(Points));
    % --- Adaptive step control ---
    if it>1 && JS >= prevJS
        step_size = max(step_size * 0.95, min_step_size);
    else
        step_size = min(step_size * 1.05, max_step_size);
    end
    prevJS = JS;
end
Points = gather(Pg);
fprintf("section-7 done\n");
try
    g = gpuDevice;
    wait(g);
    reset(g);
catch
    warning('GPU reset skipped (device busy)');
end
%% ---- Cleanup: remove very long bonds & keep LCC (CPU fast) ----
%% --- Prune nodes incident to very long edges ---
maxL = 5 * target_mean_length;
% Compute bond lengths (GPU)
Lg = compute_bond_lengths_gpu(bonds, gpuArray(single(Points)), false);
bl = double(Lg);
longE = bl > maxL;
badNodes = unique(bonds(longE, :)); keepNodes = true(size(Points, 1), 1); keepNodes(badNodes) = false;
% Keep only edges with both endpoints valid
keepE = keepNodes(bonds(:,1)) & keepNodes(bonds(:,2));
bonds = bonds(keepE, :);
Points = Points(keepNodes, :);
% --- Remap node indices to compact 1..N ---
map = zeros(size(keepNodes), 'uint32');
map(keepNodes) = 1:nnz(keepNodes);
bonds = [map(bonds(:,1)), map(bonds(:,2))];
fprintf("test point 1\n");
%% --- Largest connected component ---
G = graph(bonds(:,1), bonds(:,2)); % Compact indices now
comp = conncomp(G); % Component label per node
[~, big] = max(histcounts(comp, 1:(max(comp)+1)));
% Keep only nodes/edges from largest component
keepMask = (comp == big);
G2 = subgraph(G, find(keepMask));
% Extract compact data
Points = Points(keepMask, :);
bonds = double(G2.Edges.EndNodes);
% valency -> platelet centers
conn = accumarray([bonds(:), ones(numel(bonds),1)], 1, [size(Points,1) 1]);
four_idx = find(conn==4);
n_pl = min(numel(four_idx), round(platelet_ratio*numel(four_idx)));
if n_pl>0, four_idx = four_idx(randperm(numel(four_idx), n_pl)); else, four_idx = []; end
platelet_centers = Points(four_idx,:);
fprintf("test point 2\n");
%% -------- SECTION 8: RBC packing and inclusion filling ----------
%% ---- RBC packing inside inclusions (CUDA MEX; procedural + relaxation) ----
stepR = resolution; % voxel step ~0.5 µm
[Pt_local, ~] = build_canonical_rbc_cloud(10, stepR);
Pt_local = single(Pt_local);
rbc_r_eff = max(vecnorm(double(Pt_local),2,2)); % biconcave "radius"
min_sep = 1.7 * rbc_r_eff; % tuned for compact packing
target_phi = 0.50; % target effective filling (~0.5)
use_cuda = (exist('rbc_pack_cuda','file')==3);
rbc_pts_all = single([]);
rbc_clusters = {};
try
    g = gpuDevice;
    wait(g);
catch
    warning('GPU reset skipped (device busy)');
end
% --- Optional: check system memory (Linux) ---
try
    [~, meminfo] = system('free -h | grep Mem');
    fprintf('Memory status: %s\n', strtrim(meminfo));
catch
end
for s = 1:numel(sphere_radii)
    C = single(sphere_centers(s,:));
    R = single(sphere_radii(s));
    if R < 1.0, continue; end
    %% (1) Poisson-disk seeding
    r_equiv = min_sep/2;
    vol_cell = (4/3)*pi*r_equiv^3;
    nTarget = floor( target_phi * ((4/3)*pi*double(R)^3) / vol_cell );
    nTarget = max(64, nTarget);
    [centers0, ~] = poisson3D_in_ball(C, R - r_equiv, r_equiv, 6*nTarget);
    centers0 = centers0(1:min(size(centers0,1), round(1.5*nTarget)), :);
    angles0 = single(rand(size(centers0,1),3) * pi);
    %% (2) Short local relaxation on centers (CUDA neighbor search)
    iters = 20;
    alpha_push = 0.55;
    eps_nbr = r_equiv;
    keep_inside = true;
    centers_relaxed = pbd_relax_centers(centers0, C, R, ...
        min_sep, eps_nbr, iters, alpha_push, keep_inside);
    % keep only inside
    inside = sum((centers_relaxed - C).^2,2) <= (R - 0.5)^2;
    centers_relaxed = centers_relaxed(inside,:);
    angles0 = angles0(inside,:);
    %% (3) Instantiate RBC clouds with CUDA
    centers_local = single(centers_relaxed - C);
    [pts_out, ~, ~] = rbc_pack_cuda( ...
        single(Pt_local), centers_local, single(angles0), ...
        C, R, single(bbmin), single(bbmax), int32(64));
    % shift from local to global
    pts_out = pts_out + C;
    if isempty(pts_out), continue; end
    %% (4) Append cluster (store per-sphere)
    rbc_clusters{end+1} = pts_out; %#ok<SAGROW>
    fprintf("progress %f\n",100*s/numel(sphere_radii));
    if mod(s,20)==0
        % --- Optional: check system memory (Linux) ---
        try
            [~, meminfo] = system('free -h | grep Mem');
            fprintf('Memory status: %s\n', strtrim(meminfo));
        catch
        end
        fprintf("Flushing GPU and workspace...\n");
        drawnow; pause(0.1);

        % --- Flush GPU ---
        try
            g = gpuDevice;
            wait(g);

            gpuDevice(g.Index); % reinitialize GPU context
        catch ME
            warning(ME.identifier,'%s', ME.message);
        end

        % --- Suggest garbage collection to the JVM (optional but harmless) ---
        try
            java.lang.System.gc();
        catch
            % ignore if JVM not available
        end

        % --- Clear only temporary per-sphere variables ---
        clear C R r_equiv vol_cell nTarget centers0 angles0 ...
            iters alpha_push eps_nbr keep_inside centers_relaxed ...
            inside centers_local pts_out

        drawnow;
    end

end
% --- Reset GPU to free VRAM ---
try
    g = gpuDevice;
    wait(g);
    reset(g)
catch
    warning('GPU reset skipped (device busy)');
end

fprintf("section-8 done\n")
%% -------- SECTION 9: Optimized On-Demand PA Simulation per Voxel ----------
fprintf('\n=== SECTION 9: On-Demand PA simulation per voxel (optimized) ===\n');

%% --- Compute coarse voxel grid
bbmin = min(SV,[],1);
bbmax = max(SV,[],1);
half  = voxel_size/2;

ex = bbmin(1):voxel_size:(bbmax(1)-voxel_size);
ey = bbmin(2):voxel_size:(bbmax(2)-voxel_size);
ez = bbmin(3):voxel_size:(bbmax(3)-voxel_size);
cx = ex + half; cy = ey + half; cz = ez + half;
nxg = numel(cx); nyg = numel(cy); nzg = numel(cz);

if nxg==0 || nyg==0 || nzg==0
    warning('No macro voxels fit inside shell bbox.');
    return;
end

fprintf('Grid: %d x %d x %d voxels (%.2f µm voxel size)\n', nxg, nyg, nzg, voxel_size);

nfx = round(voxel_size/resolution);
nfy = nfx; nfz = nfx;

r_pad_rbc = max(1.5*resolution, 0.5);
r_pad_fib = max(resolution, fibrin_radius_um);

out_dir = fullfile(pwd,'Voxel_PA_responses');
if ~isfolder(out_dir), mkdir(out_dir); end

cuda_dir = '/home/hghodsi/Software_packages/k-wave-toolbox-version-1.4/k-Wave/kspaceFirstOrder-CUDA';

%% -------- SECTION 9 (Serial): On-demand PA Simulation (Unique Voxels) --------
fprintf('\n=== SECTION 9: SERIAL On-demand PA simulation (unique voxels) ===\n');

bbmin = min(SV,[],1);
bbmax = max(SV,[],1);
half  = voxel_size/2;

ex = bbmin(1):voxel_size:(bbmax(1)-voxel_size);
ey = bbmin(2):voxel_size:(bbmax(2)-voxel_size);
ez = bbmin(3):voxel_size:(bbmax(3)-voxel_size);
cx = ex + half; cy = ey + half; cz = ez + half;
nxg = numel(cx); nyg = numel(cy); nzg = numel(cz);

nfx = round(voxel_size/resolution);
nfy = nfx; nfz = nfx;

out_dir = fullfile(pwd,'Voxel_PA_responses');
if ~isfolder(out_dir), mkdir(out_dir); end
cuda_dir = '/home/hghodsi/Software_packages/k-wave-toolbox-version-1.4/k-Wave/kspaceFirstOrder-CUDA';

num_clusters = numel(rbc_clusters);

%% --- Step 1: Build voxel → cluster lookup table
fprintf('Building voxel-cluster map...\n');
voxel_clusters = containers.Map('KeyType','uint64','ValueType','any');

for ci = 1:num_clusters
    pts = rbc_clusters{ci};
    if isempty(pts), continue; end

    ix = floor((pts(:,1)-bbmin(1))/voxel_size) + 1;
    iy = floor((pts(:,2)-bbmin(2))/voxel_size) + 1;
    iz = floor((pts(:,3)-bbmin(3))/voxel_size) + 1;
    valid = ix>=1 & ix<=nxg & iy>=1 & iy<=nyg & iz>=1 & iz<=nzg;
    vox_ids = unique(sub2ind([nxg,nyg,nzg], ix(valid), iy(valid), iz(valid)));

    for v = vox_ids'
        if isKey(voxel_clusters, v)
            voxel_clusters(v) = [voxel_clusters(v), ci];
        else
            voxel_clusters(v) = ci;
        end
    end

    if mod(ci,50)==0
        fprintf('  processed %d / %d clusters\n', ci, num_clusters);
    end
end

voxel_keys = keys(voxel_clusters);
fprintf('Total non-empty voxels: %d\n', numel(voxel_keys));

%% --- Step 2: Serial simulation per unique voxel
for vi = 1:numel(voxel_keys)
    fprintf('Simulating Voxel:%d/%d \n',vi,numel(voxel_keys));
    key = voxel_keys{vi};
    clist = voxel_clusters(key);
    [ix,iy,iz] = ind2sub([nxg,nyg,nzg], str2double(key));

    C = [cx(ix), cy(iy), cz(iz)];
    cmin = C - half; cmax = C + half;

    Vox = zeros(nfx,nfy,nfz,'uint8');

    % --- add RBCs from all clusters that touch this voxel
    clist = voxel_clusters(id);
    for ci = clist
        pts = rbc_clusters{ci};
        if isempty(pts), continue; end
        mask = pts(:,1)>=cmin(1) & pts(:,1)<=cmax(1) & ...
               pts(:,2)>=cmin(2) & pts(:,2)<=cmax(2) & ...
               pts(:,3)>=cmin(3) & pts(:,3)<=cmax(3);
        pts_vox = pts(mask,:);
        if isempty(pts_vox), continue; end

        ixr = max(1, min(nfx, round((pts_vox(:,1)-cmin(1))/resolution) + 1));
        iyr = max(1, min(nfy, round((pts_vox(:,2)-cmin(2))/resolution) + 1));
        izr = max(1, min(nfz, round((pts_vox(:,3)-cmin(3))/resolution) + 1));
        Vox(sub2ind([nfx,nfy,nfz], ixr, iyr, izr)) = 1;
    end

    % --- Fibrin and Platelets
    if ~isempty(bonds)
        Vox = DrawFibrinTubes(Vox, Points, bonds, cmin, resolution, fibrin_radius_um);
    end
    if ~isempty(platelet_centers)
        Vox = DrawSpheres(Vox, platelet_centers, cmin, resolution, platelet_radius_um, 3);
    end

    % --- compute composition
    nRBC = nnz(Vox==1); nFib = nnz(Vox==2); nPlt = nnz(Vox==3);
    nSolid = nRBC + nFib + nPlt;
    composition = nRBC / max(1,nSolid);
    porosity = 1 - double(nSolid)/numel(Vox);

    % --- simulate PA
    try
        voxel_size_um = size(Vox,1)*resolution;
        PA_response = calculate_PA_response_micro_gpu(Vox, voxel_size_um, cuda_dir);
        if isempty(PA_response), continue; end

        save_name = fullfile(out_dir, sprintf('clot_%s_response_%04d_%04d_%04d.mat', ...
            mesh_name, ix, iy, iz));
        save(save_name, 'cmin','composition','porosity','PA_response','-v7.3');
    catch ME
        warning('Voxel %d simulation error: %s', vi, ME.message);
        continue;
    end

    % --- cleanup between voxels
    if mod(vi,5)==0
        java.lang.System.gc();
        try
            g = gpuDevice;
            wait(g); reset(g);
        catch
        end
        system('free -h');
    end
end

fprintf('✅ SERIAL PA simulation complete for %d unique voxels.\n', numel(voxel_keys));


end
%% ------Helper functions-----
function x = getP(S, f, d) % Safe param getter: x = S.f if exists & nonempty, else default d
if isstruct(S) && isfield(S,f) && ~isempty(S.(f))
    x = S.(f);
else
    x = d;
end
end

function Vox = DrawSpheres(Vox, centers, cmin, h, rad, label)
% Vox: nfx x nfy x nfz (uint8)
% centers: Mx3
% cmin: world coords of Vox(1,1,1)
% h: voxel size (um)
% rad: sphere radius (um)
% label: uint8 value to write
[nx, ny, nz] = size(Vox);
for k = 1:size(centers,1)
    c = centers(k,:);
    % quick AABB in voxel coords
    ix = clamp_idx( (c(1)-rad - cmin(1))/h + 1, (c(1)+rad - cmin(1))/h + 1, nx );
    iy = clamp_idx( (c(2)-rad - cmin(2))/h + 1, (c(2)+rad - cmin(2))/h + 1, ny );
    iz = clamp_idx( (c(3)-rad - cmin(3))/h + 1, (c(3)+rad - cmin(3))/h + 1, nz );
    if ix(1)>ix(2) || iy(1)>iy(2) || iz(1)>iz(2)
        continue; % fully outside
    end
    % local grid (small)
    gx = (double(ix(1):ix(2)) - 1) * h + double(cmin(1));
    gy = (double(iy(1):iy(2)) - 1) * h + double(cmin(2));
    gz = (double(iz(1):iz(2)) - 1) * h + double(cmin(3));
    [X, Y, Z] = ndgrid(gx, gy, gz);
    % inside sphere mask
    M = (X-c(1)).^2 + (Y-c(2)).^2 + (Z-c(3)).^2 <= double(rad)^2;
    % write label (don't overwrite nonzero if you want precedence logic)
    sub = Vox(ix(1):ix(2), iy(1):iy(2), iz(1):iz(2));
    sub(M) = uint8(label);
    Vox(ix(1):ix(2), iy(1):iy(2), iz(1):iz(2)) = sub;
end
end

function Vox = DrawFibrinTubes(Vox, Points, bonds, cmin, h, rad)
%DRAWFIBRINTUBES_FAST Efficient rasterization of curved fibrin bonds.
% Vox : uint8[nx,ny,nz]
% Points : Nx3
% bonds : Mx2
% cmin : world coord of Vox(1,1,1)
% h : voxel size
% rad : tube radius (um)

[nx, ny, nz] = size(Vox);

% Parameters
step = max(0.5*h, 0.5*rad);
curve_strength = 0.25;
max_bulge_um = 3*rad;

% Precompute small sphere mask (radius in voxels)
rvox = max(1, round(rad/h));
[xs, ys, zs] = ndgrid(-rvox:rvox);
sphere_mask = (xs.^2 + ys.^2 + zs.^2) <= (rad/h)^2;
mask_size = size(sphere_mask);

for e = 1:size(bonds,1)
    i = bonds(e,1); j = bonds(e,2);
    p1 = double(Points(i,:));
    p2 = double(Points(j,:));
    chord = p2 - p1;
    L = norm(chord);
    if L < 1e-9, continue; end

    % --- deterministic curvature ---
    dir = edge_randdir(p1,p2);
    t_hat = chord / L;
    dir = dir - dot(dir,t_hat)*t_hat;
    nd = norm(dir);
    if nd < 1e-9, dir = [1 0 0]; else, dir = dir/nd; end
    bulge = min(curve_strength*L, max_bulge_um);
    ctrl = (p1+p2)/2 + bulge*dir;

    % --- sample Bézier curve ---
    nS = max(3, ceil(L/step));
    t = linspace(0,1,nS).';
    omt = 1 - t;
    P = (omt.^2).*p1 + 2*(omt.*t).*ctrl + (t.^2).*p2; % nS×3

    % Convert to voxel coordinates (1-based)
    Pv = (P - cmin) / h + 1;

    % Clip to voxel box bounds
    valid = all(Pv >= 1 & Pv <= [nx ny nz], 2);
    Pv = Pv(valid,:);
    if isempty(Pv), continue; end

    % Compute all voxel indices to fill for this tube
    for k = 1:size(Pv,1)
        cx = round(Pv(k,1)); cy = round(Pv(k,2)); cz = round(Pv(k,3));
        % Determine bounding box in voxel coordinates
        ix = max(1, cx - rvox):min(nx, cx + rvox);
        iy = max(1, cy - rvox):min(ny, cy + rvox);
        iz = max(1, cz - rvox):min(nz, cz + rvox);

        % Sphere mask fits entirely within local region
        sub = Vox(ix,iy,iz);
        sub_mask = sphere_mask(...
            1+(ix(1)-cx+rvox):mask_size(1)-(cx+ rvox - ix(end)), ...
            1+(iy(1)-cy+rvox):mask_size(2)-(cy+ rvox - iy(end)), ...
            1+(iz(1)-cz+rvox):mask_size(3)-(cz+ rvox - iz(end)) );
        sub(sub_mask) = uint8(2);
        Vox(ix,iy,iz) = sub;
    end
end
end

function v = edge_randdir(p1, p2)
% Deterministic pseudo-random unit vector from edge endpoints.
% Uses cheap trig hashing; no RNG state, safe in parfor.
a = p1(:).'; b = p2(:).';
s = sin( 1e-3* (a*[12.9898 78.233 37.719].' + b*[93.989 67.345 12.345].') );
t = cos( 2e-3* (a*[45.332 11.135 97.713].' + b*[22.447 8.771 3.141].') );
u = sin( 3e-3* (sum(a.*[0.707 0.577 0.408]) - sum(b.*[0.447 0.894 0.000])) );
v = [s, t, u];
n = norm(v);
if n < 1e-12, v = [1 0 0]; else, v = v / n; end
end

function r = clamp_idx(a, b, lo)
% Return a 1x2 integer index range clamped to [1, lo].
% Accepts either (low, high, lo) as scalars OR ([low high], lo).
if nargin == 2
    % called as clamp_idx([low high], lo)
    lo = b;
    low = a(1);
    high = a(2);
else
    % called as clamp_idx(low, high, lo)
    low = a;
    high = b;
end
% round to nearest voxel and clamp
i1 = max(1, min(lo, floor(low)));
i2 = max(1, min(lo, ceil(high)));
if i1 > i2
    % swap if the order came reversed
    t = i1; i1 = i2; i2 = t;
end
r = [int32(i1) int32(i2)];
end

function [V,F] = readOBJ_local(fn)
fid = fopen(fn,'r'); assert(fid>0,'Cannot open %s',fn);
V = []; F = [];
while true
    t = fgetl(fid); if ~ischar(t), break; end
    t = strtrim(t);
    if startsWith(t,'v ')
        V(end+1,:) = sscanf(t(3:end),'%f %f %f').';
    elseif startsWith(t,'f ')
        raw = regexp(t(3:end),'\s+','split'); idx = zeros(1,3);
        for k = 1:3
            p = split(raw{k},'/');
            idx(k) = str2double(p{1});
        end
        F(end+1,:) = idx;
    end
end
fclose(fid);
end

function Q = ensureNx3double(Q)
Q = double(Q);
if ~ismatrix(Q), Q = reshape(Q, [], 3);
elseif size(Q,2) ~= 3
    if size(Q,1) == 3, Q = Q.'; else, Q = reshape(Q, [], 3); end
end
end

function vol = polyVolume(V,F)
v1 = V(F(:,1),:);
v2 = V(F(:,2),:);
v3 = V(F(:,3),:);
vol = abs(sum(dot(v1, cross(v2,v3,2), 2)))/6;
end

function L = compute_bond_lengths_gpu(E, Pg, verbose)
% Safe, chunked GPU computation of bond lengths.
% E : [M×2] uint32 or double edge indices
% Pg: [N×3] gpuArray(single)
% verbose: true/false (optional)
if nargin < 3, verbose = false; end
if ~isa(Pg, 'gpuArray')
    error('Pg must be a gpuArray(single) of point coordinates');
end
M = size(E,1);
chunkSize = 1e7; % adjust based on your GPU memory
numChunks = ceil(M / chunkSize);
if verbose
    fprintf('Computing bond lengths on GPU in %d chunks (~%.1f M bonds each)\n', ...
        numChunks, chunkSize/1e6);
end
L = zeros(M,1,'single');
for c = 1:numChunks
    sIdx = (c-1)*chunkSize + 1;
    eIdx = min(c*chunkSize, M);
    mLoc = eIdx - sIdx + 1;
    E_sub = E(sIdx:eIdx,:);
    try
        % Pull sub-points from GPU in chunks
        p1 = Pg(E_sub(:,1),:);
        p2 = Pg(E_sub(:,2),:);
        diff = p1 - p2;
        L(sIdx:eIdx) = gather( sqrt(sum(diff.^2, 2, 'omitnan')));
    catch ME
        if strcmpi(ME.identifier, 'parallel:gpu:array:OOM') || ...
                contains(ME.message, 'Maximum variable size allowed')
            warning('GPU OOM at chunk %d, switching to CPU fallback for this chunk.', c);
            Pg_cpu = gather(Pg);
            p1 = Pg_cpu(E_sub(:,1),:);
            p2 = Pg_cpu(E_sub(:,2),:);
            L(sIdx:eIdx) = sqrt(sum((p1 - p2).^2, 2, 'omitnan'));
            clear Pg_cpu
        else
            rethrow(ME);
        end
    end
    if verbose && mod(c,5)==0
        fprintf(' ✔ processed chunk %d/%d (%d bonds)\n', c, numChunks, mLoc);
    end
end
if verbose
    fprintf('GPU bond length computation complete (%d bonds total)\n', M);
end
end

function [idx_all, dist_all] = knn_grid_call(X, k)
%KNN_GRID_CALL  Fully fault-tolerant GPU batched kNN with auto chunk scaling.
%
% Handles >2e9 neighbor queries gracefully without ever exhausting GPU RAM.
% If CUDA OOM or illegal memory access occurs, the chunk size is halved and retried.
%
% Inputs
%   X : [N×3 single]  coordinate array
%   k : integer number of neighbors
%
% Outputs
%   idx_all  : [N×k int32]   neighbor indices
%   dist_all : [N×k single]  neighbor distances (Euclidean)
%
% Requires knn_grid_cuda MEX on GPU.

X = single(X);
N = size(X,1);
assert(size(X,2)==3, 'X must be Nx3');
assert(exist('knn_grid_cuda','file')==3, 'Missing MEX: knn_grid_cuda');
assert(gpuDeviceCount>0, 'No GPU detected.');
assert(isscalar(k) && k>0, 'k must be a positive scalar.');

fprintf('knn_grid_call: N=%d, k=%d [adaptive GPU mode]\n', N, k);

% --- memory-aware initial chunking ---
g = gpuDevice;
freeB = double(g.AvailableMemory);
bytes_per_pair = 8;            % int32 + single
maxBytes = 0.5 * freeB;        % use 50% of free memory for safety
chunkN = floor(maxBytes / (bytes_per_pair * k));
chunkN = max(2e5, min(chunkN, 2e7));   % clamp practical limits
nChunks = ceil(N / chunkN);

fprintf('  GPU free %.1f GB → start chunk %.1f M pts (%d chunks)\n', ...
    freeB/1e9, chunkN/1e6, nChunks);

% --- preallocate outputs on CPU ---
idx_all  = zeros(N, k, 'int32');
dist_all = zeros(N, k, 'single');
Xref     = single(X);

maxRetries = 5;        % maximum retry attempts per chunk
minChunk   = 50;       % minimum number of points per batch

c = 1;
while c <= nChunks
    i1 = (c-1)*chunkN + 1;
    i2 = min(c*chunkN, N);
    Q  = single(X(i1:i2,:));
    fprintf('  Chunk %2d/%d: %d pts...', c, nChunks, i2-i1+1);

    retryCount = 0;
    success = false;

    while ~success && retryCount < maxRetries
        try
            [idx_chunk, dist_chunk] = knn_grid_cuda(Xref, Q, int32(k));
            idx_all(i1:i2,:)  = idx_chunk;
            dist_all(i1:i2,:) = dist_chunk;
            fprintf(' ✓\n');
            clear idx_chunk dist_chunk Q
            success = true;

        catch ME
            msg = string(ME.message);
            if contains(msg, {'out of memory','CUDA_ERROR_ILLEGAL_ADDRESS','OOM'}, 'IgnoreCase',true)
                retryCount = retryCount + 1;
                warning('⚠ GPU OOM in chunk %d — attempt %d/%d', c, retryCount, maxRetries);

                % Reduce chunk size progressively
                oldChunk = chunkN;
                chunkN = max(floor(chunkN / 2), minChunk);
                nChunks = ceil(N / chunkN);

                try
                    reset(gpuDevice);
                catch
                    warning('GPU reset failed; continuing without reset.');
                end

                fprintf('  → Retrying chunk %d with smaller batch (%.0f → %.0f pts)...\n', ...
                    c, oldChunk, chunkN);
                Q = single(X(i1:i2,:)); % reload batch after reset
            else
                rethrow(ME);
            end
        end
    end

    if ~success
        warning('❌ Failed to process chunk %d after %d retries. Skipping.', c, maxRetries);
        % leave idx_all(i1:i2,:) as zeros; skip to next chunk safely
    end

    % Move to next chunk (even if failed)
    c = c + 1;
end


fprintf('knn_grid_call complete (%d×%d results)\n', N, k);
end

function [bonds, bond_lengths, connection_counts] = generate_bonds_from_knn( ...
    Points, idx, dists, target_valency, preferred_direction, direction_weight, useGPU)
%GENERATE_BONDS_FROM_KNN Memory-light, parallel greedy b-matching over KNN edges.
% - Handles 10^8+ points safely
% - Batch-based anisotropy scoring
% - Parallelized edge emission
% - GPU-accelerated bond length computation
% % Inputs:
% Points [N×3 single] Node coordinates
% idx, dists [N×K int32/single] KNN indices/distances
% target_valency Scalar, mean desired degree per node
% preferred_direction [1×3] unit vector, anisotropy bias
% direction_weight Scalar, anisotropy strength
% useGPU Logical, whether to use GPU for bond lengths
% % Outputs:
% bonds [E×2 uint32] Edge list (i,j)
% bond_lengths [E×1 single] Edge lengths
% connection_counts [N×1 int32] Node degree after selection
% -------------------------------------------------------------------------
% --- PRE-CHECKS
% -------------------------------------------------------------------------
if nargin < 7 || isempty(useGPU), useGPU = false; end
if isempty(preferred_direction), preferred_direction = [1 0 0]; end
preferred_direction = preferred_direction(:) / max(1e-12, norm(preferred_direction));
N = size(Points,1);
K = size(idx,2);
if N < 2 || K < 1
    bonds = zeros(0,2,'uint32');
    bond_lengths = zeros(0,1,'single');
    connection_counts = zeros(N,1,'int32');
    return
end
if ~isa(idx,'int32'), idx = int32(idx); end
if ~isa(dists,'single'), dists = single(dists); end
if ~isa(Points,'single'), Points = single(Points); end
% -------------------------------------------------------------------------
% --- DEGREE DISTRIBUTION (shifted geometric)
% -------------------------------------------------------------------------
p = 1 / max(target_valency - 2, 0.1);
tgt = round( geornd(p, N, 1) + target_valency );
tgt = max(1, min(tgt, K)); % 1..K
cap = int32(tgt(:));
tmax = max(cap);
% -------------------------------------------------------------------------
% --- ANISOTROPY SETUP
% -------------------------------------------------------------------------
dotPref = double(Points) * double(preferred_direction); % Nx1 projection
T = min(K, max(8, min(2*ceil(tmax), 32))); % keep top-T neighbors
batch = 5e7; % per-pass rows
keepCounts = zeros(N,1,'uint32');
% -------------------------------------------------------------------------
% --- PASS 1: LOCAL PRUNING (BATCHED, MEMORY SAFE)
% -------------------------------------------------------------------------
fprintf('Pass 1/2: evaluating top-%d neighbors in batches...\n', T);
for s = 1:batch:N
    e = min(s+batch-1, N);
    idx_b = idx(s:e,:);
    dist_b = dists(s:e,:);
    dotSeg = dotPref(s:e);
    for i = 1:size(idx_b,1)
        nei = double(idx_b(i,:));
        dj = double(dist_b(i,:));
        mask = (nei > 0) & (nei ~= s+i-1);
        if ~any(mask), continue; end
        nei = nei(mask);
        dj = dj(mask);
        if direction_weight ~= 0
            sc = dj - direction_weight * (double(dotPref(nei)) - double(dotSeg(i)));
        else
            sc = dj;
        end
        tkeep = min(T, numel(sc));
        [~, kk] = mink(sc, tkeep, 2);
        nei = nei(kk);
        keepCounts(s+i-1) = uint32(sum(nei > (s+i-1)));
    end
    clear idx_b dist_b dotSeg
end
% -------------------------------------------------------------------------
% --- ALLOCATE EDGES (BASED ON ESTIMATED TOTAL)
% -------------------------------------------------------------------------
Mcap = sum(double(keepCounts));
E_i = zeros(Mcap,1,'uint32'); %#ok<*PREALL>
E_j = zeros(Mcap,1,'uint32');
E_sc = zeros(Mcap,1,'single');
% -------------------------------------------------------------------------
% --- PASS 2: EDGE EMISSION (GPU-BATCHED, VECTORIZED, SAFE)
% -------------------------------------------------------------------------
fprintf('Starting bond emission (GPU batches, N = %d, k = %d)...\n', N, K);
batchSize = 5e7;
numChunks = ceil(N / batchSize);
fprintf('Processing %d chunks (~%.1f M points per batch)\n', numChunks, batchSize/1e6);
useAnisotropy = (direction_weight ~= 0);
if useAnisotropy
    dotPref_g = gpuArray(single(dotPref));
end
E_i = cell(numChunks,1);
E_j = cell(numChunks,1);
E_sc = cell(numChunks,1);
for c = 1:numChunks
    sIdx = (c-1)*batchSize + 1;
    eIdx = min(c*batchSize, N);
    nLoc = eIdx - sIdx + 1;
    % --- Copy batch to GPU
    idx_g = gpuArray(idx(sIdx:eIdx,:));
    dist_g = gpuArray(dists(sIdx:eIdx,:));
    % --- Compute anisotropy-weighted scores (if needed)
    if useAnisotropy
        dot_nei = dotPref_g(idx_g);
        diffe = dot_nei - dotPref_g(sIdx:eIdx);
        sc_g = dist_g - direction_weight * diffe;
    else
        sc_g = dist_g; % skip anisotropy math
    end
    % --- Bring to CPU for flexible logic (smaller now)
    sc = gather(sc_g); idxl = gather(idx_g);
    clear idx_g dist_g sc_g dot_nei diffe
    % --- Vectorized filtering
    maskValid = (idxl > 0);
    rowIdx = repelem((sIdx:eIdx)', 1, K);
    rowIdx = rowIdx(maskValid);
    colIdx = idxl(maskValid);
    scores = sc(maskValid);
    % --- keep only i < j (avoid duplicates)
    keepMask = colIdx > rowIdx;
    if ~any(keepMask)
        E_i{c} = zeros(0,1,'uint32');
        E_j{c} = zeros(0,1,'uint32');
        E_sc{c}= zeros(0,1,'single');
        fprintf(' ⚠ chunk %d/%d empty (no valid edges)\n', c, numChunks);
        continue;
    end
    rowIdx = rowIdx(keepMask);
    colIdx = colIdx(keepMask);
    scores = scores(keepMask);
    % --- Sort locally, keep top-T neighbors per node
    % group by source i, take min(T) edges
    [rowSorted, sortIdx] = sort(rowIdx);
    colSorted = colIdx(sortIdx);
    scSorted = scores(sortIdx);
    % fast grouping
    diffIdx = [true; diff(rowSorted) ~= 0];
    groupStart = find(diffIdx);
    groupEnd = [groupStart(2:end)-1; numel(rowSorted)];
    keepList = false(numel(rowSorted),1);
    for g = 1:numel(groupStart)
        g1 = groupStart(g);
        g2 = groupEnd(g);
        count = g2 - g1 + 1;
        nKeep = min(T, count);
        [~, localIdx] = mink(scSorted(g1:g2), nKeep);
        keepList(g1 - 1 + localIdx) = true;
    end
    % collect results
    E_i{c} = uint32(rowSorted(keepList));
    E_j{c} = uint32(colSorted(keepList));
    E_sc{c} = single(scSorted(keepList));
    fprintf(' ✔ chunk %d/%d complete (%d edges)\n', c, numChunks, numel(E_i{c}));
    clear rowIdx colIdx scores rowSorted colSorted scSorted
end
% --- Concatenate all edges safely ---
for c = 1:numChunks
    if isempty(E_i{c}),  E_i{c}  = zeros(0,1,'uint32'); end
    if isempty(E_j{c}),  E_j{c}  = zeros(0,1,'uint32'); end
    if isempty(E_sc{c}), E_sc{c} = zeros(0,1,'single');  end
end

E_i  = vertcat(E_i{:});
E_j  = vertcat(E_j{:});
E_sc = vertcat(E_sc{:});
fprintf('Edge emission complete: %d candidate edges\n', numel(E_i));
% -------------------------------------------------------------------------
% --- PASS 3: GLOBAL SORT + GREEDY DEGREE CONSTRAINTS
% -------------------------------------------------------------------------
[~, ord] = sort(E_sc, 'ascend');
Ei = double(E_i(ord));
Ej = double(E_j(ord));
deg = zeros(N,1,'int32');
keep = false(numel(Ei),1);
for e = 1:numel(Ei)
    a = Ei(e);
    b = Ej(e);
    if deg(a) < cap(a) && deg(b) < cap(b)
        keep(e) = true;
        deg(a) = deg(a) + 1;
        deg(b) = deg(b) + 1;
    end
end
bonds = [Ei(keep) Ej(keep)];
fprintf('Final bonds: %d kept (%.2f%%)\n', size(bonds,1), 100*size(bonds,1)/numel(Ei));
connection_counts = deg;
fprintf('Bond generation complete.\n');
% -------------------------------------------------------------------------
% --- GPU BOND LENGTH COMPUTATION
% -------------------------------------------------------------------------
if isempty(bonds)
    bond_lengths = zeros(0,1,'single');
else
    if useGPU
        Pg = gpuArray(Points);
        Lg = compute_bond_lengths_gpu(bonds, Pg, true);
        bond_lengths = gather(Lg);
    else
        dP = Points(bonds(:,1),:) - Points(bonds(:,2),:);
        bond_lengths = sqrt(sum(dP.^2,2,'native'));
    end
end
fprintf('✅ Bonds generated: %d | Mean degree: %.2f\n', size(bonds,1), mean(double(connection_counts)));
end


function [centers] = pbd_relax_centers(centers, C, R, min_sep, ~, iters, alpha_push, keep_inside)
% Centers-only Projected Dynamics: push apart overlapping pairs
% Neighbor search via knn_grid_cuda for speed (radius screen with k large)
centers = single(centers);
N = size(centers,1);
k = int32(48); % neighbor cap per point for search
for it=1:iters
    % KNN (self as query)
    try
        [nbrIdx, nbrDist] = knn_grid_cuda(single(centers), single(centers), k);
    catch % CPU fallback (slow)
        D = squareform(pdist(double(centers)));
        [nbrDist, nbrIdx] = sort(D,2,'ascend');
        nbrIdx = int32(nbrIdx(:, 2:min(end,double(k))));
        nbrDist = single(nbrDist(:,2:min(end,double(k))));
    end
    % push apart pairs under min_sep (symmetrically)
    minD = min_sep;
    for i=1:N
        js = double(nbrIdx(i,:));
        js = js(js>0 & js<=N & js~=i);
        dist = double(nbrDist(i,1:numel(js)));
        under = find(dist < minD);
        if isempty(under), continue; end
        pi = double(centers(i,:));
        for u=under
            j = js(u);
            pj = double(centers(j,:));
            v = pj - pi;
            d = norm(v) + 1e-9;
            corr = (minD - d) * (v/d) * 0.5 * alpha_push;
            pi = pi - corr; pj = pj + corr;
            centers(i,:) = single(pi);
            centers(j,:) = single(pj);
        end
    end
    % project back into sphere (optional)
    if keep_inside
        d = sqrt(sum((centers - C).^2,2));
        bad = d > (R - 0.25);
        if any(bad)
            u = bsxfun(@rdivide, (centers(bad,:) - C), d(bad) + 1e-9);
            centers(bad,:) = C + u * (R - 0.5);
        end
    end
end
end

function [Pt_local, vox] = build_canonical_rbc_cloud(range, step)
% Quartic RBC (biconcave) voxelization
d=8; br=1; h=2.12;
P=-(d^2/2) + (h^2/2)*((d^2/br^2)-1) - h^2/2*((d^2/br^2)-1)*sqrt(1 - (br^2/h^2));
Q=P*(d^2/br^2) + (br^2/4)*(d^4/br^4 - 1);
R=-P*(d^2/4) - d^4/16;
g=-range:step:range;
[x,y,z]=ndgrid(g,g,g);
eq=(x.^2+y.^2+z.^2).^2 + P*(x.^2+y.^2) + Q*(z.^2) + R;
m=(eq<=0);
Pt_local=[x(m), y(m), z(m)];
vox=size(Pt_local,1);
end

function [samples, tried] = poisson3D_in_ball(C, R, rmin, max_try)
% light 3D Poisson-disk sampler in a ball (dart throwing + grid)
samples = single(zeros(0,3));
tried = 0;
if rmin <= 0, rmin = 1; end
cell = rmin/sqrt(3);
gx = floor((2*R)/cell)+1;
gy=gx; gz=gx;
grid = -ones(gx,gy,gz, 'int32');
to_grid = @(p) max(1, min([gx gy gz], 1 + floor((double(p - (C-R))) ./ cell)));
while tried < max_try
    tried = tried + 1;
    dir = randn(1,3);
    dir=dir/norm(dir);
    rad = R * (rand()^(1/3)); p = C + single(rad*dir);
    if sum((p-C).^2) > (R - rmin)^2, continue; end
    gi = to_grid(p); ok = true;
    for ix = gi(1)-2:gi(1)+2
        if ix<1 || ix>gx, continue; end
        for iy = gi(2)-2:gi(2)+2
            if iy<1 || iy>gy, continue; end
            for iz = gi(3)-2:gi(3)+2
                if iz<1 || iz>gz, continue; end
                j = grid(ix,iy,iz);
                if j>0
                    if norm(double(samples(j,:)) - double(p)) < rmin
                        ok=false; break;
                    end
                end
            end
            if ~ok, break; end
        end
        if ~ok, break; end
    end
    if ~ok, continue; end
    samples = [samples; p]; %#ok<AGROW>
    grid(gi(1),gi(2),gi(3)) = size(samples,1);
end
end


