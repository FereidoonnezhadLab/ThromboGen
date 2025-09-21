function create_microstructure_bins(seedOffset)
% CREATE_MICROSTRUCTURE_BINS (random + resumable via bin_status.m) with locking
% Randomly fills porosity–composition bins in parallel; progress is stored
% in clot_volumes_bins/bin_status.m. Safe for multiple concurrent instances
% via a lock directory (bin_status.lock) used for both reading and writing.

if nargin < 1, seedOffset = 0; end

%% PARAMETERS
min_per_bin = 50;
por_bins  = linspace(0.10, 0.95, 18); % 17 bins
comp_bins = linspace(0.00, 1.00, 21); % 20 bins

PRANGE.numSpheres           = [0 800];
PRANGE.max_diameter_factor  = [0.05 1.0];
PRANGE.platelet_ratio       = [0   1.0];
PRANGE.fibrin_concentration = [0.01 6.0];
PRANGE.rbc_filling_factor   = [0   1.0];

%% OUTPUT / RESUME STATE
outDir = fullfile(pwd, 'clot_volumes_bins');
if ~isfolder(outDir), mkdir(outDir); end

binStatusFile = fullfile(outDir, 'bin_status.m');
lockDir       = fullfile(outDir, 'bin_status.lock');

% Resume: read bin_status.m under lock (or start fresh)
bin_counts = zeros(length(por_bins)-1, length(comp_bins)-1);
S = read_bin_status_locked(binStatusFile, lockDir, 30, 600); % 30s timeout, 10min stale
if ~isempty(S)
    if isfield(S,'bin_counts') && all(size(S.bin_counts)==size(bin_counts)) ...
       && isfield(S,'min_per_bin') && S.min_per_bin==min_per_bin
        bin_counts = S.bin_counts;
        fprintf('Resumed: %d/%d bins already full.\n', nnz(bin_counts>=min_per_bin), numel(bin_counts));
    else
        warning('bin_status.m present but incompatible; starting fresh.');
    end
end
% Ensure we have a status file on disk (also under lock)
write_bin_status_locked(binStatusFile, lockDir, bin_counts, por_bins, comp_bins, min_per_bin, 30, 600);

%% PARALLEL POOL
pool = gcp('nocreate');
if isempty(pool), pool = parpool; end
targetDepth = max(pool.NumWorkers * 2, pool.NumWorkers);

%% MAIN LOOP
futures = parallel.FevalFuture.empty(0,1);
seed = 1 + seedOffset;

while true
    % stop condition
    if all(bin_counts(:) >= min_per_bin)
        fprintf('\n🎉 All bins filled! (%d total)\n', numel(bin_counts));
        break
    end

    % Submit until queue is topped up
    while numel(futures) < targetDepth
        P = randomParams(PRANGE);
        futures(end+1) = parfeval(pool, @safeThrombogen, 5, seed, P); %#ok<AGROW>
        seed = seed + 1;
    end

    % Fetch next completed job (wait up to 1 s)
    [idx, por, comp, M, seed_used, P_used] = fetchNext(futures, 1);
    if isempty(idx), continue; end
    futures(idx) = [];

    % Skip invalid
    if isempty(M) || any(isnan([por comp]))
        continue
    end

    % Find bin (closed on upper edge so edges aren't lost)
    por_idx  = find(por  >= por_bins(1:end-1) & por  <= por_bins(2:end), 1);
    comp_idx = find(comp >= comp_bins(1:end-1) & comp <= comp_bins(2:end), 1);
    if isempty(por_idx) || isempty(comp_idx), continue; end

    % Save only if bin needs samples
    if bin_counts(por_idx, comp_idx) < min_per_bin
        bin_counts(por_idx, comp_idx) = bin_counts(por_idx, comp_idx) + 1;

        filename = fullfile(outDir, sprintf('clot_%06d.mat', seed_used));
        params = P_used; %#ok<NASGU>
        save(filename, 'M', 'por', 'comp', 'seed_used', 'params', '-v7.3');

        % Update the status .m file under lock
        write_bin_status_locked(binStatusFile, lockDir, bin_counts, por_bins, comp_bins, min_per_bin, 30, 600);

        fprintf('✓ saved bin (%d,%d) Por=%.3f Comp=%.3f | %d/%d bins full\n', ...
            por_idx, comp_idx, por, comp, nnz(bin_counts>=min_per_bin), numel(bin_counts));
    end
end
end

%% --- Safe wrapper around Thrombogen (5 outputs to match parfeval/fetchNext)
function [por, comp, M, seed_out, P_used] = safeThrombogen(seed, P_in)
seed_out = seed; por = NaN; comp = NaN; M = []; P_used = P_in;
try
    [por, comp, M, ~] = Thrombogen(seed, P_in);
catch ME
    warning('Seed %d failed: %s', seed, ME.message);
    M = [];
end
end

%% --- Random parameter generator
function P = randomParams(R)
P.numSpheres           = randi([R.numSpheres(1), R.numSpheres(2)]);
P.max_diameter_factor  = randIn(R.max_diameter_factor);
P.platelet_ratio       = randIn(R.platelet_ratio);
P.fibrin_concentration = randIn(R.fibrin_concentration);
P.rbc_filling_factor   = randIn(R.rbc_filling_factor);
end

function x = randIn(rr), x = rr(1) + (rr(2)-rr(1))*rand(); end

%% =================== LOCKED READ/WRITE HELPERS ===================

function S = read_bin_status_locked(binStatusFile, lockDir, timeout_s, stale_s)
% Acquire the lock to ensure we don't read during a write, then read.
lock = acquire_lock(lockDir, timeout_s, stale_s);
cleanupObj = onCleanup(@() release_lock(lockDir, lock));
if exist(binStatusFile, 'file')
    S = bin_status();  % call the function file
else
    S = [];
end
end

function write_bin_status_locked(binStatusFile, lockDir, bin_counts, por_bins, comp_bins, min_per_bin, timeout_s, stale_s)
% Acquire the lock, write to a temp file, then atomically replace the target.
lock = acquire_lock(lockDir, timeout_s, stale_s);
cleanupObj = onCleanup(@() release_lock(lockDir, lock));

% write to temp, then move (best-effort atomic replace)
tmpFile = [binStatusFile, '.tmp_', char(java.util.UUID.randomUUID())];
write_bin_status(tmpFile, bin_counts, por_bins, comp_bins, min_per_bin);
[ok, msg] = movefile(tmpFile, binStatusFile, 'f');
if ~ok
    % clean up temp if move failed
    if exist(tmpFile,'file'), delete(tmpFile); end
    error('Failed to update bin status: %s', msg);
end
end

function lock = acquire_lock(lockDir, timeout_s, stale_s)
% Try to create the lock directory. If it exists, wait until released or stale.
t0 = tic;
lock = struct('owner','', 'time', now);
while true
    [ok, msg, msgid] = mkdir(lockDir);
    if ok
        % We own the lock. Write an owner file (optional; helps debugging).
        fid = fopen(fullfile(lockDir,'owner.txt'),'w');
        if fid>0
            fprintf(fid, 'pid=%d time=%s host=%s\n', feature('getpid'), datestr(now,31), getHostname());
            fclose(fid);
        end
        return
    end
    % Check stale (directory older than stale_s seconds)
    if exist(lockDir,'dir')
        d = dir(lockDir);
        % Use the newest timestamp inside the lock dir if present
        lockAge_s = (now - max([d.datenum])) * 86400;
        if lockAge_s > stale_s
            warning('Lock appears stale (%.0fs). Forcing removal.', lockAge_s);
            try, rmdir(lockDir, 's'); catch, end
            pause(0.2);
            continue
        end
    end
    if toc(t0) > timeout_s
        error('Timeout acquiring lock %s (waited %.1fs).', lockDir, toc(t0));
    end
    pause(0.1 + 0.2*rand); % jitter to reduce contention
end
end

function release_lock(lockDir, ~)
% Best-effort release.
if exist(lockDir,'dir')
    try, rmdir(lockDir, 's'); catch, end
end
end

function h = getHostname()
try
    h = char(java.net.InetAddress.getLocalHost.getHostName);
catch
    h = 'unknown';
end
end

%% --------- status file writer (human-readable .m you can run) ----------
function write_bin_status(targetFile, bin_counts, por_bins, comp_bins, min_per_bin)
fid = fopen(targetFile, 'w');
assert(fid>0, 'Cannot write %s', targetFile);

filled = bin_counts >= min_per_bin;
rows = size(filled,1); cols = size(filled,2);
ascii = strings(rows,1);
for i = 1:rows
    line = repmat('.',1,cols);
    line(find(filled(i,:))) = 'F'; %#ok<FNDSB>
    ascii(i) = line;
end

fprintf(fid, 'function S = bin_status()\n');
fprintf(fid, '%% Auto-generated. Shows/resumes bin filling progress.\n');
fprintf(fid, 'S.min_per_bin = %d;\n', min_per_bin);
fprintf(fid, 'S.por_bins = [%s];\n', numList(por_bins));
fprintf(fid, 'S.comp_bins = [%s];\n', numList(comp_bins));
fprintf(fid, 'S.bin_counts = [%s];\n', matList(bin_counts));
fprintf(fid, 'S.filled = S.bin_counts >= S.min_per_bin;\n');
fprintf(fid, 'S.timestamp = ''%s'';\n', datestr(now, 31));
fprintf(fid, 'disp(''bin_status: F = full, . = not full'');\n');
for i = 1:rows
    fprintf(fid, 'disp(''%s'');\n', ascii(i));
end
fprintf(fid, 'fprintf(''Filled bins: %%d/%%d\\n'', nnz(S.filled), numel(S.filled));\n');
fprintf(fid, 'end\n');
fclose(fid);
end

function s = numList(v)
s = sprintf('%.10g ', v);
s = strtrim(s);
end

function s = matList(M)
[r,c] = size(M);
rows = strings(r,1);
for i = 1:r
    rows(i) = sprintf('%.10g ', M(i,:));
    rows(i) = "[" + strtrim(rows(i)) + "]";
end
s = strjoin(rows.', '; ');
end
