function create_microstructure
% -------------------------------------------------------------------------
% PARAMETERS
% -------------------------------------------------------------------------
N        = 500;                                        % total voxels
outDir   = fullfile(pwd,'clot_volumes');                % 3-D volumes go here
metaFile = fullfile(outDir,'metadata.mat');             % metadata table
mkdir(outDir);

% Pre-allocate lightweight metadata
Porosity    = nan(N,1,'single');
Composition = nan(N,1,'single');
FileNames   = strings(N,1);

% -------------------------------------------------------------------------
% LAUNCH A POOL (reuse if one already exists)
% -------------------------------------------------------------------------
pool = gcp('nocreate');
if isempty(pool)
    pool = parpool;                                     % default profile
end

% -------------------------------------------------------------------------
% SUBMIT ALL FUTURES
% -------------------------------------------------------------------------
futs(N) = parallel.FevalFuture;                         % pre-size array
for i = 1:N
    % 3 outputs:  porosity, composition, filename
    futs(i) = parfeval(pool,@safeThrombogen,3,i,outDir);
end

% -------------------------------------------------------------------------
% COLLECT RESULTS AS THEY FINISH
% -------------------------------------------------------------------------
nDone = 0;
for k = 1:N
    [idx, por, comp, filename] = fetchNext(futs);          % blocks until 1 done
    nDone = nDone + 1;
    fprintf('Progress: %d/%d (%.1f%%)\n',nDone,N,100*nDone/N);

    if ~isnan(por) && por~=0 && ~isnan(comp) && comp~=0 && filename ~= ""
        Porosity(idx)    = por;
        Composition(idx) = comp;
        FileNames(idx)   = filename;
    end
end

% -------------------------------------------------------------------------
% POST-PROCESS & SAVE METADATA
% -------------------------------------------------------------------------
valid = ~isnan(Porosity) & Porosity~=0 & ...
        ~isnan(Composition) & Composition~=0 & ...
        FileNames ~= "";

Porosity_valid    = Porosity(valid);
Composition_valid = Composition(valid);
Files_valid       = FileNames(valid);

save(metaFile,'Porosity_valid','Composition_valid','Files_valid','-v7.3');
fprintf('\n✓ Finished %d valid volumes (saved in %s)\n',nnz(valid),outDir);
end

function [por,comp,filename] = safeThrombogen(seed,outDir)
% Runs on the worker. Returns NaNs if anything goes wrong so the main
% script can skip the result without restarting the job.

    filename = "";
    por = NaN; comp = NaN;                    % default “failure” outputs

    try
        % -----------------------------------------------------------------
        % YOUR ACTUAL GENERATOR
        % -----------------------------------------------------------------
        [por,comp,M] = Thrombogen(seed);

        % Basic validation
        if isnan(por) || por==0 || isnan(comp) || comp==0
            return                                   % leave as NaN
        end

        % Cast early to single to keep RAM low
        M = single(M);

        % -----------------------------------------------------------------
        % SAVE VOLUME TO DISK (transparency-safe)
        % -----------------------------------------------------------------
        S = struct("M",M);
        filename = fullfile(outDir,sprintf('clot_%04d.mat',seed));
        parsave(filename,S);

        % free most memory before function exits
        M = []; 
    catch ME
        % Any error (including worker OOM) drops us here
        warning('Seed %d skipped: %s',seed,ME.message);
    end
end

function parsave(fname,S)
% The -fromstruct flag keeps parfeval / parfor “workspace-transparent”.
    save(fname,"-fromstruct",S);
end
