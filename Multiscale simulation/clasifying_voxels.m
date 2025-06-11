% Load data
% Assuming ClotMatrix is 636 x 240 x 240 x 240
% Composition_valid and Porosity_valid are 1 x 636
% ResponseMatrix is 636 x 6 x 20001

% Define group ranges based on the table
groups = {
    'I',  45, 95, 65, 75;
    'II', 45, 95, 85, 95;
    'III', 80, 95, 85, 95;
    'IV', 45, 60, 85, 95;
    'V', 70, 80, 65, 75;
    'VI', 70, 80, 85, 95;
    'VII', 80, 95, 65, 75;
    'IIX', 45, 60, 85, 95;
};
% Initialize group assignments
GroupAssignments = cell(1, 636);
GroupIndices = struct();

% Loop through all samples and classify
for i = 1:636
    p = Porosity_valid(i) * 100; % Convert to percentage
    c = Composition_valid(i) * 100; % Convert to percentage
    assigned_groups = {};

    for j = 1:size(groups, 1)
        por_min = groups{j, 2};
        por_max = groups{j, 3};
        comp_min = groups{j, 4};
        comp_max = groups{j, 5};

        if (p >= por_min && p <= por_max) && (c >= comp_min && c <= comp_max)
            assigned_groups{end+1} = groups{j, 1}; % Add group
            if ~isfield(GroupIndices, groups{j, 1})
                GroupIndices.(groups{j, 1}) = []; % Initialize if not existing
            end
            GroupIndices.(groups{j, 1}) = [GroupIndices.(groups{j, 1}), i]; % Store index
        end
    end
    
    if isempty(assigned_groups)
        GroupAssignments{i} = 'Unclassified';
    else
        GroupAssignments{i} = strjoin(assigned_groups, ', '); % Store multiple groups as a string
    end
end

% Paired groups
paired_groups = {'I', 'II'; 'III', 'IV'; 'V', 'VI'; 'VII', 'IIX'};

% Initialize structures to store selected indices
SelectedIndices = struct();
SelectedResponseIndices = struct();

for i = 1:size(paired_groups, 1)
    g1 = paired_groups{i, 1};
    g2 = paired_groups{i, 2};
    
    if isfield(GroupIndices, g1) && isfield(GroupIndices, g2)
        % Get the indices
        indices_g1 = GroupIndices.(g1);
        indices_g2 = GroupIndices.(g2);
        
        % Determine the minimum count for balance
        min_count = min(length(indices_g1), length(indices_g2));
        
        % Randomly select min_count samples
        selected_g1 = randsample(indices_g1, min_count);
        selected_g2 = randsample(indices_g2, min_count);
        
        % Store selected indices
        SelectedIndices.(g1) = selected_g1;
        SelectedIndices.(g2) = selected_g2;
        
        % Save the ClotMatrix for each group
        eval(sprintf('ClotMatrix_%s = ClotMatrix_valid(selected_g1, :, :, :);', g1));
        eval(sprintf('ClotMatrix_%s = ClotMatrix_valid(selected_g2, :, :, :);', g2));

        % Save the corresponding response matrix
        eval(sprintf('averaged_responses_downsampled_%s = averaged_responses_downsampled(selected_g1, :, :);', g1));
        eval(sprintf('averaged_responses_downsampled_%s = averaged_responses_downsampled(selected_g2, :, :);', g2));

        % Store the selected response indices as well
        SelectedResponseIndices.(g1) = selected_g1;
        SelectedResponseIndices.(g2) = selected_g2;
    end
end

% Save all the selected indices
save('SelectedIndices.mat', 'SelectedIndices');
save('SelectedResponseIndices.mat', 'SelectedResponseIndices');

% Display final counts
disp('Final selected counts per group:');
for i = 1:size(paired_groups, 1)
    g1 = paired_groups{i, 1};
    g2 = paired_groups{i, 2};
    fprintf('%s: %d samples, %s: %d samples\n', g1, length(SelectedIndices.(g1)), g2, length(SelectedIndices.(g2)));
end
