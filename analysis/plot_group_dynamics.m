function [] = plot_group_dynamics(groups, num, dur, col)
%PLOT_GROUP_DYNAMICS Plot means, standard deviations, correlations, and cross correlations
%   for simulation groups. Load simulation workspace files from <group_name>_<sim#>.mat
%   and combine the salhist variables by group to be plotted and analyzed.
    
    figure()
    hold on
    
    for i = 1:length(groups)
        for j = 1:num
            name = [groups{i}];
            load([groups{i} '_Workspace/babble_daspnet_multi_' name '.mat'], 'muscleState', 'salhist');
            musc1 = permute(muscleState(1,:,1:dur), [3 2 1]);
            musc2 = permute(muscleState(2,:,1:dur), [3 2 1]);
            m1 = mean(musc1, 2);
            m2 = mean(musc2, 2);
            scatter(m1, m2, [], salhist);
        end
    end
    
    hold off
end
