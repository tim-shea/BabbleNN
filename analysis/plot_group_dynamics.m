function [] = plot_group_dynamics(groups, num, dur, col)
%PLOT_GROUP_DYNAMICS Plot means, standard deviations, correlations, and cross correlations
%   for simulation groups. Load simulation workspace files from <group_name>_<sim#>.mat
%   and combine the salhist variables by group to be plotted and analyzed.
    
    figure()
    hold on
    
    for i = 1:length(groups)
        for j = 1:num
            name = [groups{i} '_' num2str(j)];
            load(['Cosyne16_Sims/' name '.mat'], 'muscleState');
            musc1 = permute(muscleState(1,:,1:dur), [3 2 1]);
            musc2 = permute(muscleState(2,:,1:dur), [3 2 1]);
            m1 = mean(musc1, 2);
            m2 = mean(musc2, 2);
            plot(mean(reshape(m1, 360, dur / 360)), mean(reshape(m2, 360, dur / 360)), col(i));
        end
    end
    
    hold off
end
