function [] = plot_group_dynamics(group, num, dur)
%PLOT_GROUP_DYNAMICS Plot means, standard deviations, correlations, and cross correlations
%   for simulation groups. Load simulation workspace files from
%   working_directory/<group_name>_<sim#>_Workspace/babble_daspnet_multi_<group_name>_<sim#>.mat
%   and combine the salhist variables by group to be plotted and analyzed.
    
    figure()
    
    for i = 1:num
        name = [group '_' num2str(i)];
        load([name '_Workspace/babble_daspnet_multi_' name '.mat'], 'muscleState');
        musc1 = permute(muscleState(1,:,1:dur), [3 2 1]);
        musc2 = permute(muscleState(2,:,1:dur), [3 2 1]);
        m1 = mean(musc1, 2);
        m2 = mean(musc2, 2);
        %s1 = [s1, std(musc1, 0, 2)];
        %s2 = [s2, std(musc2, 0, 2)];
        %cv = [cv, mean((musc1 - repmat(m1(:,i), 1, 1000)) .* (musc2 - repmat(m2(:,i), 1, 1000)), 2)];
        %r = [r, cv(:,i) ./ (s1(:,i) .* s2(:,i))];
        %musc1concat = reshape(musc1', 1000 * dur, 1);
        %musc2concat = reshape(musc2', 1000 * dur, 1);
        %[xr_i, xlags] = xcorr(musc1concat, musc2concat, 1000, 'coeff');
        %xr = [xr, xr_i];
        
        subplot(2, 3, i);
        plot(50:50:dur, mean(reshape(m1, 50, dur / 50)), 'b', 50:50:dur, mean(reshape(m2, 50, dur / 50)), 'g');
        axis([0 dur -0.5 0.5]);
    end
end
