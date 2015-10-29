function [] = plot_salience( group_names, group_size, sim_duration, colors )
%PLOT_SALIENCE Plot the salience history of several groups of babble sims.
%   Load simulation workspace files from
%   working_directory/<group_name>_<sim#>_Workspace/babble_daspnet_multi_<group_name>_<sim#>.mat
%   and combine the salhist variables by group to be plotted and analyzed.
    
    figure()
    hold on
    for g = 1:numel(group_names)
        group_name = group_names{g};
        salience = [];
        for sim_number = 1:group_size
            load([group_name '_' num2str(sim_number) '_Workspace/babble_daspnet_multi_' ...
                group_name '_' num2str(sim_number) '.mat'], 'salhist');
            salience = [salience, salhist];
        end
        col = colors(g);
        group_salience = smooth(mean(salience, 2), 0.1, 'rlowess');
        plot(group_salience, col, 'LineWidth', 1.1);
        text(1000, 10 - 0.5 * g, ...
            [group_name ' (M=' num2str(mean2(salience)) ', SD=' ...
            num2str(std2(salience)) ')'], 'Color', col, 'FontWeight', 'bold')
    end
    xlabel('Time (sec)');
    ylabel('RMS Auditory Salience');
    hold off

end

