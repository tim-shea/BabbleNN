function [] = plot_salience( group_names, group_size, sim_duration, colors )
%PLOT_SALIENCE Plot the salience history of several groups of babble sims.
%   Load simulation workspace files from
%   working_directory/<group_name>_<sim#>_Workspace/babble_daspnet_multi_<group_name>_<sim#>.mat
%   and combine the salhist variables by group to be plotted and analyzed.
    
    figure()
    hold on
    for g = 1:numel(group_names)
        group_name = group_names{g};
        group_salience = [];
        for sim_number = 1:group_size
            load([group_name '_' num2str(sim_number) '_Workspace/babble_daspnet_multi_' ...
                group_name '_' num2str(sim_number) '.mat'], 'salhist');
            group_salience = [group_salience, salhist];
        end
        col = colors(g);
        plot(1:sim_duration, group_salience, ['.' col]);
        smooth_salience = smooth(group_salience, 20);
        plot(1:sim_duration, smooth_salience, col, 'LineWidth', 2);
        text(sim_duration / 2, 10 - g, ...
            [group_name ' (M=' num2str(mean2(group_salience)) ', SD=' ...
            num2str(std2(group_salience)) ')'], 'Color', col, 'FontWeight', 'bold')
    end
    xlabel('Time (sec)');
    ylabel('RMS Auditory Salience');
    hold off

end

