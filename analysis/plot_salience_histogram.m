function [] = plot_salience_histogram( group_names, group_size, sim_duration )
%PLOT_SALIENCE_HIST Plot the salience histogram of several groups of babble sims.
%   Load simulation workspace files from
%   working_directory/<group_name>_<sim#>_Workspace/babble_daspnet_multi_<group_name>_<sim#>.mat
%   and combine the salhist(ory) variables by group to be plotted and analyzed.
    
    figure()
    hold on
    legend_entries = {};
    for g = 1:numel(group_names)
        group_name = group_names{g};
        group_salience = [];
        for sim_number = 1:group_size
            load([group_name '_' num2str(sim_number) '_Workspace/babble_daspnet_multi_' ...
                group_name '_' num2str(sim_number) '.mat'], 'salhist');
            group_salience = [group_salience; salhist];
        end
        h = histogram(group_salience, 'BinWidth', 0.25, 'Normalization', 'probability');
        legend_entries{g} = [group_name ' (M=' num2str(mean2(group_salience)) ', SD=' ...
            num2str(std2(group_salience)) ')'];
    end
    legend(legend_entries{1}, legend_entries{2}, legend_entries{3});
    xlabel('RMS Auditory Salience');
    ylabel('Frequency');
    hold off

end

