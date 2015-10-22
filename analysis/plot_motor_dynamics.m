function [] = plot_motor_dynamics( name, sim_duration )
%PLOT_MOTOR_DYNAMICS Plot the muscle values of several groups of babble sims.
%   Load simulation workspace files from
%   working_directory/<group_name>_<sim#>_Workspace/babble_daspnet_multi_<group_name>_<sim#>.mat
%   and combine the salhist variables by group to be plotted and analyzed.
    
    figure()
    
    load([name '_Workspace/babble_daspnet_multi_' name '.mat'], 'muscleState');
    musc1 = permute(muscleState(1,:,1:sim_duration), [3 2 1]);
    musc2 = permute(muscleState(2,:,1:sim_duration), [3 2 1]);
    mean1 = mean(musc1, 2);
    mean2 = mean(musc2, 2);
    std1 = std(musc1, 0, 2);
    std2 = std(musc2, 0, 2);
    covariance = mean((musc1 - repmat(mean1, 1, 1000)) .* (musc2 - repmat(mean2, 1, 1000)), 2);
    correlation = covariance ./ (std1 .* std2);
    
    subplot(3, 1, 1);
    plot(1:sim_duration, smooth(mean1, 20), 'b', 1:sim_duration, smooth(mean2, 20), 'g', 'LineWidth', 1.5);
    ylabel('Mean');
    subplot(3, 1, 2);
    plot(1:sim_duration, smooth(std1, 20), 'b', 1:sim_duration, smooth(std2, 20), 'g', 'LineWidth', 1.5);
    ylabel('Std Dev.');
    subplot(3, 1, 3);
    plot(1:sim_duration, smooth(correlation, 20), 'm', 'LineWidth', 1.5);
    ylabel('Correlation');
    xlabel('Time (sec)');
    
    figure()
    
    musc1concat = reshape(musc1, 1000 * sim_duration, 1);
    musc2concat = reshape(musc2, 1000 * sim_duration, 1);
    
    crosscorr = zeros(sim_duration, 2001);
    for i = 0:3
        tStart = 1 + i * (1000 * sim_duration / 4);
        tEnd = (i + 1) * (1000 * sim_duration / 4);
        [r, lags] = xcorr(musc1concat(tStart:tEnd), musc2concat(tStart:tEnd), 1000, 'coeff');
        subplot(4, 1, i + 1);
        plot(lags, smooth(r, 10));
    end
    xlabel('Orbicularis Oris Time Lag (ms)');
    ylabel('Normalized Cross Recurrence');
end

