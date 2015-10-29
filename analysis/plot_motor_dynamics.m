function [] = plot_motor_dynamics(name, dur)
%PLOT_MOTOR_DYNAMICS Plot the muscle values of several groups of babble sims.
%   Load simulation workspace files from
%   working_directory/<group_name>_<sim#>_Workspace/babble_daspnet_multi_<group_name>_<sim#>.mat
%   and combine the salhist variables by group to be plotted and analyzed.
    
    hFig = figure();
    set(hFig, 'name', name);
    
    load([name '_Workspace/babble_daspnet_multi_' name '.mat'], 'muscleState');
    musc1 = permute(muscleState(1,:,1:dur), [3 2 1]);
    musc2 = permute(muscleState(2,:,1:dur), [3 2 1]);
    mean1 = mean(musc1, 2);
    mean2 = mean(musc2, 2);
    std1 = std(musc1, 0, 2);
    std2 = std(musc2, 0, 2);
    covariance = mean((musc1 - repmat(mean1, 1, 1000)) .* (musc2 - repmat(mean2, 1, 1000)), 2);
    correlation = covariance ./ (std1 .* std2);
    
    subplot(3, 1, 1);
    plot(10:10:dur, mean(reshape(mean1, 10, dur / 10)), 'b', 10:10:dur, mean(reshape(mean2, 10, dur / 10)), 'g');
    legend('Masseter', 'Orbicularis Oris');
    ylabel('Mean');
    xlim([0 dur]);
    subplot(3, 1, 2);
    plot(10:10:dur, mean(reshape(std1, 10, dur / 10)), 'b', 10:10:dur, mean(reshape(std2, 10, dur / 10)), 'g');
    ylabel('Std Dev.');
    xlim([0 dur]);
    subplot(3, 1, 3);
    plot(10:10:dur, mean(reshape(correlation, 10, dur / 10)), 'm');
    ylabel('Correlation');
    xlabel('Time (sec)');
    xlim([0 dur]);
    
    figure()
    
    musc1concat = reshape(musc1', 1000 * dur, 1);
    musc2concat = reshape(musc2', 1000 * dur, 1);
    
    crosscorr = zeros(dur, 2001);
    for i = 0:3
        tStart = 1 + i * (1000 * dur / 4);
        tEnd = (i + 1) * (1000 * dur / 4);
        [r, lags] = xcorr(musc1concat(tStart:tEnd), musc2concat(tStart:tEnd), 1000, 'coeff');
        subplot(4, 1, i + 1);
        %plot((tStart:tEnd) / 1000, musc1concat(tStart:tEnd), 'b', (tStart:tEnd) / 1000, musc2concat(tStart:tEnd), 'g');
        plot(lags, r);
        %axis([-1000 1000 0 0.05]);
    end
    xlabel('Orbicularis Oris Time Lag (ms)');
    ylabel('Normalized Cross Correlation');
end
