function [] = plot_mean_salience(group, num, dur)

    figure()
    
    group_m1 = [];
    group_m2 = [];
    group_s1 = [];
    group_s2 = [];
    group_sal = [];
    group_t = [];
    for i = 1:num
        name = [group];
        load([name '_Workspace/babble_daspnet_multi_' name '.mat'], 'muscleState', 'salhist');
        musc1 = permute(muscleState(1,:,1:dur), [3 2 1]);
        musc2 = permute(muscleState(2,:,1:dur), [3 2 1]);
        m1 = mean(musc1, 2);
        m2 = mean(musc2, 2);
        s1 = std(musc1, 0, 2);
        s2 = std(musc2, 0, 2);
        subplot(2, num / 2, i);
        scatter(m1, m2, [], (1:dur)', '.');
        %plot(smooth(salhist, 20));
        %ylim([0 25]);
        axis([-0.8 0.8 -0.8 0.8]);
        
        group_m1 = [group_m1; m1];
        group_m2 = [group_m2; m2];
        group_s1 = [group_s1; s1];
        group_s2 = [group_s2; s2];
        group_sal = [group_sal; salhist];
        group_t = [group_t; (1:dur)'];
    end
    
    figure();
    scatter(group_m1, group_m2, [], group_t, '.');
    %axis([-0.5 0.5 -0.5 0.5]);
end
