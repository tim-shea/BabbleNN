function [] = graspnet_multi(id, trials, yoke, plotOn)
%   GRASPNET_MULTI
%   Neural network model of the development of multi-dimensional targeted movements.
%   
%   Description of Input Arguments:
%       id          Unique identifier for this simulation. Must not contain white space.
%       trials      Number of trials for the run.
%       yoke        Indicates whether to run an experiment or yoked control simulation. Set false to run a regular simulation.
%                   Set true to run a yoked control. There must be a saved workspace of the same name on the MATLAB path
%                   for the simulation to yoke to.
%       plotOn      Set true to plot spikes, motor states, and synapse weights for each trial.
%
%   Example of Use:
%   graspnet_multi('Mortimer', 600, false, false);
%
%   For updates, see https://github.com/tim-shea/BabbleNN

    % Initialization
    rng shuffle;
    
    proprioception = true;
    numberOfMuscles = 2;
    motorScale = 0.03;
    groupSize = 800;
    excInhRatio = 4;
    SExcMot = 200;
    maximumSynapticWeight = 4;
    
    trialDuration = 2000;
    interTrialInterval = 500;
    
    errorThreshold = 0.5;
    rewardCountThreshold = 3;
    errorThresholdScaling = 0.99;
    dopamineIncrement = 10.0;
    dopamineDelay = 100;
    
    % Check simulation id for spaces
    if any(isspace(id))
        disp('Simulation id may not contain spaces.');
        return
    end
    
    % Check yoke directory
    if yoke
        if ~exist(id, 'dir')
            disp(['Directory ' id ' does not exist. Cannot continue yoked simulation.']);
            return
        end
        yokeId = id
        yokeWorkspace = [yokeId '\' yokeId '.mat']
        id = [id '_yoke'];
    end
    
    % Setup data directories
    if ~exist(id, 'dir')
        mkdir(id);
    else
        overwrite = '';
        while ~strcmpi(overwrite, 'y')
            overwrite = input(['Directory ' id ' already exists. Press [y] to confirm overwrite or [n] to exit: '], 's');
            if overwrite == 'n'
                return
            end
        end
    end
    addpath(id);
    workspace = [id '/' id '.mat'];
    
    % Initialize network configuration
    NMot = groupSize * numberOfMuscles;
    NExc = NMot;
    NInh = NMot / excInhRatio;
    
    SExcInh = SExcMot / excInhRatio;
    SInhMot = SExcMot;
    
    motorTargets = zeros(numberOfMuscles, 1);
    motorDeltas = zeros(trials, trialDuration + 1, numberOfMuscles);
    motorStates = zeros(trials, trialDuration + 1, numberOfMuscles);
    
    error = zeros(trials, 1);
    rewards = zeros(trials, 1);
    
    % Set time scales of membrane recovery variable
    aExc = 0.02;
    aInh = 0.1;
    aMot = 0.02;
    % Membrane recovery variable after-spike shift
    dExc = 8;
    dInh = 2;
    dMot = 8;
    
    % Generate sparse synaptic connections
    postExcMot = ceil(NMot * rand(NExc, SExcMot));
    postExcInh = ceil(NInh * rand(NExc, SExcInh));
    postInhMot = mod(repmat(1:SInhMot, NInh, 1) + repmat((0:excInhRatio:(NMot - 1))', 1, SInhMot), NMot) + 1;
    
    wExcMot = rand(NExc, SExcMot);
    wExcInh = rand(NExc, SExcInh);
    wInhMot = -2 * ones(NInh, SInhMot);
    
    stdpExc = zeros(NExc, 1);
    stdpInh = zeros(NInh, 1);
    stdpMot = zeros(NMot, 1);
    eligExcInh = zeros(NExc, SExcInh);
    eligExcMot = zeros(NExc, SExcMot);
    
    % Initialize membrane potentials
    vExc = -65 * ones(NExc, 1);
    vInh = -65 * ones(NInh, 1);
    vMot = -65 * ones(NMot, 1);
    % Initialize adaptation variable
    uExc = 0.2 .* vExc;
    uInh = 0.2 .* vInh;
    uMot = 0.2 .* vMot;
    
    % Initialize duration and reward variables
    if yoke
        load(yokeWorkspace, 'trials', 'rewards');
    end
    
    saveInterval = 100;
    
    if plotOn
        hNetworkActivity = figure();
        hSynapseMatrix = figure();
        hErrorReward = figure();
    end
    
    % Run the simulation
    for trial = 1:trials
        
        display('********************************************');
        display(['Trial ' num2str(trial) ' of ' num2str(trials)]);
        
        % Spikes for the current time step
        spikesExc = [];
        spikesInh = [];
        spikesMot = [];
        
        % Record spike history for the current trial
        excSpikeRaster = [];
        inhSpikeRaster = [];
        motSpikeRaster = [];
        
        % Zero the dopamine trace for the current trial
        DA = 0;
        
        % Reset motor states for this trial
        motorStates(trial,1,:) = 2 * rand(numberOfMuscles, 1) - 1;
        
        % Simulate 1 millisecond timesteps for each trial
        for t = 1:trialDuration
            
            % Generate random thalamic input
            IExc = 11 * (rand(NExc, 1) - 0.5);
            IInh = 12 * (rand(NInh, 1) - 0.5);
            IMot = 11 * (rand(NMot, 1) - 0.5);
            
            % Apply spikes from the previous time step
            for k = 1:length(spikesExc)
                n = spikesExc(k);
                eligExcInh(n,:) = eligExcInh(n,:) - 1.5 * stdpInh(postExcInh(n,:))';
                eligExcMot(n,:) = eligExcMot(n,:) - 1.5 * stdpMot(postExcMot(n,:))';
            end
            
            for k = 1:length(spikesInh)
                synInd = postExcInh == spikesInh(k);
                [rows, ~] = find(synInd);
                eligExcInh(synInd) = eligExcInh(synInd) + 1.0 * stdpExc(rows);
            end
            
            for k = 1:length(spikesMot)
                synInd = postExcMot == spikesMot(k);
                [rows, ~] = find(synInd);
                eligExcMot(synInd) = eligExcMot(synInd) + 1.0 * stdpExc(rows);
            end
            
            % Add proprioceptive inputs to excitatory inputs
            if proprioception
                receptiveFields = repmat(2 * (1:groupSize)' / groupSize - 1, numberOfMuscles, 1);
                motorValues = reshape(repmat(motorStates(trial,t,:), groupSize, 1), NMot, 1);
                responses = max(2 - 6 * abs(receptiveFields - motorValues), 0);
                IExc(1:NExc) = IExc(1:NExc) + responses;
            end
            
            % Calculate post synaptic conductances
            for i = 1:size(spikesExc)
                n = spikesExc(i);
                synInhInd = postExcInh(n,:);
                synMotInd = postExcMot(n,:);
                IInh(synInhInd) = IInh(synInhInd) + wExcInh(n)';
                IMot(synMotInd) = IMot(synMotInd) + wExcMot(n)';
            end
            
            for i = 1:size(spikesInh)
                n = spikesInh(i);
                synInd = postInhMot(n,:);
                IMot(synInd) = IMot(synInd) + wInhMot(n)';
            end
            
            % Detect spikes in the current time step
            spikesExc = find(vExc > 30);
            spikesInh = find(vInh > 30);
            spikesMot = find(vMot > 30);
            
            % Reset the voltages for those neurons that fired
            vExc(spikesExc) = -65;
            vInh(spikesInh) = -65;
            vMot(spikesMot) = -65;
            uExc(spikesExc) = uExc(spikesExc) + dExc;
            uInh(spikesInh) = uInh(spikesInh) + dInh;
            uMot(spikesMot) = uMot(spikesMot) + dMot;
            
            % Apply current spikes to presynaptic STDP traces
            stdpExc(spikesExc) = 1.0;
            stdpInh(spikesInh) = 1.0;
            stdpMot(spikesMot) = 1.0;
            
            % Update neuronal membrane potentials
            vExc = vExc + 0.5 * ((0.04 * vExc + 5) .* vExc + 140 - uExc + IExc);
            vExc = vExc + 0.5 * ((0.04 * vExc + 5) .* vExc + 140 - uExc + IExc);
            vInh = vInh + 0.5 * ((0.04 * vInh + 5) .* vInh + 140 - uInh + IInh);
            vInh = vInh + 0.5 * ((0.04 * vInh + 5) .* vInh + 140 - uInh + IInh);
            vMot = vMot + 0.5 * ((0.04 * vMot + 5) .* vMot + 140 - uMot + IMot);
            vMot = vMot + 0.5 * ((0.04 * vMot + 5) .* vMot + 140 - uMot + IMot);
            
            % Update neuronal adaptation variables
            uExc = uExc + aExc .* (0.2 * vExc - uExc);
            uInh = uInh + aInh .* (0.2 * vInh - uInh);
            uMot = uMot + aMot .* (0.2 * vMot - uMot);
            
            % Exponential decay of the traces of presynaptic neuron firing
            stdpExc = 0.95 * stdpExc;
            stdpInh = 0.95 * stdpInh;
            stdpMot = 0.95 * stdpMot;
            
            % Apply spike timing dependent plasticity to synapses
            if (mod(t,10) == 0)
                wExcInh = max(0, min(maximumSynapticWeight, wExcInh + (DA + 0.002) * eligExcInh));
                wExcMot = max(0, min(maximumSynapticWeight, wExcMot + (DA + 0.002) * eligExcMot));
                % Apply eligibility decay
                eligExcInh = 0.95 * eligExcInh;
                eligExcMot = 0.95 * eligExcMot;
            end
            
            motorFiringRates = zeros(groupSize, numberOfMuscles);
            motorFiringRates(spikesMot) = 1;
            responses = repmat([-ones(groupSize / 2, 1); ones(groupSize / 2, 1)], 1, numberOfMuscles);
            motorDeltas(trial,t,:) = reshape(motorScale * sum(responses .* motorFiringRates, 1), 1, 1, numberOfMuscles);
            motorStates(trial,t+1,:) = min(max(motorStates(trial,t,:) + motorDeltas(trial,t,:), -1), 1);
            
            if t == trialDuration - interTrialInterval
                error(trial) = sum(abs(motorStates(trial,:,1) - motorTargets(1))) / (trialDuration - interTrialInterval);
                if ~yoke && error(trial) < errorThreshold
                    rewards(trial) = 1;
                    if trial > 10 && sum(rewards(trial - 10:trial)) == rewardCountThreshold
                        errorThreshold = errorThreshold * errorThresholdScaling;
                    end
                    DA = DA + dopamineIncrement;
                end
            end
            
            DA = DA * 0.99;
            
            % Update the history of spikes for the current trial
            excSpikeRaster = [excSpikeRaster; t * ones(length(spikesExc), 1), spikesExc];
            inhSpikeRaster = [inhSpikeRaster; t * ones(length(spikesInh), 1), spikesInh];
            motSpikeRaster = [motSpikeRaster; t * ones(length(spikesMot), 1), spikesMot];
        end
        
        % Print an update
        display(['Excitatory Firing Rate: ' num2str(size(excSpikeRaster, 1) / size(vExc, 1))]);
        display(['Inhibitory Firing Rate: ' num2str(size(inhSpikeRaster, 1) / size(vInh, 1))]);
        display(['Motor Firing Rate: ' num2str(size(motSpikeRaster, 1) / size(vMot, 1))]);
        display(['Mean Error: ' num2str(error(trial))]);
        display(['Error Threshold: ' num2str(errorThreshold)]);
        
        % Plot reservoir, output, and motor spikes, muscle states, and motor synapses
        if plotOn
            set(0, 'currentfigure', hNetworkActivity);
            set(hNetworkActivity, 'name', ['Neural Spiking for Trial: ', num2str(trial)], 'numbertitle', 'off');
            subplot(3, 1, 1);
            % Update spike raster plots
            plot([excSpikeRaster(:,1); inhSpikeRaster(:,1)], [excSpikeRaster(:,2); (inhSpikeRaster(:,2) + NExc)], '.');
            title('Exc/Inh Spike Raster Plot', 'fontweight', 'bold');
            axis([0 trialDuration 0 NExc + NInh]);
            ylabel('Neuron Index');
            
            subplot(3, 1, 2);
            plot(motSpikeRaster(:,1), motSpikeRaster(:,2), '.');
            title('Motor Spike Raster Plot', 'fontweight', 'bold');
            axis([0 trialDuration 0 NMot]);
            ylabel('Neuron Index');
            
            subplot(3, 1, 3);
            plot(1:trialDuration, permute(motorStates(trial,1:trialDuration,:), [2 3 1]), '-', ...
                [ones(1, numberOfMuscles); trialDuration * ones(1, numberOfMuscles)], ...
                [motorTargets' .* ones(1, numberOfMuscles); motorTargets' .* ones(1, numberOfMuscles)], '--', ...
                'LineWidth', 1.25);
            title('Motor Group Activity', 'fontweight', 'bold');
            axis([0 trialDuration -1 1]);
            xlabel('Time (ms)');
            ylabel('Activity');
            
            set(0, 'currentfigure', hSynapseMatrix);
            set(hSynapseMatrix, 'name', ['Synaptic Strengths for Trials: ' num2str(trial)], 'numbertitle', 'off');
            imagesc([wExcMot wExcInh]);
            set(gca, 'YDir', 'normal');
            colorbar;
            title('Synapse Strength from Excitatory Neurons', 'fontweight', 'bold');
            xlabel('Synapse Index');
            ylabel('Excitatory Neuron Index');
            
            set(0, 'currentfigure', hErrorReward);
            set(hErrorReward, 'name', 'Error and Reward', 'numbertitle', 'off');
            subplot(2, 1, 1);
            plot(error(1:trial), '.r');
            axis([0 trial 0 1]);
            xlabel('Trial');
            ylabel('Mean Error');
            subplot(2, 1, 2);
            plot(cumsum(rewards(1:trial,1)));
            xlabel('Trial');
            ylabel('Cumulative Reward');
            
            drawnow;
        end
        
        % Periodically save the workspace
        if mod(trial, saveInterval) == 0 || trial == trials
            display('Saving workspace. Do not exit program.');
            save(workspace)
        end
    end
end
