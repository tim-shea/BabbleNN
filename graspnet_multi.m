function [] = graspnet_multi(id, duration, reinforcer, yoke, plotOn)
%   GRASPNET_MULTI
%   Neural network model of the development of multi-dimensional targeted movements.
%   
%   Description of Input Arguments:
%       id          Unique identifier for this simulation. Must not contain white space.
%       duration    Duration of experiment in seconds. Can be 'delta_error'.
%       reinforcer  Type of reinforcement.
%       yoke        Indicates whether to run an experiment or yoked control simulation. Set false to run a regular simulation.
%                   Set true to run a yoked control. There must be a saved workspace of the same name on the MATLAB path
%                   for the simulation to yoke to.
%       plotOn      Set true to plot spikes, motor states, and motor synapse weights for each vocalization.
%
%   Example of Use:
%   graspnet_multi('Mortimer', 7200, 'delta_error', false, false);
%
%   For updates, see https://github.com/tim-shea/BabbleNN

    % Initialization
    rng shuffle;
    
    proprioception = true;
    dopamineDelay = 100;
    dopamineScale = 500;
    maximumSynapticWeight = 4;
    muscleScale = 0.2;
    muscleSmooth = 0.01;
    excInhRatio = 4;
    
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
    wavdir = [id '/wav'];
    spkdir = [id '/spk'];
    if ~exist(id, 'dir')
        mkdir(id);
        mkdir(wavdir);
        mkdir(spkdir);
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
    addpath(wavdir);
    addpath(spkdir);
    workspace = [id '/' id '.mat'];
    
    % Initialize network configuration
    numberOfMuscles = 1;
    numberOfGroups = 2 * numberOfMuscles;
    groupSize = 400;
    NMot = groupSize * numberOfGroups;
    NExc = NMot;
    NInh = NMot / excInhRatio;
    
    SExcMot = 200;
    SExcInh = SExcMot / excInhRatio;
    SInhMot = SExcMot;
    maximumDelay = 1;
    
    groupSpikeCounts = zeros(numberOfGroups);
    muscleDeltas = zeros(numberOfMuscles, 1);
    muscleStates = zeros(duration, 1001, numberOfMuscles);
    targets = rand(numberOfMuscles, 1);
    error = zeros(1000, numberOfMuscles);
    smoothDeltaError = 0;
    DA = zeros(1000 + dopamineDelay, 1);
    error_history = zeros(duration, numberOfMuscles);
    da_history = zeros(duration);
    
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
    wInhMot = -rand(NInh, SInhMot);
    
    stdpPre = zeros(NExc, 1001 + maximumDelay);
    stdpInh = zeros(NInh, 1001 + maximumDelay);
    stdpMot = zeros(NMot, 1001 + maximumDelay);
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
    
    % Neuron spikes for the current second
    spikesExc = [-maximumDelay 0];
    spikesInh = [-maximumDelay 0];
    spikesMot = [-maximumDelay 0];
    
    % Initialize duration and reward variables
    if yoke
        load(yokeWorkspace, 'T');
    else
        T = duration;
    end
    
    saveInterval = 100;
    
    if plotOn
        hNetworkActivity = figure();
        hSynapseMatrix = figure();
        hErrorReward = figure();
    end
    
    % Run the simulation
    sec = 0;
    for sec = (sec + 1):T
        
        display('********************************************');
        display(['Second ' num2str(sec) ' of ' num2str(T)]);
        
        % Simulate 1000 millisecond timesteps for each second
        for t = 1:1000
            
            % Random Thalamic Input.
            IExc = 12.5 * (rand(NExc, 1) - 0.5);
            IInh = 12.5 * (rand(NInh, 1) - 0.5);
            IMot = 11 * (rand(NMot, 1) - 0.5);
            
            % Indices of fired neurons
            firedExc = find(vExc >= 30);
            firedInh = find(vInh >= 30);
            firedMot = find(vMot >= 30);
            
            % Reset the voltages for those neurons that fired
            vExc(firedExc) = -65;
            vInh(firedInh) = -65;
            vMot(firedMot) = -65;
            uExc(firedExc) = uExc(firedExc) + dExc;
            uInh(firedInh) = uInh(firedInh) + dInh;
            uMot(firedMot) = uMot(firedMot) + dMot;
            
            % Spike-timing dependent plasticity computations
            stdpPre(firedExc, t + maximumDelay) = 1.0;
            stdpInh(firedInh, t + maximumDelay) = 1.0;
            stdpMot(firedMot, t + maximumDelay) = 1.0;
            
            % Update eligibility traces
            for k = 1:length(firedExc)
                n = firedExc(k);
                eligExcInh(n,:) = eligExcInh(n,:) - 1.25 * stdpInh(postExcInh(n,:),t)';
                eligExcMot(n,:) = eligExcMot(n,:) - 1.25 * stdpMot(postExcMot(n,:),t)';
            end
            
            for k = 1:length(firedInh)
                synInd = find(postExcInh == firedInh(k));
                eligExcInh(synInd) = eligExcInh(synInd) + 1.0 * stdpPre(ceil(synInd / SExcInh),t);
            end
            
            for k = 1:length(firedMot)
                synInd = find(postExcMot == firedMot(k));
                eligExcMot(synInd) = eligExcMot(synInd) + 1.0 * stdpPre(ceil(synInd / SExcMot),t);
            end
            
            % Update the record of when neuronal firings occurred
            spikesExc = [spikesExc; t * ones(length(firedExc), 1), firedExc];
            spikesInh = [spikesInh; t * ones(length(firedInh), 1), firedInh];
            spikesMot = [spikesMot; t * ones(length(firedMot), 1), firedMot];
            
            % Add proprioceptive inputs to excitatory inputs
            if proprioception
                responses = 1.5 * sin(2 * pi * (muscleStates(sec,t,1) - ((1:NExc)' / NExc)));
                IExc(1:NExc) = IExc(1:NExc) + responses;
            end
            
            % Calculate post synaptic conductances
            k = size(spikesExc, 1);
            while spikesExc(k,1) > t - maximumDelay
                synInhInd = postExcInh(spikesExc(k,2),:);
                synMotInd = postExcMot(spikesExc(k,2),:);
                IInh(synInhInd) = IInh(synInhInd) + wExcInh(spikesExc(k,2))';
                IMot(synMotInd) = IMot(synMotInd) + wExcMot(spikesExc(k,2))';
                k = k - 1;
            end
            
            k = size(spikesInh, 1);
            while spikesInh(k,1) > t - maximumDelay
                synInd = postInhMot(spikesInh(k,2),:);
                IMot(synInd) = IMot(synInd) + wInhMot(spikesInh(k,2))';
                k = k - 1;
            end
            
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
            stdpPre(:,t + maximumDelay + 1) = 0.95 * stdpPre(:,t + maximumDelay);
            stdpInh(:,t + maximumDelay + 1) = 0.95 * stdpInh(:,t + maximumDelay);
            stdpMot(:,t + maximumDelay + 1) = 0.95 * stdpMot(:,t + maximumDelay);
            
            % Apply spike timing dependent plasticity to synapses
            if (mod(t,10) == 0)
                wExcInh = max(0, min(maximumSynapticWeight, wExcInh + (DA(t) + 0.002) * eligExcInh));
                wExcMot = max(0, min(maximumSynapticWeight, wExcMot + (DA(t) + 0.002) * eligExcMot));
                % Apply eligibility decay
                eligExcInh = 0.95 * eligExcInh;
                eligExcMot = 0.95 * eligExcMot;
            end
            
            % Calculate muscle group activation levels based on spike counts
            for g = 1:numberOfGroups
                groupIndices = ((g - 1) * groupSize + 1):(g * groupSize);
                groupSpikeCounts(g) = sum(vMot(groupIndices) > 30);
            end
            
            spikeDeltas = groupSpikeCounts(1:2:numberOfGroups) - groupSpikeCounts(2:2:numberOfGroups);
            muscleDeltas = muscleSmooth * (muscleScale * spikeDeltas - muscleDeltas + 0.5 * (rand(numberOfMuscles, 1) - 0.5));
            muscleStates(sec,t+1,:) = min(max(muscleStates(sec,t,:) + muscleDeltas, 0), 1);
            error(t,:) = abs(muscleStates(sec,t+1,:) - targets);
            if and(strcmp(reinforcer, 'delta_error'), t > 1)
                deltaError = sum(error(t,:)) - sum(error(t-1,:));
                smoothDeltaError = smoothDeltaError + 0.01 * (deltaError - smoothDeltaError);
                DA(t+dopamineDelay) = -smoothDeltaError * dopamineScale;
            end
            
            % Print an update
            if t == 1000
                display(['Excitatory Firing Rate: ' num2str(size(spikesExc, 1) / size(vExc, 1))]);
                display(['Inhibitory Firing Rate: ' num2str(size(spikesInh, 1) / size(vInh, 1))]);
                display(['Motor Firing Rate: ' num2str(size(spikesMot, 1) / size(vMot, 1))]);
                error_history(sec,:) = mean(sum(error, 2));
                da_history(sec) = mean(DA);
                display(['Mean Error: ' num2str(error_history(sec))]);
                display(['Mean DA: ' num2str(da_history(sec))]);
            end
        end
        
        % Write spikes for this second to a text file
        if mod(sec, saveInterval) == 0 || sec == T
            display('Saving spikes to file. Do not exit program.');
            spikes_fid = fopen([spkdir '/spikes_' num2str(sec) '.txt'],'w');
            for spike = 1:size(spikesExc, 1)
                fprintf(spikes_fid, '%i\t', sec);
                fprintf(spikes_fid, '%i\t%i', spikesExc(spike,:));
                fprintf(spikes_fid, '\n');
            end
            fclose(spikes_fid);
            save(workspace)
        end
        
        % Plot reservoir, output, and motor spikes, muscle states, and motor synapses
        if plotOn
            set(0, 'currentfigure', hNetworkActivity);
            set(hNetworkActivity, 'name', ['Neural Spiking for Second: ', num2str(sec)], 'numbertitle', 'off');
            subplot(3,1,1);
            % Update spike raster plots
            plot([spikesExc(:,1); spikesInh(:,1)], [spikesExc(:,2); (spikesInh(:,2) + NExc)], '.');
            title('Exc/Inh Spike Raster Plot', 'fontweight', 'bold');
            axis([0 1000 0 NExc + NInh]);
            ylabel('Neuron Index');
            subplot(3,1,2);
            plot(spikesMot(:,1), spikesMot(:,2), '.');
            title('Motor Spike Raster Plot', 'fontweight', 'bold');
            axis([0 1000 0 NMot]);
            ylabel('Neuron Index');
            subplot(3,1,3);
            muscles = permute(muscleStates(sec,1:1000,:), [2 3 1]);
            plot([muscles, repmat(targets, 1000, 1), DA(1:1000)]);
            title('Motor Group Activity', 'fontweight', 'bold');
            xlabel('Time (ms)');
            ylabel('Activity');
            
            set(0, 'currentfigure', hSynapseMatrix);
            set(hSynapseMatrix, 'name', ['Synaptic Strengths for Second: ', num2str(sec)], 'numbertitle', 'off');
            imagesc([wExcMot wExcInh]);
            set(gca, 'YDir', 'normal');
            colorbar;
            title('Synapse Strength from Excitatory Neurons', 'fontweight', 'bold');
            xlabel('Post Synaptic Motor Neuron Index');
            ylabel('Synapse Index');
            
            set(0, 'currentfigure', hErrorReward);
            set(hErrorReward, 'name', 'Error and Reward', 'numbertitle', 'off');
            subplot(2,1,1);
            plot(1:sec, error_history(1:sec), '.b');
            title('Mean Error', 'fontweight', 'bold');
            ylabel('Error');
            subplot(2,1,2);
            plot(1:sec, da_history(1:sec));
            title('Reward', 'fontweight', 'bold');
            xlabel('Time (s)');
            ylabel('Reward');
            
            drawnow;
        end
        
        % Prepare for the following 1000 ms
        stdpPre(:,1:maximumDelay + 1) = stdpPre(:,1001:1001 + maximumDelay);
        stdpInh(:,1:maximumDelay + 1) = stdpInh(:,1001:1001 + maximumDelay);
        stdpMot(:,1:maximumDelay + 1) = stdpMot(:,1001:1001 + maximumDelay);
        DA(1:dopamineDelay) = DA(1001:1000+dopamineDelay);
        
        muscleStates(sec+1,1,:) = muscleStates(sec,1001,:);
        
        indExc = find(spikesExc(:,1) > 1001 - maximumDelay);
        spikesExc = [-maximumDelay 0; spikesExc(indExc,1) - 1000, spikesExc(indExc,2)];
        indInh = find(spikesInh(:,1) > 1001 - maximumDelay);
        spikesInh = [-maximumDelay 0; spikesInh(indInh,1) - 1000, spikesInh(indInh,2)];
        indMot = find(spikesMot(:,1) > 1001 - maximumDelay);
        spikesMot = [-maximumDelay 0; spikesMot(indMot,1) - 1000, spikesMot(indMot,2)];
    end
end
