function [] = contrastive_hebbian_graspnet(id, duration, target_function, plotOn)
%   CONTRASTIVE_HEBBIAN_GRASPNET
%   Neural network model of the error-driven learning of targeted movements.
%   
%   Description of Input Arguments:
%       id          Unique identifier for this simulation. Must not contain white space.
%       duration    Duration of experiment in seconds. Can be 'delta_error'.
%       target_function  Function to drive input currents for target neurons.
%       plotOn      Set true to plot spikes, motor states, and motor synapse weights for each vocalization.
%
%   For updates, see https://github.com/tim-shea/BabbleNN

    % Initialization
    rng shuffle;
    
    learningRate = 0.5;
    minimumSynapticWeight = 0;
    maximumSynapticWeight = 8;
    groupSize = 2500;
    
    % Check simulation id for spaces
    if any(isspace(id))
        disp('Simulation id may not contain spaces.');
        return
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
    NOut = groupSize;
    NExc = NOut;
    NTrg = NOut;
    
    inputs = zeros(NExc, 1);
    targets = zeros(NTrg, 1);
    
    %SExcOut = 200;
    maximumDelay = 1;
    
    % Set time scales of membrane recovery variable
    aExc = 0.02;
    aOut = 0.02;
    aTrg = 0.02;
    % Membrane recovery variable after-spike shift
    dExc = 8;
    dOut = 8;
    dTrg = 8;
    
    % Generate sparse synaptic connections
    %postExcOut = ceil(NOut * rand(NExc, SExcOut));
    %wExcOut = rand(NExc, SExcOut);
    
    % Generate all-to-all synaptic connections
    wExcOut = rand(NExc, NOut);
    
    stdpPre = zeros(NExc, 1001 + maximumDelay);
    
    % Initialize membrane potentials
    vExc = -65 * ones(NExc, 1);
    vOut = -65 * ones(NOut, 1);
    vTrg = -65 * ones(NTrg, 1);
    % Initialize adaptation variable
    uExc = 0.2 .* vExc;
    uOut = 0.2 .* vOut;
    uTrg = 0.2 .* vTrg;
    
    % Neuron spikes for the current second
    spikesExc = [-maximumDelay 0];
    spikesOut = [-maximumDelay 0];
    spikesTrg = [-maximumDelay 0];
    
    T = duration;
    saveInterval = 100;
    
    if plotOn
        hNetworkActivity = figure();
        hSynapseMatrix = figure();
    end
    
    % Run the simulation
    sec = 0;
    for sec = (sec + 1):T
        
        display('********************************************');
        display(['Second ' num2str(sec) ' of ' num2str(T)]);
        
        % Simulate 1000 millisecond timesteps for each second
        for t = 1:1000
            
            % Random thalamic input
            IExc = 12 * (rand(NExc, 1) - 0.5);
            IOut = 12 * (rand(NOut, 1) - 0.5);
            ITrg = 12 * (rand(NTrg, 1) - 0.5);
            
            % Indices of fired neurons
            firedExc = find(vExc >= 30);
            firedOut = find(vOut >= 30);
            firedTrg = find(vTrg >= 30);
            
            % Reset the voltages for those neurons that fired
            vExc(firedExc) = -65;
            vOut(firedOut) = -65;
            vTrg(firedTrg) = -65;
            uExc(firedExc) = uExc(firedExc) + dExc;
            uOut(firedOut) = uOut(firedOut) + dOut;
            uTrg(firedTrg) = uTrg(firedTrg) + dTrg;
            
            % Spike-timing dependent plasticity computations
            stdpPre(firedExc,t + maximumDelay) = 1.0;
            
            % Apply output-based anti-hebbian plasticity
            for spk = 1:length(firedOut)
                n = firedOut(spk);
                wExcOut(:,n) = max(minimumSynapticWeight, wExcOut(:,n) - learningRate * stdpPre(:,t));
            end
            
            % Apply target-based hebbian plasticity
            for spk = 1:length(firedTrg)
                n = firedTrg(spk);
                wExcOut(:,n) = min(maximumSynapticWeight, wExcOut(:,n) + learningRate * stdpPre(:,t));
            end
            
            % Update the record of when neuronal firings occurred
            spikesExc = [spikesExc; t * ones(length(firedExc), 1), firedExc];
            spikesOut = [spikesOut; t * ones(length(firedOut), 1), firedOut];
            spikesTrg = [spikesTrg; t * ones(length(firedTrg), 1), firedTrg];
            
            % Add input and target currents
            if strcmp(target_function, 'plus0.5')
                %inputs = mod((sec * 1000 + t) / 400, 1.0) * ones(NExc, 1);
                if mod(t, 250) == 0
                    inputs = rand() * ones(NExc, 1);
                end
                targets = mod(2 * abs(0.5 - inputs), 1.0);
                
                responses = 4 * max(0, 0.25 - abs((1:NExc)' / NExc - inputs));
                IExc(1:NExc) = IExc(1:NExc) + responses;
                
                responses = 4 * max(0, 0.25 - abs((1:NTrg)' / NTrg - targets));
                ITrg(1:NTrg) = ITrg(1:NTrg) + responses;
            end
            
            % Calculate post synaptic conductances
            k = size(spikesExc, 1);
            while spikesExc(k,1) > t - maximumDelay
                %synInd = postExcOut(spikesExc(k,2),:);
                %IOut(synInd) = IOut(synInd) + wExcOut(spikesExc(k,2))';
                IOut = IOut + wExcOut(spikesExc(k,2),:)';
                k = k - 1;
            end
            
            % Update neuronal membrane potentials
            vExc = vExc + 0.5 * ((0.04 * vExc + 5) .* vExc + 140 - uExc + IExc);
            vExc = vExc + 0.5 * ((0.04 * vExc + 5) .* vExc + 140 - uExc + IExc);
            vOut = vOut + 0.5 * ((0.04 * vOut + 5) .* vOut + 140 - uOut + IOut);
            vOut = vOut + 0.5 * ((0.04 * vOut + 5) .* vOut + 140 - uOut + IOut);
            vTrg = vTrg + 0.5 * ((0.04 * vTrg + 5) .* vTrg + 140 - uTrg + ITrg);
            vTrg = vTrg + 0.5 * ((0.04 * vTrg + 5) .* vTrg + 140 - uTrg + ITrg);
            
            % Update neuronal adaptation variables
            uExc = uExc + aExc .* (0.2 * vExc - uExc);
            uOut = uOut + aOut .* (0.2 * vOut - uOut);
            uTrg = uTrg + aTrg .* (0.2 * vTrg - uTrg);
            
            % Exponential decay of the traces of presynaptic neuron firing
            stdpPre(:,t + maximumDelay + 1) = 0.99 * stdpPre(:,t + maximumDelay);
            
            % Print an update
            if t == 1000
                display(['Input Firing Rate: ' num2str(size(spikesExc, 1) / NExc)]);
                display(['Output Firing Rate: ' num2str(size(spikesOut, 1) / NOut)]);
                display(['Target Firing Rate: ' num2str(size(spikesTrg, 1) / NTrg)]);
            end
        end
        
        if mod(sec, 100) == 0
            learningRate = learningRate / 2;
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
        
        % Plot spikes and synapses
        if plotOn
            set(0, 'currentfigure', hNetworkActivity);
            set(hNetworkActivity, 'name', ['Neural Spiking for Second: ', num2str(sec)], 'numbertitle', 'off');
            plot([spikesExc(:,1); spikesOut(:,1); spikesTrg(:,1)], ...
                [spikesExc(:,2); (spikesOut(:,2) + NExc); (spikesTrg(:,2) + NExc + NOut)], '.');
            title('Spike Raster Plot', 'fontweight', 'bold');
            axis([0 1000 0 NExc + NOut + NTrg]);
            ylabel('Neuron Index');
            
            set(0, 'currentfigure', hSynapseMatrix);
            set(hSynapseMatrix, 'name', ['Synaptic Strengths for Second: ', num2str(sec)], 'numbertitle', 'off');
            imagesc(wExcOut);
            set(gca, 'YDir', 'normal');
            colorbar;
            title('Synapse Strength from Excitatory Neurons', 'fontweight', 'bold');
            xlabel('Post Synaptic Motor Neuron Index');
            ylabel('Synapse Index');
            
            drawnow;
        end
        
        % Prepare for the following 1000 ms
        stdpPre(:,1:maximumDelay + 1) = stdpPre(:,1001:1001 + maximumDelay);
        
        indExc = find(spikesExc(:,1) > 1001 - maximumDelay);
        spikesExc = [-maximumDelay 0; spikesExc(indExc,1) - 1000, spikesExc(indExc,2)];
        indOut = find(spikesOut(:,1) > 1001 - maximumDelay);
        spikesOut = [-maximumDelay 0; spikesOut(indOut,1) - 1000, spikesOut(indOut,2)];
        indTrg = find(spikesTrg(:,1) > 1001 - maximumDelay);
        spikesTrg = [-maximumDelay 0; spikesTrg(indTrg,1) - 1000, spikesTrg(indTrg,2)];
    end
end
