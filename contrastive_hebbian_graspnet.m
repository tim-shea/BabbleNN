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
    minimumSynapticWeight = -8;
    maximumSynapticWeight = 8;
    maximumDelay = 1;
    NOut = 1000;
    NIn = 1000;
    NTrg = 1000;
    
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
    
    inputs = [0 0];
    targets = 0;
    
    % Set time scales of membrane recovery variable
    aIn = 0.02;
    aOut = 0.02;
    aTrg = 0.02;
    % Membrane recovery variable after-spike shift
    dIn = 8;
    dOut = 8;
    dTrg = 8;
    
    % Generate all-to-all synaptic connections
    w = rand(NIn, NOut);
    stdpPre = zeros(NIn, 1001 + maximumDelay);
    
    % Initialize membrane potentials
    vIn = -65 * ones(NIn, 1);
    vOut = -65 * ones(NOut, 1);
    vTrg = -65 * ones(NTrg, 1);
    % Initialize adaptation variable
    uIn = 0.2 .* vIn;
    uOut = 0.2 .* vOut;
    uTrg = 0.2 .* vTrg;
    
    % Neuron spikes for the current second
    spikesIn = [-maximumDelay 0];
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
            IIn = 10 * (rand(NIn, 1) - 0.5);
            IOut = 10 * (rand(NOut, 1) - 0.5);
            ITrg = 10 * (rand(NTrg, 1) - 0.5);
            
            % Indices of fired neurons
            firedIn = find(vIn >= 30);
            firedOut = find(vOut >= 30);
            firedTrg = find(vTrg >= 30);
            
            % Reset the voltages for those neurons that fired
            vIn(firedIn) = -65;
            vOut(firedOut) = -65;
            vTrg(firedTrg) = -65;
            uIn(firedIn) = uIn(firedIn) + dIn;
            uOut(firedOut) = uOut(firedOut) + dOut;
            uTrg(firedTrg) = uTrg(firedTrg) + dTrg;
            
            % Spike-timing dependent plasticity computations
            stdpPre(firedIn,t + maximumDelay) = 1.0;
            
            % Apply output-based anti-hebbian plasticity
            for spk = 1:length(firedOut)
                n = firedOut(spk);
                w(:,n) = max(minimumSynapticWeight, w(:,n) - learningRate * stdpPre(:,t));
            end
            
            % Apply target-based hebbian plasticity
            for spk = 1:length(firedTrg)
                n = firedTrg(spk);
                w(:,n) = min(maximumSynapticWeight, w(:,n) + learningRate * stdpPre(:,t));
            end
            
            % Update the record of when neuronal firings occurred
            spikesIn = [spikesIn; t * ones(length(firedIn), 1), firedIn];
            spikesOut = [spikesOut; t * ones(length(firedOut), 1), firedOut];
            spikesTrg = [spikesTrg; t * ones(length(firedTrg), 1), firedTrg];
            
            % Add input and target currents
            if strcmp(target_function, 'xor')
                %inputs = mod((sec * 1000 + t) / 400, 1.0) * ones(NExc, 1);
                if mod(t, 250) == 0
                    inputs = [round(rand()), round(rand())];
                end
                targets = [xor(inputs(1), inputs(2)), not(xor(inputs(1), inputs(2)))];
                
                inputResponses = [inputs(1) * ones(NIn / 2, 1); inputs(2) * ones(NIn / 2, 1)];
                targetResponses = [targets(1) * ones(NTrg / 2, 1); targets(2) * ones(NTrg / 2, 1)];
                
                %function currents = continuous_response(value, rows, sharpness)
                %    responses = sharpness * max(0, (1 / sharpness) - abs((1:rows)' / rows - value));
                %end
                
                IIn(1:NIn) = (10 + 5 * inputResponses) .* (rand(NIn, 1) - 0.5);
                ITrg(1:NTrg) = (10 + 5 * targetResponses) .* (rand(NTrg, 1) - 0.5);
            end
            
            % Calculate post synaptic conductances
            k = size(spikesIn, 1);
            while spikesIn(k,1) > t - maximumDelay
                IOut = IOut + w(spikesIn(k,2),:)';
                k = k - 1;
            end
            
            % Update neuronal membrane potentials
            vIn = vIn + 0.5 * ((0.04 * vIn + 5) .* vIn + 140 - uIn + IIn);
            vIn = vIn + 0.5 * ((0.04 * vIn + 5) .* vIn + 140 - uIn + IIn);
            vOut = vOut + 0.5 * ((0.04 * vOut + 5) .* vOut + 140 - uOut + IOut);
            vOut = vOut + 0.5 * ((0.04 * vOut + 5) .* vOut + 140 - uOut + IOut);
            vTrg = vTrg + 0.5 * ((0.04 * vTrg + 5) .* vTrg + 140 - uTrg + ITrg);
            vTrg = vTrg + 0.5 * ((0.04 * vTrg + 5) .* vTrg + 140 - uTrg + ITrg);
            
            % Update neuronal adaptation variables
            uIn = uIn + aIn .* (0.2 * vIn - uIn);
            uOut = uOut + aOut .* (0.2 * vOut - uOut);
            uTrg = uTrg + aTrg .* (0.2 * vTrg - uTrg);
            
            % Exponential decay of the traces of presynaptic neuron firing
            stdpPre(:,t + maximumDelay + 1) = 0.99 * stdpPre(:,t + maximumDelay);
            
            % Print an update
            if t == 1000
                display(['Input Firing Rate: ' num2str(size(spikesIn, 1) / NIn)]);
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
            for spike = 1:size(spikesIn, 1)
                fprintf(spikes_fid, '%i\t', sec);
                fprintf(spikes_fid, '%i\t%i', spikesIn(spike,:));
                fprintf(spikes_fid, '\n');
            end
            fclose(spikes_fid);
            save(workspace)
        end
        
        % Plot spikes and synapses
        if plotOn
            set(0, 'currentfigure', hNetworkActivity);
            set(hNetworkActivity, 'name', ['Neural Spiking for Second: ', num2str(sec)], 'numbertitle', 'off');
            plot([spikesIn(:,1); spikesOut(:,1); spikesTrg(:,1)], ...
                [spikesIn(:,2); (spikesOut(:,2) + NIn); (spikesTrg(:,2) + NIn + NOut)], '.');
            title('Spike Raster Plot', 'fontweight', 'bold');
            axis([0 1000 0 NIn + NOut + NTrg]);
            ylabel('Neuron Index');
            
            set(0, 'currentfigure', hSynapseMatrix);
            set(hSynapseMatrix, 'name', ['Synaptic Strengths for Second: ', num2str(sec)], 'numbertitle', 'off');
            imagesc(w);
            set(gca, 'YDir', 'normal');
            colorbar;
            title('Synapse Weights', 'fontweight', 'bold');
            xlabel('Neuron Index');
            ylabel('Neuron Index');
            
            drawnow;
        end
        
        % Prepare for the following 1000 ms
        stdpPre(:,1:maximumDelay + 1) = stdpPre(:,1001:1001 + maximumDelay);
        
        indIn = find(spikesIn(:,1) > 1001 - maximumDelay);
        spikesIn = [-maximumDelay 0; spikesIn(indIn,1) - 1000, spikesIn(indIn,2)];
        indOut = find(spikesOut(:,1) > 1001 - maximumDelay);
        spikesOut = [-maximumDelay 0; spikesOut(indOut,1) - 1000, spikesOut(indOut,2)];
        indTrg = find(spikesTrg(:,1) > 1001 - maximumDelay);
        spikesTrg = [-maximumDelay 0; spikesTrg(indTrg,1) - 1000, spikesTrg(indTrg,2)];
    end
end
