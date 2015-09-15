function [] = babble_daspnet_multi(id, newT, reinforcer, yoke, plotOn)
%   BABBLE_DASPNET_MULTI
%   Neural network model of the development of multi-dimensional vocal learning.
%   Estimates of auditory salience calculated from a modified version of
%   Coath et. al. (2009) auditory salience algorithms.
%   
%   Description of Input Arguments:
%       id          Unique identifier for this simulation. Must not contain white space.
%       newT        Time experiment is to run in seconds. Can specify new times (longer or shorter) 
%                   for experimental runs by changing this value when a simulation is restarted.
%       reinforcer  Type of reinforcement. Can be 'human', 'relhisal', or 'range'.
%       yoke        Indicates whether to run an experiment or yoked control simulation. Set false to run a regular simulation. 
%                   Set true to run a yoked control. There must be a saved workspace of the same name on the MATLAB path
%                   for the simulation to yoke to.
%       plotOn      Set true to plot spikes, motor states, and motor synapse weights for each vocalization.
%
%   Example of Use:
%   babble_daspnet_multi('Mortimer', 7200, 'relhisal', false, false);
%
%   For updates, see https://github.com/tim-shea/BabbleNN

% Initialization
rng shuffle;

salthresh = 15;             % Initial salience value for reward (used in 'relhisal' reinforcment)
DAinc = 0.1;                  % Amount of dopamine given during reward
sm = 4;                     % Maximum synaptic weight
testint = 1;                % Number of seconds between vocalizations

% Directory names for data
wavdir = [id, '_Wave'];
firingsdir = [id, '_Firings'];
workspacedir = [id, '_Workspace'];
yokeworkspacedir = [id, '_YokedWorkspace'];

% Error Checking.
if(any(isspace(id)))
    disp('Please choose an id without spaces.');
    return
end

% Creating data directories.
if ~exist(wavdir, 'dir')
    mkdir(wavdir);
else
    addpath(wavdir);
end

if ~exist(firingsdir, 'dir')
    mkdir(firingsdir);
else
    addpath(firingsdir);
end

if ~exist(workspacedir, 'dir')
    mkdir(workspacedir);
else
    addpath(workspacedir);
end

if yoke
    if ~exist(yokeworkspacedir, 'dir')
        mkdir(yokeworkspacedir);
    else
        addpath(yokeworkspacedir);
    end
    % Where to put the yoked control simulation data.
    workspaceFilename=[yokeworkspacedir,'/babble_daspnet_multi_',id,'_yoke.mat'];
    % Where to find the original simulation data.
    yokeSourceFilename=[workspacedir,'/babble_daspnet_multi_',id,'.mat'];
else
    workspaceFilename=[workspacedir,'/babble_daspnet_multi_',id,'.mat'];
end

% Directory for Coath et. al. Saliency Detector.
%addpath('auditorysaliencymodel'); 

if exist(workspaceFilename) > 0 && not(yoke)
    load(workspaceFilename);
else
    
    M=100;                 % number of synapses per neuron
    D=1;                   % maximal conduction delay
    
    numberOfMuscles = 7;
    numberOfGroups = 2 * numberOfMuscles;
    groupSize = 100;
    Nout = groupSize * numberOfGroups;
    Nmot = Nout;
    
    % excitatory neurons   % inhibitory neurons      % total number
    Ne=Nout;               Ni=Ne / 4;                N=Ne+Ni;
    
    groupSpikeCounts = zeros(numberOfGroups);
    muscleDelta = zeros(numberOfMuscles, 1);
    muscleState = zeros(numberOfMuscles, 1000, newT);
    
    a=[0.02*ones(Ne,1);    0.1*ones(Ni,1)];     % Sets time scales of membrane recovery variable.
    d=[   8*ones(Ne,1);    2*ones(Ni,1)];       % Membrane recovery variable after-spike shift. 
    a_mot=.02*ones(Nmot,1);
    d_mot=8*ones(Nmot,1);
    post=ceil([N*rand(Ne,M);Ne*rand(Ni,M)]); % Assign the postsynaptic neurons for each neuron's synapse in the reservoir.
    
    % Connect output groups to their respective motor groups (e.g. 1:100 to 1:100)
    post_mot = [];
    for g = 1:numberOfGroups
        post_mot = cat(1, post_mot, repmat((g - 1) * groupSize + 1:g * groupSize, groupSize, 1));
    end
    
    s=[rand(Ne,M);-rand(Ni,M)]; % Synaptic weights in the reservoir.
    sout=rand(Nout,Nmot); % Synaptic weights from the reservoir output neurons to the motor neurons.
    
    % Normalizing the synaptic weights.
    sout=sout./(mean(mean(sout)));
    
    sd=zeros(Nout,Nmot); % The change to be made to sout.
    
    for i=1:N
        delays{i,1}=1:M;
    end
    
    for i = 1:Nout
        delays_mot{i,1} = 1:groupSize;
    end
    
    STDP = zeros(Nout,1001+D);
    v = -65*ones(N,1);          % Membrane potentials.
    v_mot = -65*ones(Nmot,1);
    u = 0.2.*v;                 % Membrane recovery variable.
    u_mot = 0.2.*v_mot;
    firings=[-D 0];     % All reservoir neuron firings for the current second.
    outFirings=[-D 0];  % Output neuron spike timings.
    motFirings=[-D 0];  % Motor neuron spike timings.
    
    DA=0; % Level of dopamine above the baseline.
    
    muscsmooth=100; % Spike train data sent to Praat is smoothed by doing a 100 ms moving average.
    sec=0;
    
    rewcount=0; 
    rew=[];
    
    % Initializing reward policy variables.
    if strcmp(reinforcer,'relhisal')
        temprewhist=zeros(1,10); % Keeps track of rewards given at a threshold value for up to 10 previous sounds.
    end
    
    if strcmp(reinforcer, 'range')
        temprewhist=zeros(1,10);
        range_hist = NaN(newT, 1); % Keeps a record of the range over the entire simulation run time.
        rangethresh = 0.75; % Starting threshold for reward. 
        Ranginc = .05; % How much to increase the threshold by.
    end
    
    % Variables for saving data.
    %vlstsec = 0; % Record of when v_mot_hist was last saved.
    switch reinforcer  % Sets how often to save data.
        case 'human'
            SAVINTV = 10;
        otherwise
            SAVINTV = 100;
    end
    
end

T=newT;
clearvars newT;

% Special initializations for a yoked control.
if yoke
    load(yokeSourceFilename,'rew', 'yokedruntime');
    if (T > yokedruntime)
        T = yokedruntime;
    end
end

%RUNNING THE SIMULATION%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for sec = (sec + 1):T % T is the duration of the simulation in seconds.
    
    display('********************************************');
    display(['Second ',num2str(sec),' of ',num2str(T)]);
    
    % How long a yoked control could be run. Assumes rewards are
    % assigned for the current second only.
    yokedruntime = sec;
    
    for t=1:1000                            % Millisecond timesteps
        
        %Random Thalamic Input.
        I=13*(rand(N,1)-0.5);
        I_mot=13*(rand(Nmot,1)-0.5);
            
        fired = find(v>=30);                % Indices of fired neurons
        fired_out = find(v(1:numberOfGroups * groupSize)>=30);
        fired_mot = find(v_mot>=30);
        v(fired)=-65;                       % Reset the voltages for those neurons that fired
        v_mot(fired_mot)=-65;
        u(fired)=u(fired)+d(fired);         % Individual neuronal dynamics
        u_mot(fired_mot)=u_mot(fired_mot)+d_mot(fired_mot);
        
        % Spike-timing dependent plasticity computations:
        STDP(fired_out,t+D)=0.1; % Keep a record of when the output neurons spiked.
        for k=1:length(fired_mot)
            sd(:,fired_mot(k))=sd(:,fired_mot(k))+STDP(:,t); % Adjusting sd for synapses eligible for potentiation.
        end
        firings=[firings;t*ones(length(fired),1),fired];                % Update the record of when neuronal firings occurred.
        outFirings=[outFirings;t*ones(length(fired_out),1),fired_out];
        motFirings=[motFirings;t*ones(length(fired_mot),1),fired_mot];
        % For any presynaptic neuron that just fired, calculate the current to add
        % as proportional to the synaptic strengths from its postsynaptic neurons.
        I(1:Ne)=I(1:Ne)+1*muscleState(ceil((1:Ne) / (2 * groupSize)), t, sec);
        k=size(firings,1);
        while firings(k,1)>t-D
            del=delays{firings(k,2),t-firings(k,1)+1};
            ind = post(firings(k,2),del);
            I(ind)=I(ind)+s(firings(k,2), del)';
            k=k-1;
        end;
        % Calculating currents to add for motor neurons. 
        k=size(outFirings,1);
        while outFirings(k,1)>t-D
            del_mot=delays_mot{outFirings(k,2),t-outFirings(k,1)+1};
            ind_mot = post_mot(outFirings(k,2),del_mot);
            I_mot(ind_mot)=I_mot(ind_mot)+2*sout(outFirings(k,2), del_mot)';
            k=k-1;
        end;
        
        % Individual neuronal dynamics computations:
        v=v+0.5*((0.04*v+5).*v+140-u+I);                            % for numerical
        v=v+0.5*((0.04*v+5).*v+140-u+I);                            % stability time
        v_mot=v_mot+0.5*((0.04*v_mot+5).*v_mot+140-u_mot+I_mot);    % step is 0.5 ms
        v_mot=v_mot+0.5*((0.04*v_mot+5).*v_mot+140-u_mot+I_mot);
        %v_mot_hist{sec}=[v_mot_hist{sec},v_mot];
        u=u+a.*(0.2*v-u);                   
        u_mot=u_mot+a_mot.*(0.2*v_mot-u_mot);

		% Exponential decay of the traces of presynaptic neuron firing
        STDP(:,t+D+1)=0.95*STDP(:,t+D);                             % tau = 20 ms
        
		% Exponential decay of the dopamine concentration over time.
        DA=DA*0.995; 
        
        % Modify synaptic weights.
        if (mod(t,10)==0)
            sout=max(0,min(sm,sout+DA*sd));

            % Normalizing the synaptic weights.
            sout=sout./(mean(mean(sout)));
            
            sd=0.99*sd; % Decrease in synapse's eligibility to change over time.
        end;
        
        % Every testint seconds, use the motor neuron spikes to generate a sound.
        if (mod(sec,testint) == 0)
            
            if t < 1000
                for g = 1:numberOfGroups
                    groupIndices = ((g - 1) * groupSize + 1):(g * groupSize);
                    groupSpikeCounts(g) = sum(v_mot(groupIndices) > 30);
                end
                for m = 1:numberOfMuscles
                    spikeDelta = groupSpikeCounts(2 * m - 1) - groupSpikeCounts(2 * m);
                    muscleDelta(m) = 0.99 * muscleDelta(m) + 0.005 * spikeDelta;
                    muscleState(m,t+1,sec) = min(max(muscleState(m,t,sec) + muscleDelta(m), -1), 1);
                end
            end
            
            if t == 1000 % Based on the 1 s timeseries of smoothed summed motor neuron spikes, generate a sound.
                
                if ~strcmp(reinforcer,'range')
                    % Synthesize a vocalization based on the previous second of activity
                    [name, fid] = createVocalization(id, yoke, sec, wavdir);
                    setVocalTarget(fid, 0.0, 0.1 + 0.05 * mean(muscleState(1,:,sec)), 'Lungs');
                    setVocalTarget(fid, 0.1, 0.0, 'Lungs');
                    setVocalTarget(fid, 1.0, 0.0, 'Lungs');
                    for tPratt = 1:1000
                        setVocalTarget(fid, tPratt / 1000, 0.5 + 0.05 * muscleState(2,tPratt,sec), 'Interarytenoid');
                        setVocalTarget(fid, tPratt / 1000, 0.4 + 0.2 * muscleState(3,tPratt,sec), 'Hyoglossus');
                        setVocalTarget(fid, tPratt / 1000, 0.5 * muscleState(4,tPratt,sec), 'Masseter');
                        setVocalTarget(fid, tPratt / 1000, 0.5 * muscleState(5,tPratt,sec), 'UpperTongue');
                        setVocalTarget(fid, tPratt / 1000, 0.25 + 0.25 * muscleState(6,tPratt,sec), 'OrbicularisOris');
                        setVocalTarget(fid, tPratt / 1000, 0.6 + 0.2 * muscleState(7,tPratt,sec), 'LevatorPalatini');
                    end
                    executeVocalization(name, fid, wavdir, true);
                    
                    % Find the auditory salience of the sound:
                    if not(yoke)
                        salienceResults = auditorySalience([wavdir '/sound_' id '_' num2str(sec) '.wav'], 0);
                    else
                        display([wavdir '/sound_' id '_yoke_' num2str(sec) '.wav']);
                        salienceResults = auditorySalience([wavdir '/sound_' id '_yoke_' num2str(sec) '.wav'], 0);
                    end
                    salience = sum(abs(salienceResults.saliency(31:180))); % Summing over salience trace to produce a single value.
                    salhist(sec,1) = salience; % History of salience over entire simulation.
          
                    if ~(strcmp(reinforcer,'human') && not(yoke))
                        display(['salience = ',num2str(salience)]);
                    end
                end
                
                % Assign Reward.
                if yoke
                    % Yoked controls use reward assigned from the
                    % experiment they are yoked to.
                    if any(rew==sec*1000+t)
                        display('rewarded');
                    else
                        display('not rewarded');
                    end
                    % Play the sound automatically for yoked simulations
                    if verLessThan('matlab', '8.0.0')
                        [mysound,Fs] = wavread([wavdir,'/sound_',id,'_yoke_',num2str(sec),'.wav']);
                    else
                        [mysound,Fs] = audioread([wavdir,'/sound_',id,'_yoke_',num2str(sec),'.wav']);
                    end
                    sound(mysound,Fs);
                elseif strcmp(reinforcer,'human')
                    % Asking the human listener to provide or withold reinforcement for this sound.
                    tempInput = input('Press Return/Enter to play the sound.','s');
                    % Read and play the sound file. (maintains backwards compatibility with wavread)
                    if verLessThan('matlab', '8.0.0')
                        [mysound,Fs] = wavread([wavdir,'/sound_',id,'_',num2str(sec),'.wav']);
                    else
                        [mysound,Fs] = audioread([wavdir,'/sound_',id,'_',num2str(sec),'.wav']);
                    end
                    sound(mysound,Fs);
                    % Get listener's reinforcment decision.
                    user_reinforceAmount = '2';
                    while ~max(strcmp(user_reinforceAmount,{'0','1'}))
                        user_reinforceAmount = input('Enter 1 to reinforce this sound. Enter zero to withhold reinforcement.','s');
                    end
                    if strcmp(user_reinforceAmount,'1')
                        rew = [rew,sec*1000+t];
                    end
                elseif strcmp(reinforcer,'relhisal')
                    display(['salthresh: ',num2str(salthresh)]);
                    temprewhist(1:9) = temprewhist(2:10);
                    % Reward if the salience of the sound is above
                    % threshold value.
                    if salience > salthresh
                        display('rewarded');
                        rew = [rew, sec * 1000 + t];
                        rewcount = rewcount + 1;
                        temprewhist(10) = 1;
                        % If at least 3 of the last 10 sounds were above
                        % threshold, raise the threshold value and reset the count.
                        if mean(temprewhist) >= .5
                            salthresh = salthresh + 0.5;
                            temprewhist = zeros(1,10);
                        end
                    else
                        display('not rewarded');
                        temprewhist(10) = 0;
                    end
                    display(['temprewhist: ' num2str(temprewhist)]);
                    display(['mean(temprewhist): ' num2str(mean(temprewhist))]);
                elseif strcmp(reinforcer, 'range')
                    muscrange = range(smoothmusc(:,sec)); % Range of motor neuron activation during this second.
                    range_hist(sec) = muscrange;
					display(['Range Threshold: ', num2str(rangethresh)]);
                    display(['Current Range: ', num2str(muscrange)]);
                    temprewhist(1:9)=temprewhist(2:10);
                    % Reward if the range of the muscle activation is above
                    % threshold value.
                    if muscrange>rangethresh
                        display('rewarded');
                        rew=[rew,sec*1000+t];
                        rewcount=rewcount+1;
                        temprewhist(10)=1;
                        % If at least 3 of the last 10 sounds were above
                        % threshold, raise the threshold value and reset the count.
                        if mean(temprewhist)>=.3
                            rangethresh = rangethresh + Ranginc;
                            temprewhist=zeros(1,10);
                        end
                    else
                        display('not rewarded');
                        temprewhist(10)=0;
                    end
                    display(['temprewhist: ',num2str(temprewhist)]);
                    display(['mean(temprewhist): ',num2str(mean(temprewhist))]);
                end
                
                % Display reward count information.
                if not(yoke) && ~strcmp(reinforcer,'human')
                    display(['rewcount: ',num2str(rewcount)]);
                end
                
            end
        end
        
        % If the human listener decided to reinforce (or if the yoked control
        % schedule says to reinforce), increase the dopamine concentration.
        if any(rew == sec * 1000 + t)
            DA = DA + DAinc;
        end
    end
    
    % Writing reservoir neuron firings for this second to a text file.
    if mod(sec,SAVINTV*testint)==0 || sec==T        
        if not(yoke)
            firings_fid = fopen([firingsdir,'/babble_daspnet_firings_',id,'_',num2str(sec),'.txt'],'w');
        else
            firings_fid = fopen([firingsdir,'/babble_daspnet_firings_',id,'_yoke_',num2str(sec),'.txt'],'w');
        end
        for firingsrow = 1:size(firings,1)
            fprintf(firings_fid,'%i\t',sec);
            fprintf(firings_fid,'%i\t%i',firings(firingsrow,:));
            fprintf(firings_fid,'\n');
        end
        fclose(firings_fid);
    end
    
    % Plot reservoir, output, and motor spikes, muscle states, and motor synapses
    if plotOn
        hNeural = figure(103);
        set(hNeural, 'name', ['Neural Spiking for Second: ', num2str(sec)], 'numbertitle','off');
        subplot(4,1,1)
        plot(firings(:,1),firings(:,2),'.'); % Plot all the neurons' spikes
        title('Reservoir Firings', 'fontweight','bold');
        axis([0 1000 0 N]);
        subplot(4,1,2)
        plot(outFirings(:,1),outFirings(:,2),'.'); % Plot the output neurons' spikes
        title('Output Neuron Firings', 'fontweight','bold');
        axis([0 1000 0 Nout]);
        subplot(4,1,3)
        plot(motFirings(:,1),motFirings(:,2),'.'); % Plot the motor neurons' spikes
        title('Motor Neuron Firings', 'fontweight','bold');
        axis([0 1000 0 Nmot]);
        subplot(4,1,4);
        plot((1:1000), muscleState(:,:,sec));
        title('Motor Group Activity', 'fontweight','bold');
        
        drawnow;
        
        hSyn = figure(113);
        set(hSyn, 'name', ['Synaptic Strengths for Second: ', num2str(sec)], 'numbertitle','off');
        imagesc(sout)
        set(gca,'YDir','normal')
        colorbar;
        title('Synapse Strength between Output Neurons and Motor Neurons', 'fontweight','bold');
        xlabel('postsynaptic motor neuron index', 'fontweight','bold');
        ylabel('presynaptic output neuron index', 'fontweight','bold');
        
    end
    
    % Prepare STDP and firings for the following 1000 ms.
    STDP(:,1:D+1) = STDP(:,1001:1001 + D);
    ind = find(firings(:,1) > 1001 - D);
    firings = [-D 0; firings(ind,1) - 1000, firings(ind,2)];
    ind_out = find(outFirings(:,1) > 1001 - D);
    outFirings = [-D 0; outFirings(ind_out,1) - 1000, outFirings(ind_out,2)];
    ind_mot = find(motFirings(:,1) > 1001 - D);
    motFirings = [-D 0; motFirings(ind_mot,1)-1000, motFirings(ind_mot,2)];
    
    % Every so often, save the workspace in case the simulation is interupted all data is not lost.
    if mod(sec, SAVINTV * testint) == 0 || sec == T
        display('Data Saving..... Do not exit program.');
        save(workspaceFilename)
    end
end
end

%   Write a praat script for synthesizing a vocalization
%   id      Simulation identifier
%   yoke    True if the simulation is yoked
%   sec     The simulation time in seconds (to distinguish vocalizations)
function [name, fid] = createVocalization(id, yoke, sec, wavdir)
    name = [id iff(yoke, '_yoke_', '_') num2str(sec)];
    fid = fopen([wavdir '/script_' name '.praat'], 'w');
    fprintf(fid, 'Create Speaker... speaker Female 2\n');
    fprintf(fid, ['Create Artword... babble 1.0\n']);
    fprintf(fid, 'select Artword babble\n');
end

%   Writes a set target line to a praat scripts for articulatory synthesis.
%   voc         Vocalization identifier returned by a call to createVocalization
%   time        Time for the articulatory synthesizer to reach the target activation level
%   activation  Level of activity for the articulator
%   articulator Text name of the articulator to target (e.g. UpperTongue)
function [] = setVocalTarget(fid, time, activation, articulator)
    fprintf(fid, ['Set target... ' num2str(time, '%.3f') ' ' num2str(activation, '%.3f') ' ' articulator '\n']);
end

%   Finalize and run a praat vocalization script
%   voc     Vocalization identifier returned by a call to createVocalization
%   remove  Whether to delete the vocalization script after running it
function [] = executeVocalization(name, fid, wavdir, removeScript)
    fprintf(fid, 'select Speaker speaker\n');
    fprintf(fid, 'plus Artword babble\n');
    fprintf(fid, 'To Sound... 22050 25 0 0 0 0 0 0 0 0 0\n');
    fprintf(fid, '\tselect Sound babble_speaker\n');
    fprintf(fid, ['\tWrite to WAV file... sound_' name '.wav\n']);
    fclose(fid);
    
    % Execute the Praat script
    if ismac
        praatPath = '/Applications/Praat.app/Contents/MacOS/Praat';
    elseif isunix
        praatPath = 'praat';
    elseif ispc
        praatPath = 'c:/users/lab/praatcon';
    end
    system([praatPath ' ' wavdir '/script_' name '.praat']);
    
    % Remove the script
    if removeScript
        delete([wavdir '/script_' name '.praat']);
    end
end

%   Return value r1 if cond is true or r2 otherwise
function r = iff(cond, r1, r2)
    if cond
        r = r1;
    else
        r = r2;
    end
end