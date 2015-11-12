
load('Cosyne16_Sims/TwoDoF_1.mat');

hNetworkActivity = figure();
set(hNetworkActivity, 'name', ['Example Network Activity'], 'numbertitle','off');

subplot(2, 2, 1);
excInd = firings(:,2) <= 800;
inhInd = firings(:,2) > 800;
mot1Ind = motFirings(:,2) <= 400;
mot2Ind = motFirings(:,2) > 400;
plot(firings(excInd,1), firings(excInd,2), '.r', firings(inhInd,1), firings(inhInd,2), '.b', ...
     motFirings(mot1Ind,1), motFirings(mot1Ind,2) + 1000, '.m', ...
     motFirings(mot2Ind,1), motFirings(mot2Ind,2) + 1000, '.c');
aTitle = title('A', 'fontweight','bold');
axis([0 1000 0 1800]);

subplot(2, 2, 3);
plot(permute(muscleState(:,1:1000,sec), [2 1 3]));
title('B', 'fontweight','bold');
axis([0 1000 -1 1]);

subplot(2, 2, 2);
imagesc(salienceResults.cortResp);
title('C', 'fontweight','bold');

subplot(2, 2, 4);
plot(1:5:1000, salienceResults.saliency);
xlim([0 1000]);
