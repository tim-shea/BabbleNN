BabbleNN
========

This repository is a branch of Anne Warlaumont's BabbleNN model, which is a computational model of human articulatory and auditory systems used to investigate infant speech development. This branch extends the model to explore multiple degrees of freedom by generalizing the motor control code.

The code is written in MATLAB. It requires Praat (http://www.fon.hum.uva.nl/praat/) to be installed and was developed for Mac and Windows. Call the babble_daspnet_multi(...) function to run the model.

Results produced with this model were submitted to COSYNE 2016. The full data for the abstract are available upon request (omitted due to size: 4 groups x 6 runs x 3600 vocalizations = 24 hours of audio, approx. 5 GB total). Representative vocalizations are available in the Cosyne16 subdirectory of the repository.

The model is derived from the following work: 

A. S. Warlaumont, “Salience-based reinforcement of a spiking neural network leads to increased syllable production,” in IEEE International Conference on Development and Learning and Epigenetic Robotics (ICDL), 2013.

E. M. Izhikevich, “Solving the distal reward problem through linkage of STDP and dopamine signaling,” Cerebral Cortex, vol. 17, no. 10, pp. 2443–2452, 2007. Code available at http://izhikevich.org/publications/dastdp.htm

M. Coath, S. L. Denham, L. Smith, H. Honing, A. Hazan, P. Holonwicz, and H. Purwins, “An auditory model for the detection of perceptual onsets and beat tracking in singing,” in Neural Information Processing Systems, Workshop on Music Processing in the Brain, 2007.
