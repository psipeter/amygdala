# A Scalable Spiking Amygdala Model that Explains Fear Conditioning, Extinction, Renewal, and Generalization

The amygdala (AMY) is widely implicated in fear learning and behavior, but it remains unclear how the many biological components present within AMY interact to achieve these abilities. Building on previous work, we hypothesize that individual AMY nuclei represent different quantities, and that fear conditioning arises from error-driven learning on the synapses between AMY nuclei. We present a computational model of AMY that (a) recreates the divisions and connections between AMY nuclei and their constituent pyramidal and inhibitory neurons; (b) accommodates scalable high-dimensional representations of external stimuli; (c) learns to associate complex stimuli with the presence (or absence) of an aversive stimulus; (d) preserves feature information when mapping inputs to salience estimates, such that these estimates generalize to similar stimuli; and (e) induces a diverse profile of neural responses within each nucleus. Our model predicts (1) defensive responses and neural activities in several experimental conditions, (2) the consequence of externally inactivating particular nuclei, and (3) the tendency to generalize defensive responses to novel stimuli. We test these predictions by comparing model outputs to neural and behavioral data from animals and humans. Despite the relative simplicity of our model, we find significant overlap between simulated and empirical data, which supports our claim that the model captures many of the neural mechanisms that support fear conditioning. We conclude by comparing our model to other computational models, and by characterizing the theoretical relationship between pattern separation and fear generalization in healthy versus anxious individuals.

For questions or comments, please submit an issue, or email psipeter@gmail.com


Clone the repository
=======
  
  git clone https://github.com/psipeter/amygdala.git

  cd amygdala

    
Install dependencies
=======
    
  pip install numpy scipy matplotlib seaborn pandas nengo jupyter


Run the jupyter notebook
=======
  
  jupyter notebook experiments.py