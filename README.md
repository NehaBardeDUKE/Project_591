# TTS Audio Data Generation using MelGAN

### High Level idea: 
Combining prompt engineering with GANs to generate large audio datasets that sound very close to real life audios. Make use of prompt engineering techniques or any other heuristic probability distribution techniques to generate a large dataset of sentences that are later fed to an an expert generator model (vocoder), which is trained to generate realistic audios. 

### Impact:
For training acoustic models, a large amount of data that is either graded carefully by humans or created by humans, is required to accurately train the model to determine the patterns it needs to recognize. With current state of the art workflows, involving humans placed in front of mics and asked to repeat the listed prompts, we suffer both with the time to achieve "x" number of utterances and the cost to record those. Additionally, a user sitting in front of a mic and reciting a script is not a real world scenario. There is an added effort to then noise-augment these clean audios, to more accurately replicate real world embeddings and contect for an acoustic model. With a large dataset, it becomes difficult to see if the noise-augmentation is actually randomized enough so that the acoustic model results are not skewed.
Alternatively grading a small subset of a huge dataset can be more cost effective but again leaves us with a lot of uncertainty wrt the dataset quality.
Another issue with the dataset is the coverage of all usecases for a given problem. Consider we are buidling a virtual agent. If during the training pahse we only team the virtual agent that "call xyz" is a command it should act on, it would maybe try to generalize but again it becomes very uncertain with what exactly are we asking it to generalize on (is it "call" or "xyz").
In production use cases where not all users care to fully enroll with a virtual assistant but still want to use the virtual assistant to a certain extent, we need to be able to deploy a model that can show how generalizability wrt the accents or pronunciations of a given individual.
This project aims at creating POC that takes us 1 step closer to finding the solution to all the above problems by creating a bespoke dataset of any desired size, which can then be used to train the acoustic models.

### Architecture:

Below is the basic flow diagram of the MelGAN architecture along with a few tweaks made to the original architecture, which I wanted to explore.

![image](https://github.com/NehaBardeDUKE/Project_591/assets/110474064/7e481499-291d-4339-a9a5-73b621dba3d6)

More detailed MelGAN architecture:

Generator:

![image](https://github.com/NehaBardeDUKE/Project_591/assets/110474064/7c01097a-1751-407e-94d3-d42e9b8f3186)

Discriminator:

![image](https://github.com/NehaBardeDUKE/Project_591/assets/110474064/e59f51bd-2e8d-4b5c-ac05-8755b0e1d3b8)

### Issues Faced :
Mode Collapse: The GANs are very prone to mode collapse in general but with audio inputs, I felt this was extremely unmanageable at the beginning of the training. This is because for about 500 epochs, I did not hear much of a difference between the generated and the original signal. I tried tweaking the objective functions based on differnt publications (like parallel-WavGAN, HifiGAN and Wavenet architectures) but the problem persisted. The weights werent updating , which meant that the generator would keep on generating close to the same audio. This is where I applied the class reversal method , after every 1000 epochs to ensure that due to the discriminator's "failure" the generator weights would update heavily for 1 epoch and then fall back to small increments.Even with this, due to the huge number of features (and basically using Mel Spectogram- which in itself isnt like our normal image data as it can represent a big change with a small shift in the spectogram diagram) I had to train for at least 2M epochs. In all this training took 27 days on A10. 

### Settings:
As I moved on with the training and testing the model after every 5000 epochs (MOS) I found that I had to rely on multiple papers to determine the exact hyperparaments that would be beneficial , as the the training was going to end up taking longer than expected. Below is the list of hyperparameters and optimizers that the final model was trained with-

![image](https://github.com/NehaBardeDUKE/Project_591/assets/110474064/7c4e8c14-c65c-48c5-9997-e4e592940fdb)

![image](https://github.com/NehaBardeDUKE/Project_591/assets/110474064/06275897-bfcf-467f-8ec3-3cdd0e36891a)

![image](https://github.com/NehaBardeDUKE/Project_591/assets/110474064/20c6aa3a-bb0c-41c4-99d3-58ed7de2e251)

![image](https://github.com/NehaBardeDUKE/Project_591/assets/110474064/bfa1063d-5e49-4e4c-84c6-3c3e997fdb2d)

![image](https://github.com/NehaBardeDUKE/Project_591/assets/110474064/cf33c553-cda4-4825-9a68-3b5574daa2a0)

![image](https://github.com/NehaBardeDUKE/Project_591/assets/110474064/9c65f5f4-d9d2-41ec-a62a-b8a22715a6a0)

![image](https://github.com/NehaBardeDUKE/Project_591/assets/110474064/96d27698-a88d-4e87-a1f4-97e43e72ada8)

### Next Steps:

1. In my original proposal, I had mentioned that i would make use of the genetic algorithm to create an initial dataset. However, this seemed to be an overkill as I reached as low as the 3rd generation. So instead I decided to go with the probability distribution and generated the dataset based on heuristics wrt the Samsung production training data. To transfer this into results I can present outside the org, I have simply put up a smaller version of this, using predefined arrays as this is a POC. So I would like to use the probability distribution heuristics on the ML commons dataset that is publically available, to create sentences to be fed to the vocoder.
2. I would like to train the model using the parallel WavGAN architecture, which albeit is slower and bulkier than MelGAN, but has a better MOS of it being an original audio.
3. Integrate GPT4 API to generate prompts from the probability distribution heuristics, to further completely automate the dataset generation process.

### References:

https://arxiv.org/ftp/arxiv/papers/2005/2005.00065.pdf : Generative Adversarial Networks (GANs): Challenges, Solutions, and Future Directions 


https://arxiv.org/abs/1910.06711 : MelGAN: Generative Adversarial Networks for Conditional Waveform Synthesis


https://arxiv.org/abs/1910.11480 : Parallel WaveGAN: A fast waveform generation model based on generative adversarial networks with multi-resolution spectrogram


https://arxiv.org/abs/2010.05646 : HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis


### Model weights and config:

https://drive.google.com/file/d/1tFetLG4Mv-AblySx7iAa_t-32RThoxS1/view?usp=sharing







