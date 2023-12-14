# TTS Audio Data Generation using MelGAN

##### High Level idea: 
Combining prompt engineering with GANs to generate large audio datasets that sound very close to real life audios. Make use of prompt engineering techniques or any other heuristic probability distribution techniques to generate a large dataset of sentences that are later fed to an an expert generator model (vocoder), which is trained to generate realistic audios. 

##### Impact:
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
1. The GANs are very 




