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
Mode Collapse: The GANs are very prone to mode collapse in general but with audio inputs, I felt this was extremely unmanageable at the beginning of the training. This is because for about 500 epochs, I did not hear much of a difference between the generated and the original signal. I tried tweaking the objective functions based on differnt publications (like parallel-WavGAN, HifiGAN and Wavenet architectures) but the problem persisted. The weights werent updating , which meant that the generator would keep on generating close to the same audio. This is where I applied the class reversal method , after every 50 epochs to ensure that due to the discriminator's "failure" the generator weights would update heavily for 1 epoch and then fall back to small increments.Even with this, due to the huge number of features (and basically using Mel Spectogram- which in itself isnt like our normal image data as it can represent a big change with a small shift in the spectogram diagram) I had to train for at least 400K epochs. In all this training took 17 days on A10s (courtesy of Samsung). 

### Settings:
As I moved on with the training and testing the model after every 500 epochs (MOS) I found that I had to rely on multiple papers to determine the exact hyperparaments that would be beneficial , as the the training was going to end up taking longer than expected. Below is the list of hyperparameters and optimizers that the final model was trained with-

###########################################################
#                FEATURE EXTRACTION SETTING               #
###########################################################
sampling_rate: 22050     # Sampling rate.
fft_size: 1024           # FFT size.
hop_size: 256            # Hop size.
win_length: null         # Window length.
                         # If set to null, it will be the same as fft_size.
window: "hann"           # Window function.
num_mels: 80             # Number of mel basis.
fmin: 80                 # Minimum freq in mel basis calculation.
fmax: 7600               # Maximum frequency in mel basis calculation.
global_gain_scale: 1.0   # Will be multiplied to all of waveform.
trim_silence: true       # Whether to trim the start and end of silence.
trim_threshold_in_db: 60 # Need to tune carefully if the recording is not good.
trim_frame_size: 2048    # Frame size in trimming.
trim_hop_size: 512       # Hop size in trimming.
format: "hdf5"           # Feature file format. "npy" or "hdf5" is supported.

###########################################################
#         GENERATOR NETWORK ARCHITECTURE SETTING          #
###########################################################
generator_type: "MelGANGenerator" # Generator type.
generator_params:
    in_channels: 80               # Number of input channels.
    out_channels: 1               # Number of output channels.
    kernel_size: 7                # Kernel size of initial and final conv layers.
    channels: 512                 # Initial number of channels for conv layers.
    upsample_scales: [8, 8, 2, 2] # List of Upsampling scales.
    stack_kernel_size: 3          # Kernel size of dilated conv layers in residual stack.
    stacks: 3                     # Number of stacks in a single residual stack module.
    use_weight_norm: True         # Whether to use weight normalization.
    use_causal_conv: False        # Whether to use causal convolution.

###########################################################
#       DISCRIMINATOR NETWORK ARCHITECTURE SETTING        #
###########################################################
discriminator_type: "MelGANMultiScaleDiscriminator" # Discriminator type.
discriminator_params:
    in_channels: 1                    # Number of input channels.
    out_channels: 1                   # Number of output channels.
    scales: 3                         # Number of multi-scales.
    downsample_pooling: "AvgPool1d"   # Pooling type for the input downsampling.
    downsample_pooling_params:        # Parameters of the above pooling function.
        kernel_size: 4
        stride: 2
        padding: 1
        count_include_pad: False
    kernel_sizes: [5, 3]              # List of kernel size.
    channels: 16                      # Number of channels of the initial conv layer.
    max_downsample_channels: 1024     # Maximum number of channels of downsampling layers.
    downsample_scales: [4, 4, 4, 4]   # List of downsampling scales.
    nonlinear_activation: "LeakyReLU" # Nonlinear activation function.
    nonlinear_activation_params:      # Parameters of nonlinear activation function.
        negative_slope: 0.2
    use_weight_norm: True             # Whether to use weight norm.

###########################################################
#                   STFT LOSS SETTING                     #
###########################################################
stft_loss_params:
    fft_sizes: [1024, 2048, 512]  # List of FFT size for STFT-based loss.
    hop_sizes: [120, 240, 50]     # List of hop size for STFT-based loss
    win_lengths: [600, 1200, 240] # List of window length for STFT-based loss.
    window: "hann_window"         # Window function for STFT-based loss

###########################################################
#               ADVERSARIAL LOSS SETTING                  #
###########################################################
use_feat_match_loss: true # Whether to use feature matching loss.
lambda_feat_match: 25.0   # Loss balancing coefficient for feature matching loss.
lambda_adv: 4.0           # Loss balancing coefficient for adversarial loss.

###########################################################
#                  DATA LOADER SETTING                    #
###########################################################
batch_size: 16             # Batch size.
batch_max_steps: 8192      # Length of each audio in batch. Make sure dividable by hop_size.
pin_memory: true           # Whether to pin memory in Pytorch DataLoader.
num_workers: 2             # Number of workers in Pytorch DataLoader.
remove_short_samples: true # Whether to remove samples the length of which are less than batch_max_steps.
allow_cache: true          # Whether to allow cache in dataset. If true, it requires cpu memory.

###########################################################
#             OPTIMIZER & SCHEDULER SETTING               #
###########################################################
generator_optimizer_params:
    lr: 0.0001             # Generator's learning rate.
    eps: 1.0e-6            # Generator's epsilon.
    weight_decay: 0.0      # Generator's weight decay coefficient.
generator_scheduler_params:
    step_size: 2000000     # Generator's scheduler step size.
    gamma: 0.5             # Generator's scheduler gamma.
                           # At each step size, lr will be multiplied by this parameter.
generator_grad_norm: 10    # Generator's gradient norm.
discriminator_optimizer_params:
    lr: 0.00005            # Discriminator's learning rate.
    eps: 1.0e-6            # Discriminator's epsilon.
    weight_decay: 0.0      # Discriminator's weight decay coefficient.
discriminator_scheduler_params:
    step_size: 2000000     # Discriminator's scheduler step size.
    gamma: 0.5             # Discriminator's scheduler gamma.
                           # At each step size, lr will be multiplied by this parameter.
discriminator_grad_norm: 1 # Discriminator's gradient norm.

###########################################################
#                    INTERVAL SETTING                     #
###########################################################
discriminator_train_start_steps: 100000 # Number of steps to start to train discriminator.
train_max_steps: 2000000                # Number of training steps.
save_interval_steps: 5000               # Interval steps to save checkpoint.
eval_interval_steps: 1000               # Interval steps to evaluate the network.
log_interval_steps: 100                 # Interval steps to record the training log.

###########################################################
#                     OTHER SETTING                       #
###########################################################
num_save_intermediate_results: 4  # Number of results to be saved as intermediate results.



