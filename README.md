# Speech VQVAE
Code for Aalto University master's thesis "Learning neural discrete representations for speech"

## Speech samples - original and reconstruction

### Original speech sample from dataset LJSpeech


https://user-images.githubusercontent.com/49818900/164986471-37845368-4b0d-41cd-8331-abad631acd9a.mp4


### Best speech sample - using wider residual networks


https://user-images.githubusercontent.com/49818900/164986586-7e10d994-ec80-4a74-998d-2e53001da5f9.mp4




https://user-images.githubusercontent.com/49818900/164986985-95afc9eb-0d91-4e93-b119-ca8d795af5dc.mp4



https://user-images.githubusercontent.com/49818900/164986986-79e5b97c-fe8b-43b4-88ee-23aa95f78a56.mp4




### Baseline speech-VQVAE



https://user-images.githubusercontent.com/49818900/164986601-2f22eb4e-a934-4630-9b02-0b54c7e638bb.mp4


### Downssampling factor experiment comparison

Downsampling factor of 32



https://user-images.githubusercontent.com/49818900/164986626-5282c53d-0ac0-436f-8735-3bd9ad45d392.mp4

Downsampling factor of 64


https://user-images.githubusercontent.com/49818900/164986637-b448241b-35e1-4ee6-8ae0-0ddb9381c1cd.mp4

Downsampling factor of 128


https://user-images.githubusercontent.com/49818900/164986640-1c37e8e0-1b4d-4fbb-8cfd-8978d8d9ea30.mp4


Downsampling factor of 256


https://user-images.githubusercontent.com/49818900/164986649-bc9b1721-949d-4f45-ad01-f931c6feef76.mp4


### Increasing width of the residual network

Width of 64


https://user-images.githubusercontent.com/49818900/164986672-6450008c-a57b-4f21-8659-66d29112d314.mp4

Width of 128


https://user-images.githubusercontent.com/49818900/164986675-4e64c454-178e-4e8f-a284-b1044cc87ac7.mp4


## Text-to-speech Transformer model sample from synthesis



https://user-images.githubusercontent.com/49818900/164986705-24d45228-012f-4e27-b139-7b43731851fe.mp4


Sample from model with only 2 attention heads to the text where the model does not learn to follow the given text but generates sounds freely from the learnt tokens.



https://user-images.githubusercontent.com/49818900/164988407-ce5048eb-e817-4ff9-86ad-07caaded7696.mp4



##  Install
Install the conda package manager from https://docs.conda.io/en/latest/miniconda.html    
    
conda env create --file env.yml

conda activate jukebox

conda install mpi4py=3.0.3 # if this fails, try: pip install mpi4py==3.0.3

conda install pytorch=1.4 torchvision=0.5 cudatoolkit=10.0 -c pytorch

git clone https://github.com/jarkkotulensalo/speech-vqvae.git
 
# Optional: Apex for faster training with fused_adam
conda install pytorch=1.1 torchvision=0.3 cudatoolkit=10.0 -c pytorch
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./apex
```

# Sampling
## Sampling from scratch
To sample normally, run the following command.
``` 
python jukebox/sample.py --model=single_model --name=sample_single --levels=1 --sample_length_in_seconds=5.2 \
--total_sample_length_in_seconds=5.2 --sr=22050 --n_samples=1 --hop_fraction=0.5,0.5,0.125 --epochs=1
```

# Training
## VQVAE
```
# Train VQ-VAE encoder-decoder
python3 jukebox/train.py --hps=vqvae --name=vqvae_downs_5 --sample_length=12800 --bs=128 \
--train --test --save --epochs=1000 --local_logdir='logs_audio' --prior=False --labels=True

The above trains a two-level VQ-VAE with `downs_t = (5,3)`, and `strides_t = (2, 2)` meaning we downsample the audio by `2**5 = 32` to get the first level of codes, and `2**8 = 256` to get the second level codes.  

## Train prior (transformer)
```
python3 jukebox/train.py --hps=vqvae,single_prior_70M,all_fp16,cpu_ema \
--name=prior_downs_5 \
--sample_length=12800 --bs=8 --train --test --prior --levels=1 --level=0 --epochs=100 --discretise=True --local_logdir='logs_audio'
```

Checkpoints are stored in the `logs` folder. You can monitor the training by running Tensorboard
```
tensorboard --logdir logs
```

# License 
Check license dependencies on OpenAI/Jukebox
[Noncommercial Use License](./LICENSE) 

It covers both released code and weights. 

