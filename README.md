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
To sample normally, run the following command. Model can be `5b`, `5b_lyrics`, `1b_lyrics`
``` 
python jukebox/sample.py --model=5b_lyrics --name=sample_5b --levels=3 --sample_length_in_seconds=20 \
--total_sample_length_in_seconds=180 --sr=44100 --n_samples=6 --hop_fraction=0.5,0.5,0.125
```
``` 
python jukebox/sample.py --model=1b_lyrics --name=sample_1b --levels=3 --sample_length_in_seconds=20 \
--total_sample_length_in_seconds=180 --sr=44100 --n_samples=16 --hop_fraction=0.5,0.5,0.125
```
The above generates the first `sample_length_in_seconds` seconds of audio from a song of total length `total_sample_length_in_seconds`.
To use multiple GPU's, launch the above scripts as `mpiexec -n {ngpus} python jukebox/sample.py ...` so they use `{ngpus}`

The samples decoded from each level are stored in `{name}/level_{level}`. 
You can also view the samples as an html with the aligned lyrics under `{name}/level_{level}/index.html`. 
Run `python -m http.server` and open the html through the server to see the lyrics animate as the song plays.  
A summary of all sampling data including zs, x, labels and sampling_kwargs is stored in `{name}/level_{level}/data.pth.tar`.

The hps are for a V100 GPU with 16 GB GPU memory. The `1b_lyrics`, `5b`, and `5b_lyrics` top-level priors take up 
3.8 GB, 10.3 GB, and 11.5 GB, respectively. The peak memory usage to store transformer key, value cache is about 400 MB 
for `1b_lyrics` and 1 GB for `5b_lyrics` per sample. If you are having trouble with CUDA OOM issues, try `1b_lyrics` or 
decrease `max_batch_size` in sample.py, and `--n_samples` in the script call.

On a V100, it takes about 3 hrs to fully sample 20 seconds of music. Since this is a long time, it is recommended to use `n_samples > 1` so you can generate as many samples as possible in parallel. The 1B lyrics and upsamplers can process 16 samples at a time, while 5B can fit only up to 3. Since the vast majority of time is spent on upsampling, we recommend using a multiple of 3 less than 16 like `--n_samples 15` for `5b_lyrics`. This will make the top-level generate samples in groups of three while upsampling is done in one pass.

To continue sampling from already generated codes for a longer duration, you can run
```
python jukebox/sample.py --model=5b_lyrics --name=sample_5b --levels=3 --mode=continue \
--codes_file=sample_5b/level_0/data.pth.tar --sample_length_in_seconds=40 --total_sample_length_in_seconds=180 \
--sr=44100 --n_samples=6 --hop_fraction=0.5,0.5,0.125
```
Here, we take the 20 seconds samples saved from the first sampling run at `sample_5b/level_0/data.pth.tar` and continue by adding 20 more seconds. 

You could also continue directly from the level 2 saved outputs, just pass `--codes_file=sample_5b/level_2/data.pth.tar`.
 Note this will upsample the full 40 seconds song at the end.

If you stopped sampling at only the first level and want to upsample the saved codes, you can run
```
python jukebox/sample.py --model=5b_lyrics --name=sample_5b --levels=3 --mode=upsample \
--codes_file=sample_5b/level_2/data.pth.tar --sample_length_in_seconds=20 --total_sample_length_in_seconds=180 \
--sr=44100 --n_samples=6 --hop_fraction=0.5,0.5,0.125
```
Here, we take the 20 seconds samples saved from the first sampling run at `sample_5b/level_2/data.pth.tar` and upsample the lower two levels.

## Prompt with your own music
If you want to prompt the model with your own creative piece or any other music, first save them as wave files and run
```
python jukebox/sample.py --model=5b_lyrics --name=sample_5b_prompted --levels=3 --mode=primed \
--audio_file=path/to/recording.wav,awesome-mix.wav,fav-song.wav,etc.wav --prompt_length_in_seconds=12 \
--sample_length_in_seconds=20 --total_sample_length_in_seconds=180 --sr=44100 --n_samples=6 --hop_fraction=0.5,0.5,0.125
```
This will load the four files, tile them to fill up to `n_samples` batch size, and prime the model with the first `prompt_length_in_seconds` seconds.

# Training
## VQVAE
To train a small vqvae, run
```
mpiexec -n {ngpus} python jukebox/train.py --hps=small_vqvae --name=small_vqvae --sample_length=262144 --bs=4 \
--audio_files_dir={audio_files_dir} --labels=False --train --aug_shift --aug_blend
```
Here, `{audio_files_dir}` is the directory in which you can put the audio files for your dataset, and `{ngpus}` is number of GPU's you want to use to train. 
The above trains a two-level VQ-VAE with `downs_t = (5,3)`, and `strides_t = (2, 2)` meaning we downsample the audio by `2**5 = 32` to get the first level of codes, and `2**8 = 256` to get the second level codes.  
Checkpoints are stored in the `logs` folder. You can monitor the training by running Tensorboard
```
tensorboard --logdir logs
```

# License 
Check license dependencies on OpenAI/Jukebox
[Noncommercial Use License](./LICENSE) 

It covers both released code and weights. 

