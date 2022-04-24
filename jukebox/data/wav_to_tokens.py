
import argparse
import numpy as np
import os
import torch
import torchaudio
from tqdm import tqdm
from jukebox.make_models import make_vqvae


def discretise_dataset(vqvae, in_dirpath, out_dirpath, device):
    if not os.path.exists(out_dirpath):
        os.mkdir(out_dirpath)

    wav_files = os.listdir(in_dirpath)
    for wf in tqdm(wav_files):
        in_filepath = os.path.join(in_dirpath, wf)
        out_filepath = os.path.join(out_dirpath, wf.replace(".wav", ".npy"))
        wav_to_tokens(in_filepath, vqvae, out_filepath, device)


def load_audio(filepath):
    waveform, sample_rate = torchaudio.load(filepath)
    # print(f"load_audio waveform {waveform.shape}")
    waveform = waveform.unsqueeze(0)  # (batch_size, channels, length)
    # print(f"load_audio waveform {waveform.shape}")
    return waveform


def save_z_code(path, z):
    #with open(path, "wb") as fout:
    #    np.save(fout, z)
    z_numpy = z.detach().cpu().numpy()
    # torch.save(z, path)
    np.save(path, z_numpy)


def wav_to_tokens(filepath, vqvae, out_filepath, device):

    audio = load_audio(filepath)
    audio = audio.to(device, non_blocking=True)
    z = vqvae.encode(x=audio, bs_chunks=1)
    z0 = z[0]  # first layer z codes
    # print(f"wav_to_tokens len(z) {len(z)}, z[0].shape {z[0].shape}")
    save_z_code(out_filepath, z0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VQVAE prior training')
    parser.add_argument('--in_dirpath', type=str, help='path to the directory that contains .wav files')
    parser.add_argument('--vq_fname', type=str, help='path to the file that contains pretrained VQ encoder')
    parser.add_argument('--out_dirpath', type=str, help='path to the output directory')
    args = parser.parse_args()

    if not os.path.exists(args.out_dirpath):
        os.mkdir(args.out_dirpath)

    vqvae = make_vqvae(hps, device)

    wav_files = os.listdir(args.in_dirpath)
    for wf in tqdm(wav_files):
        in_filepath = os.path.join(args.in_dirpath, wf)
        out_filepath = os.path.join(args.out_dirpath, wf.replace(".wav", ".npy"))
        wav_to_tokens(in_filepath, vqvae, out_filepath)
