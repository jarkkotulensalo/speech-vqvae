import numpy as np
import os
import torch as t
import torchaudio
import jukebox.utils.dist_adapter as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Dataset, BatchSampler, RandomSampler
from jukebox.utils.dist_utils import print_all
from jukebox.utils.audio_utils import calculate_bandwidth
import random
from jukebox.data.text_processor import TextProcessor

from torch.nn.utils.rnn import pad_sequence



class AudioDataset(Dataset):
    def __init__(self, dataset):
        """
        :param dataset: torchaudio.datasets object
        """
        super().__init__()
        self.dataset = dataset

    def __len__(self):
        print(f"len AudioDataset is {len(self.dataset)}")
        return len(self.dataset)

    def __getitem__(self, x):
        wav, sample_rate, text, norm_text = self.dataset[x]
        return wav, norm_text


class VqvaeDataset(Dataset):
    def __init__(self, dataset, vqvae_dir):
        """
        Args:
            dataset: torchaudio.datasets object
            vqvae_dir: str, folder path for encoded latent vectors
        """
        super().__init__()
        self.dataset = dataset
        self.vqvae_dir = vqvae_dir
        # self.vqvae_files = os.listdir(vqvae_dir)
        # print(f"self.dataset._flist {self.dataset.dataset._flist}")
        # for i, item in enumerate(self.dataset.dataset._flist):
        #     print(f"LJSpeech {i}th is {item[0]}. VQVAE is {self.vqvae_files[i]}")

    def __len__(self):
        print(f"len VqvaeDataset is {len(self.dataset)}")
        return len(self.dataset)

    def __getitem__(self, idx):
        # print(f"VqvaeDataset idx is {idx}")
        wav, _, _, norm_text = self.dataset[idx]
        name_of_file = self.dataset._flist[idx][0]
        name_of_pt_file = f"{name_of_file}.npy"
        # print(f"{name_of_file} -> {name_of_pt_file}")
        z = self._load_encoded_wav(name_of_pt_file)
        return wav, norm_text, z

    def _load_encoded_wav(self, filename):
        """
        Load pre-encoded VQVAE codes from files
        :param filename: string
        :return: torch.Size([1, sequence length], dtype=torch.LongTensor)
        """
        z = np.load(os.path.join(self.vqvae_dir, filename))
        z = t.from_numpy(z)
        # print(f"_load_encoded_wav vqvae_code type: {vqvae_code.type()}, shape: {vqvae_code.shape}")
        z = z.type(dtype=t.LongTensor)  # FloatTensor LongTensor
        return z


class DataProcessor():
    def __init__(self, hps, segment=True):
        """
        Use segment=True when you want to create a dataset that crops a random segment from the total audio.
        If segment=False, dataset return the full audio piece.
        :param hps: hyperparameter dict
        :param segment: boolean
        """
        # self.dataset = FilesAudioDataset(hps)
        duration = 1 if hps.prior else 600

        self.train_test_split = hps.train_test_split
        self.segment = segment
        self.text_processor = TextProcessor(v3=False)
        self._create_datasets(hps)
        self._create_samplers(hps)
        self._create_data_loaders(hps)
        self._print_stats(hps)

        # fix random seed
        t.manual_seed(64)

        #self.half_context = True
        #print(f"______________________ Splitting text context also into half {self.half_context}")

    def set_epoch(self, epoch):
        self.train_sampler.set_epoch(epoch)
        self.test_sampler.set_epoch(epoch)

    def _create_datasets(self, hps):
        """
        Create dataset to sample raw waveform to f.e. 1 second pieces
        Data format (batch_size, channels, waveform) = (batch_size, 1, sample_length)
        Splits dataset into self.train_dataset, self.test_dataset
        :param hps: hps parameters set in hparams.py
        :return:
        self.train_dataset
        self.test_dataset
        """
        """
        self.dataset = FilesAudioDataset(hps)
        duration = 1 if hps.prior else 600
        hps.bandwidth = calculate_bandwidth(self.dataset, hps, duration=duration)
        """
        dataset = torchaudio.datasets.LJSPEECH(root="./jukebox/datasets/",
                                               download=True)

        raw_dataset = AudioDataset(dataset)
        duration = 1 if hps.prior else 600
        hps.bandwidth = calculate_bandwidth(raw_dataset, hps, duration=duration)


        if hps.prior:
            # switch raw audio to tokens
            vqvae_dir = f"./jukebox/datasets/LJSpeech-1.1/tokens"
            raw_dataset = VqvaeDataset(dataset, vqvae_dir=f"./jukebox/datasets/LJSpeech-1.1/tokens")


        dataset_length = len(raw_dataset)
        train_length = int(self.train_test_split * dataset_length)
        others_length = dataset_length - train_length
        lengths = [train_length, others_length]

        t.manual_seed(64)
        self.train_dataset, self.test_dataset = t.utils.data.random_split(raw_dataset,
                                                                          lengths=lengths,
                                                                          generator=t.Generator().manual_seed(64)
                                                                          )

        # val_test_length = [int(0.9 * others_length), int(0.1 * others_length)]
        # self.val_dataset, self.test_dataset = t.utils.data.random_split(testing, lengths=val_test_length)

    def _create_samplers(self, hps):
        if not dist.is_available():
            self.train_sampler = BatchSampler(RandomSampler(self.train_dataset), batch_size=hps.bs, drop_last=True)
            # self.test_sampler = BatchSampler(RandomSampler(self.test_dataset), batch_size=hps.bs, drop_last=True)
            self.test_sampler = BatchSampler(self.test_dataset, batch_size=hps.bs, drop_last=True)
        else:
            print(f"dist.is_available() {dist.is_available()}")
            self.train_sampler = DistributedSampler(self.train_dataset)
            self.test_sampler = DistributedSampler(self.test_dataset, shuffle=False)

    def _create_data_loaders(self, hps):

        # Loader to load mini-batches
        if hps.labels:
            if hps.prior:
                print(f"hps.prior == True")
                #collate_fn = lambda batch: tuple(
                #    t.stack([self._collate_helper(i, b[i], hps) for b in batch], 0) for i in range(3))
                collate_fn = lambda batch: tuple(t.stack([self._collate_helper(i, b, hps) for b in batch], 0) for i in range(3))
                # collate_fn = lambda batch: tuple(
                #    t.stack(self._collate_helper(i, batch, hps), 0) for i in range(3))
            else:
                print(f"hps.prior == False")
                collate_fn = lambda batch: tuple(t.stack([self._collate_helper(i, b, hps) for b in batch], 0) for i in range(2))
            # collate_fn = lambda batch: tuple(t.stack([self._sample_audio_segment(wav, hps.sample_length)], 0) for (wav, _) in batch, t.stack([self._tokenise_text(text, hps.n_tokens)], 0) for (_, text) in batch)
        else:
            print(f"hps.labels == False")
            collate_fn = lambda batch: t.stack([self._sample_audio_segment(wav, hps.sample_length, hps.n_ctx) for (wav, _) in batch], 0)

        # batch_sampler = np.split(file_len.argsort()[::-1], batch_size)

        self.train_loader = DataLoader(self.train_dataset, batch_size=hps.bs, num_workers=0,
                                       sampler=self.train_sampler, pin_memory=False,
                                       drop_last=True, collate_fn=collate_fn)  # batch_sampler=batch_sampler

        self.test_loader = DataLoader(self.test_dataset, batch_size=hps.bs, num_workers=0,
                                      sampler=self.test_sampler, pin_memory=False,
                                      drop_last=False, collate_fn=collate_fn)


    def _collate_helper(self, i, values, hps):
        #print(f"values {values.shape}")
        value = values[i]
        #print(f"batch i {i} is {value.shape}")
        if i == 0:  # how to process raw audio
            # print(f"process raw audio value {value.shape}")
            val = t.stack([self._sample_audio_segment(value, hps.sample_length, hps.n_ctx)], 0)
        elif i == 1:  # how to process normalised text
            # print(f"process normalised text")
            val = t.stack([self._tokenise_text(value, hps.n_tokens)], 0)
        elif i == 2:  # how to process vqvae codes
            # print(f"process z")
            val = t.stack([self._collate_vqvae(value, hps.n_ctx)], 0)
        return val


    def _tokenise_text(self, norm_text, n_tokens):
        lyrics = self.text_processor.clean(norm_text)
        full_tokens = self.text_processor.tokenise(lyrics)

        tokens = full_tokens + [0] * (n_tokens - len(full_tokens))
        y = np.array(tokens, dtype=np.int64)
        y_torch = t.from_numpy(y)

        return y_torch

    def _sample_audio_segment(self, wav, sample_length, n_ctx, vqvae_downs=32):
        """
        Sample a fixed size segment of the total audio, where total length = sample_length.
        use randomisation and take only one sample from original audio
        :param wav: original torch audio waveform shape (batches, channels, sequence)
        :param sample_length: sample length
        :return: waveform as torch shape (wave lenght,)
        """
        # print(f"self.segment {self.segment}")
        if self.segment:
            # sample a segment of the raw audio
            wav_length = wav.shape[1]
            limit_min_idx = wav_length - sample_length - 1
            limit_min_idx = max(0, limit_min_idx)
            # print(f"index {x} wav_length is {wav_length} and limit_min_idx is {limit_min_idx}")
            sample_starting_idx = random.randint(0, limit_min_idx)
            sample_ending_idx = sample_starting_idx + sample_length
            # print(f"sample_starting_idx {sample_starting_idx} sample_ending_idx is {sample_ending_idx}")
            segmented_wav = wav[:, sample_starting_idx:sample_ending_idx]
            # segmented_wav = segmented_wav.squeeze(0)
            # print(f"segmented_wav {segmented_wav.shape}")
        else:
            wav = wav.to('cpu')
            # print(f"_sample_audio_segment wav {wav.type()}")
            len_wav = wav.shape[1]

            # VQVAE dataset contains raw audio
            # Fix vector length to equal n_ctx * downsampling factor (longest sample 10.4s ~ 229312 bits)
            n_tokens = int(n_ctx * vqvae_downs)  # n_ctx * downsampling factor (based on VQVAE) = 114688
            # If we want to limit the size of the context
            if len_wav > n_tokens:
                # print(f"len_wav '{len_wav} is greater than n_tokens {n_tokens}'. We'll cut the waveform into half.")
                len_wav = int(len_wav / 2)
                wav = wav[:, :len_wav]
                # print(f"wav.shape {wav.shape}")

            segmented_wav = t.cat([wav, t.zeros(1, n_tokens - len_wav)], dim=1)

        return segmented_wav.squeeze(0)

    def _collate_vqvae(self, z, n_ctx):
        """
        Prior dataset contains pre-encoded VQVAE vectors
        :param wav: original torch audio waveform shape (batches, channels, sequence)
        :param sample_length: sample length
        :return: waveform as torch shape (wave length,)
        """

        # print(f"vqvae_code {vqvae_code.shape}")
        z = z.to('cpu')
        len_z = z.shape[1]
        if len_z > n_ctx:
            # print(f"len_wav '{len_z} is greater than n_tokens {n_ctx}'")
            # split context to half
            len_z = int(len_z / 2)
            z = z[:, :len_z]
            # print(f"z.shape {z.shape}")

        # Fix vector size to equal n_ctx
        # Instead of t.zeros, add VQVAE index which is silent phoneme acc. to histogram of used VQVAE model or samples. (f.e. 343)
        zero_padding_mask = t.add(t.zeros(1, n_ctx - len_z).type(dtype=t.LongTensor), 557)  # int padding
        z = t.cat([z, zero_padding_mask], dim=1) # zero padding at the end of audio

        return z.squeeze(0)

    def _print_stats(self, hps):
        print_all(f"Train {len(self.train_dataset)} samples. Test {len(self.test_dataset)} samples")
        print_all(f'Train sampler: {self.train_sampler}')
        print_all(f'Train loader: {len(self.train_loader)}')
        print(f"batch_size {hps.bs}")
        print(f"len train {len(self.train_dataset)}, len test {len(self.test_dataset)}")
