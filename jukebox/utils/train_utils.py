import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def create_codebook_dataset(model, orig_model, logger, metrics, data_processor, hps):
    model.eval()
    orig_model.eval()

    with t.no_grad():
        for i, batch in logger.get_range(data_processor.test_loader):
            if isinstance(batch, (tuple, list)):
                # x, y = x
                waveform, normalized_transcript = batch
                x = waveform
                y = None
            else:
                y = None
            x_in = x.to('cuda', non_blocking=True)

            # Evaluate codebook
            ###################
            # We want the final size to equal tacotron sixe of 12.5ms ~ 1764
            # Chorowski et al. (2019) used sequences of length 5120 time-domain samples (320 ms)
            zs = orig_model.encode(x_in)

            # print(f"encode zs {zs[0].shape}")  # (4 x 400) (batches x dimensions)
            for batch_num in range(zs[0].shape[0]):
                emb_level1 = zs[0][batch_num, :]
                embeddings_level.append(emb_level1)
                # print(f"emb_level1 {emb_level1.shape}")  # (torch.Size([400]))
                if hps.levels > 1:
                    emb_level2 = zs[1][batch_num, :]
                    embeddings_level2.append(emb_level2)

def _plot_sample_wav(sample, hps):

    x = np.arange(sample.shape[0])
    fig = plt.figure()
    plt.plot(x, sample)
    plt.xlabel("Time step")
    plt.ylabel("Waveform")
    # plt.colorbar()
    plt.savefig(f'{hps.local_logdir}/{hps.name}/sample_wav.png', dpi=fig.dpi)
    plt.close(fig)


def plot_alignment_to_numpy(alignment, hps, layer=1, head=1, info=None):

    # m = np.mean(alignment, 0)
    # std = np.std(alignment, 0)
    # print(f"m {m}, std {std}")

    alignment_single = alignment[0, :, :].T
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(alignment_single, aspect='auto', origin='lower',
                   interpolation='none')
    fig.colorbar(im, ax=ax)
    xlabel = 'Speech token position'
    if info is not None:
        xlabel += '\n\n' + info
    plt.xlabel(xlabel)
    plt.ylabel('Sentence token position')
    title = f"Alignment layer {layer}, head {head}"
    plt.title(f"{title}")
    plt.tight_layout()
    plt.savefig(f'{hps.local_logdir}/{hps.name}/{info}_attention_layer_{layer}_head_{head}.png', dpi=fig.dpi)
    plt.close()


def plot_previous_token_alignment(alignment, hps, layer=1, head=1, info=None):

    alignment_single = alignment[0, :, :].T
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(alignment_single, aspect='auto', origin='lower',
                   interpolation='none')
    fig.colorbar(im, ax=ax)
    xlabel = 'Audio token position'
    if info is not None:
        xlabel += '\n\n' + info
    plt.xlabel(xlabel)
    plt.ylabel('Previous audio token position')
    title = f"Alignment layer {layer}, head {head}"
    plt.title(f"{title}")
    plt.tight_layout()
    plt.savefig(f'{hps.local_logdir}/{hps.name}/{info}_attention_layer_{layer}_head_{head}.png', dpi=fig.dpi)
    plt.close()


def _plot_mel_spec(wav_in, wav_out, hps):

    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=[6.4, 4.8 * 2])
    ax[0].set(title='Original Mel-frequency power spectrogram')
    ax[1].set(title='Decoded Mel-frequency power spectrogram')

    S = librosa.feature.melspectrogram(y=wav_in, sr=hps.sr)
    img = librosa.display.specshow(librosa.power_to_db(S, ref=np.max), ax=ax[0], y_axis='log')
    S2 = librosa.feature.melspectrogram(y=wav_out, sr=hps.sr)
    img2 = librosa.display.specshow(librosa.power_to_db(S2, ref=np.max), ax=ax[1], x_axis='time', y_axis='log')

    # plt.colorbar(img)
    plt.savefig(f'{hps.local_logdir}/{hps.name}/melspec.png', dpi=fig.dpi)
    plt.close(fig)


def _plot_codebook_sample(sample, wav_in, wav_out, level, hps):

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=[6.4, 4.8 * 2])

    x = np.arange(wav_in.shape[0])
    ax1.plot(x, wav_in)
    x2 = np.arange(sample.shape[0])
    ax2.scatter(x2, sample, marker='s')

    x3 = np.arange(wav_out.shape[0])
    ax3.plot(x3, wav_out)

    ax1.set_xlabel("Time step")
    ax1.set_ylabel("Original waveform")

    ax2.set_xlabel("Compressed time step")
    ax2.set_ylabel("Codebook index")

    ax3.set_xlabel("Time step")
    ax3.set_ylabel("Decoded waveform")
    # plt.colorbar()
    plt.savefig(f'{hps.local_logdir}/{hps.name}/codebook_sample_level{level}.png', dpi=fig.dpi)
    plt.close(fig)


def _plot_codebook_histogram(codebook, level, hps):
    # print(codebook.shape)
    bins = np.arange(hps.l_bins + 1)
    labels = np.arange(hps.l_bins)
    flattened = codebook.flatten()
    freq, bins = np.histogram(flattened, bins=bins, density=True)

    arr1inds = freq.argsort()
    sorted_freq = freq[arr1inds[::-1]]
    # sorted_labels = labels[arr1inds[::-1]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[6.4 * 2, 4.8])
    ax1.bar(labels, freq)
    ax1.set_xlabel("Codebook index")

    new_labels = labels / hps.l_bins * 100
    freq_labels = np.arange(0, 101, 10)
    ax2.bar(new_labels, sorted_freq)
    ax2.set_xticks(freq_labels)
    ax2.set_xticklabels(freq_labels)
    ax2.set_xlabel("Frequency distribution (%)")
    plt.savefig(f'{hps.local_logdir}/{hps.name}/codebook_histogram_level{level}.png', dpi=fig.dpi)
    plt.close('all')