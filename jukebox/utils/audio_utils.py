import numpy as np
import torch as t
import jukebox.utils.dist_adapter as dist
import soundfile
import librosa
from pesq import pesq
from jukebox.utils.dist_utils import print_once

class DefaultSTFTValues:
    def __init__(self, hps):
        self.sr = hps.sr
        self.n_fft = 2048
        self.hop_length = 256
        self.window_size = 6 * self.hop_length

class STFTValues:
    def __init__(self, hps, n_fft, hop_length, window_size):
        self.sr = hps.sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window_size = window_size


def calc_pesq(x, x_hat):

    # print(f"calc_pesq x {x.shape}, x_hat {x_hat.shape}")
    scores = []
    pesq_val_np = np.empty(shape=(x.shape[0]))
    for i in range(x.shape[0]):
        x_i = librosa.resample(x[i, :], orig_sr=22050, target_sr=16000)
        x_hat_i = librosa.resample(x_hat[i, :], orig_sr=22050, target_sr=16000)
        pesq_val_np[i] = pesq(ref=x_i, deg=x_hat_i, fs=16000, mode='wb')
    mean_score = np.mean(pesq_val_np)
    # print(f"calc_pesq mean_score {mean_score}")
    torch_mean_score = t.from_numpy(np.asarray(mean_score))
    return torch_mean_score

def SNR(x, x_hat, epsilon=1e-9):
    """ Signal-to-Noise Ratio

    Args:
        x: signal
        x_hat: signal

    Author:
        Lauri Juvela
        lauri.juvela@aalto.fi
    """
    # print(f"x {x.shape}, x_hat {x_hat.shape}, ")

    n_fft = 1024
    window = t.hann_window(n_fft).to(x.device)
    # STFT, normalized=True uses the orthonormal mode, which preserves energy
    X = t.stft(x, n_fft, hop_length=n_fft // 2, win_length=n_fft, window=window,
                   center=False, normalized=True, return_complex=True)
    X_hat = t.stft(x_hat, n_fft, hop_length=n_fft // 2, win_length=n_fft, window=window,
                       center=False, normalized=True, return_complex=True)

    # print(f"X {X.shape} X_hat {X_hat.shape}")
    # Signal power spectrum
    P_X = X.abs().pow(2)
    # Model power spectrum
    P_X_hat = X_hat.abs().pow(2)
    # Error power spectrum
    P_E = t.abs(P_X - P_X_hat)
    # Calculate SNR, sum over spectral bins
    # print(f"P_X {P_X.shape}, P_E {P_E.shape}")
    snr = P_X.sum(dim=1) / t.clamp(P_E.sum(dim=1), min=1e-9)

    mask = (snr > epsilon)
    # print(f"snr {snr.shape}, snr[mask] {snr[mask].shape}")

    # calculate SNR in dB
    log_snr = 10.0 * t.log10((snr[mask]))
    # reduce average over batch and channels

    # print(f"log_snr {log_snr.shape} {log_snr}")
    # print(f"t.mean(log_snr) {t.mean(log_snr)}")
    return t.mean(log_snr)

def calculate_bandwidth(dataset, hps, duration=3):
    """
    'bandwidth' is used to scale loss values to similar level.
    loss_fn / bandwidth['spec']
    loss_fn / bandwidth['l1']
    loss_fn / bandwidth['l2']
    Why default duration=600? Somewhat related to looping
    :param dataset: audio dataset as torch.utils.data.Dataset
    :param hps: hyperparameters as dict
    :param duration:
    :return:
    """
    hps = DefaultSTFTValues(hps)
    # n_samples = int(dataset.sr * duration)
    sample_rate = hps.sr
    print(f"sample rate is {sample_rate}")
    # n_samples = int(sample_rate * duration)
    n_samples = len(dataset)
    l1, total, total_sq, n_seen, idx = 0.0, 0.0, 0.0, 0.0, dist.get_rank()
    spec_norm_total, spec_nelem = 0.0, 0.0
    while idx < n_samples:
        # print(f"idx is {idx} n_samples is {n_samples} and n_seen is {n_seen}")
        x = dataset[idx]
        #print(x)
        if x is not None:
            # print(f"calculate_bandwidth x is {x}")
            if isinstance(x, (tuple, list)):
                x, y = x[0], x[1]

            # print(f"calculate_bandwidth x shape is {x.shape}")
            samples = x.numpy()
            # print(f"np.mean(samples, axis=1 {np.mean(samples, axis=1)}, np.mean(samples, axis=0) {np.mean(samples, axis=0)}")
            stft = librosa.core.stft(np.mean(samples, axis=0),
                                     n_fft=hps.n_fft,
                                     hop_length=hps.hop_length,
                                     win_length=hps.window_size)

            spec = np.absolute(stft)
            spec_norm_total += np.linalg.norm(spec)
            spec_nelem += 1
            n_seen += int(np.prod(samples.shape))
            l1 += np.sum(np.abs(samples))
            total += np.sum(samples)
            total_sq += np.sum(samples ** 2)
        #else:
        #    print(f"dataset sample idx {idx} is missing")

        idx += max(16, dist.get_world_size())

    if dist.is_available():
        from jukebox.utils.dist_utils import allreduce
        n_seen = allreduce(n_seen)
        total = allreduce(total)
        total_sq = allreduce(total_sq)
        l1 = allreduce(l1)
        spec_nelem = allreduce(spec_nelem)
        spec_norm_total = allreduce(spec_norm_total)

    mean = total / n_seen
    bandwidth = dict(l2 = total_sq / n_seen - mean ** 2,
                     l1 = l1 / n_seen,
                     spec = spec_norm_total / spec_nelem)
    print_once(bandwidth)
    return bandwidth

def audio_preprocess(x, hps):
    # Extra layer in case we want to experiment with different preprocessing
    # For two channel, blend randomly into mono (standard is .5 left, .5 right)

    # x: NTC
    x = x.float()
    print(f"audio_preprocess input x.shape {x.shape}")
    if x.shape[-1]==2:
        if hps.aug_blend:
            mix=t.rand((x.shape[0],1), device=x.device) #np.random.rand()
        else:
            mix = 0.5
        x=(mix*x[:,:,0]+(1-mix)*x[:,:,1])
    elif x.shape[-1]==1:
        x=x[:,:,0]
    else:
        assert False, f'Expected channels {hps.channels}. Got unknown {x.shape[-1]} channels'

    # x: NT -> NTC
    x = x.unsqueeze(2)

    print(f"audio_preprocess output x.shape {x.shape}")
    return x

def audio_postprocess(x, hps):
    return x

def stft(sig, hps):
    #stft1 = t.stft(sig, hps.n_fft, hps.hop_length, win_length=hps.window_size, window=t.hann_window(hps.window_size, device=sig.device))
    stft2 = t.stft(sig,
                   hps.n_fft,
                   hps.hop_length,
                   win_length=hps.window_size,
                   window=t.hann_window(hps.window_size, device=sig.device),
                   return_complex=False)
    # print(f"orig {stft1.shape}, new {stft2.shape} equal? {t.sum(t.eq(stft1, stft2))}")

    return stft2

def spec(x, hps):
    """
    Calculates norm over stft of the waveform or waveform of batches (batch_size, waveform)
    :param x: torch array in (batch_size, waveform)
    :param hps: hyperparameter dict
    :return: torch array in (batch size)
    """
    # print(f"x.shape {x.shape}, stft(x, hps) {stft(x, hps).shape}, t.norm(stft(x, hps), p=2, dim=-1) {t.norm(stft(x, hps), p=2, dim=-1).shape}")
    return t.norm(stft(x, hps), p=2, dim=-1)

def norm(x):
    # print(f"norm")
    # print(f"x {x.shape} -> x.view(x.shape[0], -1) {x.view(x.shape[0], -1).shape}")
    # print(f"(x.view(x.shape[0], -1) ** 2).sum(dim=-1) {(x.view(x.shape[0], -1) ** 2).sum(dim=-1).shape}")
    # print(f"(x.view(x.shape[0], -1) ** 2).sum(dim=-1).sqrt() {(x.view(x.shape[0], -1) ** 2).sum(dim=-1).sqrt().shape}")
    return (x.view(x.shape[0], -1) ** 2).sum(dim=-1).sqrt()

def squeeze(x):
    """
    Calculating mean waveform over channels (1 or 2 channels)
    :param x:
    :return:
    """
    if len(x.shape) == 3:
        # assert x.shape[-1] in [1,2]
        assert x.shape[1] in [1, 2]
        x = t.mean(x, 1)
        # x = t.mean(x, -1)
    if len(x.shape) != 2:
        raise ValueError(f'Unknown input shape {x.shape}')
    return x

def spectral_loss(x_in, x_out, hps):
    hps = DefaultSTFTValues(hps)
    # print(f"x_in {x_in.shape}, x_out {x_out.shape}")
    spec_in = spec(squeeze(x_in.float()), hps)
    spec_out = spec(squeeze(x_out.float()), hps)
    return norm(spec_in - spec_out)

def multispectral_loss(x_in, x_out, hps):
    losses = []
    assert len(hps.multispec_loss_n_fft) == len(hps.multispec_loss_hop_length) == len(hps.multispec_loss_window_size)
    args = [hps.multispec_loss_n_fft,
            hps.multispec_loss_hop_length,
            hps.multispec_loss_window_size]
    for n_fft, hop_length, window_size in zip(*args):
        hps = STFTValues(hps, n_fft, hop_length, window_size)
        spec_in = spec(squeeze(x_in.float()), hps)
        spec_out = spec(squeeze(x_out.float()), hps)
        losses.append(norm(spec_in - spec_out))
    return sum(losses) / len(losses)

def spectral_convergence(x_in, x_out, hps, epsilon=2e-3):
    hps = DefaultSTFTValues(hps)
    spec_in = spec(squeeze(x_in.float()), hps)
    spec_out = spec(squeeze(x_out.float()), hps)

    gt_norm = norm(spec_in)
    residual_norm = norm(spec_in - spec_out)
    mask = (gt_norm > epsilon).float()
    return (residual_norm * mask) / t.clamp(gt_norm, min=epsilon)

def log_magnitude_loss(x_in, x_out, hps, epsilon=1e-4):
    hps = DefaultSTFTValues(hps)
    spec_in = t.log(spec(squeeze(x_in.float()), hps) + epsilon)
    spec_out = t.log(spec(squeeze(x_out.float()), hps) + epsilon)
    return t.mean(t.abs(spec_in - spec_out))

def load_audio(file, sr, offset, duration, mono=False):
    # Librosa loads more filetypes than soundfile
    x, _ = librosa.load(file, sr=sr, mono=mono, offset=offset/sr, duration=duration/sr)
    if len(x.shape) == 1:
        x = x.reshape((1, -1))
    return x    

def save_wav(fname, aud, sr):
    # clip before saving?
    # print(f"save_wav aud {aud.shape}")
    aud = aud.permute(0, 2, 1)  # Added permutation
    print_once(f"Changed save_wav aud shape to {aud.shape}")
    aud = t.clamp(aud, -1, 1).cpu().numpy()
    for i in list(range(aud.shape[0])):
        print(f"save_wav aud[i] {aud[i].shape}")
        soundfile.write(f'{fname}/item_{i}.wav', aud[i], samplerate=sr, format='wav')


