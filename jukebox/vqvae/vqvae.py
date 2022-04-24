import numpy as np
import torch as t
import torch.nn as nn

from jukebox.vqvae.encdec import Encoder, Decoder, assert_shape
from jukebox.vqvae.bottleneck import NoBottleneck, Bottleneck
from jukebox.utils.logger import average_metrics
from jukebox.utils.audio_utils import spectral_convergence, spectral_loss, multispectral_loss, audio_postprocess, SNR, calc_pesq


def dont_update(params):
    for param in params:
        param.requires_grad = False

def update(params):
    for param in params:
        param.requires_grad = True

def calculate_strides(strides, downs):
    """
    Added possibility to iterate with one element and return stride for single level
    :param strides: as tuple, stride for each level
    :param downs: as tuple, downsampling factor (stride**downs) for each level
    :return: list of strides for each level
    """
    print(f"strides {strides}, downs {downs}")
    return [stride ** down for stride, down in zip(strides, downs)]

def _loss_fn(loss_fn, x_target, x_pred, hps):

    if loss_fn == 'l1':
        return t.mean(t.abs(x_pred - x_target)) / hps.bandwidth['l1']
    elif loss_fn == 'l2':
        return t.mean((x_pred - x_target) ** 2) / hps.bandwidth['l2']
    elif loss_fn == 'linf':
        residual = ((x_pred - x_target) ** 2).reshape(x_target.shape[0], -1)
        values, _ = t.topk(residual, hps.linf_k, dim=1)
        return t.mean(values) / hps.bandwidth['l2']
    elif loss_fn == 'lmix':
        loss = 0.0
        if hps.lmix_l1:
            loss += hps.lmix_l1 * _loss_fn('l1', x_target, x_pred, hps)
        if hps.lmix_l2:
            loss += hps.lmix_l2 * _loss_fn('l2', x_target, x_pred, hps)
        if hps.lmix_linf:
            loss += hps.lmix_linf * _loss_fn('linf', x_target, x_pred, hps)
        return loss
    else:
        assert False, f"Unknown loss_fn {loss_fn}"


class VQVAE(nn.Module):
    def __init__(self, input_shape, levels, downs_t, strides_t,
                 emb_width, l_bins, mu, commit, spectral, multispectral,
                 multipliers=None, use_bottleneck=True, noise_layer_idx=0, **block_kwargs):
        super().__init__()

        self.sample_length = input_shape[0]
        x_shape, x_channels = input_shape[:-1], input_shape[-1]
        self.x_shape = x_shape

        # downsamples defines how many conv1d-Resnet1D blocks are appended
        # stride ** downsample
        self.downsamples = calculate_strides(strides_t, downs_t)  # stride defines how many steps to skip
        self.hop_lengths = np.cumprod(self.downsamples)
        self.z_shapes = z_shapes = [(x_shape[0] // self.hop_lengths[level],) for level in range(levels)]
        self.levels = levels

        print(f"self.sample_length {self.sample_length}")
        print(f"self.x_shape {self.x_shape}")
        print(f"self.downsamples {self.downsamples}")
        print(f"self.hop_lengths {self.hop_lengths}")
        print(f"self.z_shapes {self.z_shapes}")
        print(f"self.levels {self.levels}")
        print(f"decoder_noise_layer_idx {noise_layer_idx}")

        if multipliers is None:
            self.multipliers = [1] * levels
        else:
            assert len(multipliers) == levels, "Invalid number of multipliers"
            self.multipliers = multipliers
        def _block_kwargs(level):
            this_block_kwargs = dict(block_kwargs)
            this_block_kwargs["width"] *= self.multipliers[level]
            this_block_kwargs["depth"] *= self.multipliers[level]
            return this_block_kwargs

        encoder = lambda level: Encoder(x_channels, emb_width, level + 1,
                                        downs_t[:level+1], strides_t[:level+1], **_block_kwargs(level))
        decoder = lambda level: Decoder(x_channels,
                                        emb_width,
                                        level + 1,
                                        downs_t[:level+1],
                                        strides_t[:level+1],
                                        noise_layer_idx,
                                        **_block_kwargs(level))
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for level in range(levels):
            self.encoders.append(encoder(level))
            self.decoders.append(decoder(level))

        # print(self.encoders)
        # print(self.decoders)

        if use_bottleneck:
            self.bottleneck = Bottleneck(l_bins, emb_width, mu, levels)
        else:
            self.bottleneck = NoBottleneck(levels)

        self.downs_t = downs_t
        self.strides_t = strides_t
        self.l_bins = l_bins
        self.commit = commit
        self.spectral = spectral
        self.multispectral = multispectral

    def preprocess(self, x):
        # x: NTC [-1,1] -> NCT [-1,1]
        assert len(x.shape) == 3
        # x = x.permute(0,2,1).float()
        return x

    def postprocess(self, x):
        # x: NTC [-1,1] <- NCT [-1,1]
        # x = x.permute(0,2,1)
        return x

    def _decode(self, zs, start_level=0, end_level=None):
        # Decode
        if end_level is None:
            end_level = self.levels
        assert len(zs) == end_level - start_level
        xs_quantised = self.bottleneck.decode(zs, start_level=start_level, end_level=end_level)
        assert len(xs_quantised) == end_level - start_level

        # Use only lowest level
        decoder, x_quantised = self.decoders[start_level], xs_quantised[0:1]
        x_out = decoder(x_quantised, all_levels=False)
        x_out = self.postprocess(x_out)
        return x_out

    def decode(self, zs, start_level=0, end_level=None, bs_chunks=1):
        """
        :param zs: [torch.Size([batch_size, n_ctx])]
        :param start_level: int
        :param end_level: int
        :param bs_chunks: int
        :return: torch.Size([batch_size, 1, sequence_length]), fe. torch.Size([8, 1, 229312])
        """
        # print(f"start_level {start_level}, end_level {end_level}")
        # print(f"decode len(zs) {len(zs)}, zs[0] {zs[0].shape}")
        # print(f"decode zs {zs}")
        z_chunks = [t.chunk(z, bs_chunks, dim=0) for z in zs]
        x_outs = []
        for i in range(bs_chunks):
            zs_i = [z_chunk[i] for z_chunk in z_chunks]
            x_out = self._decode(zs_i, start_level=start_level, end_level=end_level)
            x_outs.append(x_out)
        # print(f"t.cat(x_outs, dim=0) {t.cat(x_outs, dim=0).shape}")
        return t.cat(x_outs, dim=0)

    def _encode(self, x, start_level=0, end_level=None):
        # Encode
        if end_level is None:
            end_level = self.levels
        x_in = self.preprocess(x)
        xs = []
        for level in range(self.levels):
            encoder = self.encoders[level]
            x_out = encoder(x_in)
            xs.append(x_out[-1])
        zs = self.bottleneck.encode(xs)
        return zs[start_level:end_level]

    def encode(self, x, start_level=0, end_level=None, bs_chunks=1):
        # print(f"start_level {start_level}, end_level {end_level}")
        x_chunks = t.chunk(x, bs_chunks, dim=0)
        zs_list = []
        for x_i in x_chunks:
            zs_i = self._encode(x_i, start_level=start_level, end_level=end_level)
            zs_list.append(zs_i)
        zs = [t.cat(zs_level_list, dim=0) for zs_level_list in zip(*zs_list)]
        return zs

    def sample(self, n_samples):
        zs = [t.randint(0, self.l_bins, size=(n_samples, *z_shape), device='cuda') for z_shape in self.z_shapes]
        return self.decode(zs)

    def forward(self, x, hps, loss_fn='l1'):
        metrics = {}

        N = x.shape[0]

        # Encode/Decode
        # print(f"forward x.shape {x.shape}")
        x_in = self.preprocess(x)

        # print(f"Encoder forward")
        xs = []
        for level in range(self.levels):
            # print(f"vqvae encoder level {level}")
            encoder = self.encoders[level]
            # print(f"level {level} encoder {encoder}")
            x_out = encoder(x_in)
            # print(f"vqvae level {level} x_in.shape, {x_in.shape} -> len(x_out)={len(x_out)}: x_out {x_out[-1].shape}")
            xs.append(x_out[-1])

        # print(f"vector quantisation")
        zs, xs_quantised, commit_losses, quantiser_metrics = self.bottleneck(xs)

        # print(f"Decoder forward")
        x_outs = []
        for level in range(self.levels):
            # print(f"vqvae decoder level {level}")
            decoder = self.decoders[level]
            # print(f"input for decodel layer {level} is {xs_quantised[level:level+1].shape}")
            x_out = decoder(xs_quantised[level:level+1], all_levels=False)

            # print(f"vqvae level {level} xs_quantised {xs_quantised[-1].shape} -> x_out.shape, {x_out.shape}")
            assert_shape(x_out, x_in.shape)
            x_outs.append(x_out)

        # Loss
        """
        When using only the sample-level reconstruction loss, the model learns to reconstruct low frequencies only. 
        To capture mid-to-high frequencies, we add a spectral loss. It encourages the model to match the 
        spectral components without paying attention to phase which is more difficult to learn. 
        """
        def _spectral_loss(x_target, x_out, hps):
            if hps.use_nonrelative_specloss:
                print(f"Using use_nonrelative_specloss instead of spectral convergence")
                sl = spectral_loss(x_target, x_out, hps) / hps.bandwidth['spec']
            else:
                sl = spectral_convergence(x_target, x_out, hps)
            sl = t.mean(sl)

            sl = 10.0 * t.log10(sl)
            return sl

        def _multispectral_loss(x_target, x_out, hps):
            sl = multispectral_loss(x_target, x_out, hps) / hps.bandwidth['spec']

            sl = t.mean(sl)
            return sl

        # SNR(x_target, x_out)

        recons_loss = t.zeros(()).to(x.device)
        spec_loss = t.zeros(()).to(x.device)
        multispec_loss = t.zeros(()).to(x.device)
        x_target = audio_postprocess(x.float(), hps)
        signal_to_noise_ratio = t.zeros(()).to(x.device)
        # comment if you dont want to log PESQ score, should not be used during training
        pesq = t.zeros(()).to(x.device)

        for level in reversed(range(self.levels)):
            x_out = self.postprocess(x_outs[level])
            x_out = audio_postprocess(x_out, hps)
            this_recons_loss = _loss_fn(loss_fn, x_target, x_out, hps)
            this_spec_loss = _spectral_loss(x_target, x_out, hps)
            this_multispec_loss = _multispectral_loss(x_target, x_out, hps)
            this_snr = SNR(x_target[:,0,:], x_out[:,0,:])
            # comment if you dont want to log PESQ score, should not be used during training
            this_pesq = calc_pesq(x_target[:,0,:].detach().cpu().numpy(), x_out[:,0,:].detach().cpu().numpy())
            #metrics[f'recons_loss_l{level + 1}'] = this_recons_loss
            #metrics[f'spectral_loss_l{level + 1}'] = this_spec_loss
            #metrics[f'multispectral_loss_l{level + 1}'] = this_multispec_loss
            #metrics[f'signal_to_noise_ratio_l{level + 1}'] = this_snr
            #metrics[f'pesq_l{level + 1}'] = this_pesq
            recons_loss += this_recons_loss
            spec_loss += this_spec_loss
            multispec_loss += this_multispec_loss
            signal_to_noise_ratio += this_snr
            # comment if you dont want to log PESQ score, should not be used during training
            pesq += this_pesq

        commit_loss = sum(commit_losses)
        # add KL divergence of codebook probabilies compared to uniform transform
        loss = recons_loss + self.spectral * spec_loss + self.multispectral * multispec_loss + self.commit * commit_loss

        with t.no_grad():
            sc = t.mean(spectral_convergence(x_target, x_out, hps))
            l2_loss = _loss_fn("l2", x_target, x_out, hps)
            l1_loss = _loss_fn("l1", x_target, x_out, hps)
            linf_loss = _loss_fn("linf", x_target, x_out, hps)

        quantiser_metrics = average_metrics(quantiser_metrics)

        # comment 'pesq=pesq,' if you dont want to log PESQ score, should not be used during training
        metrics.update(dict(
            recons_loss=recons_loss,
            spectral_loss=spec_loss,
            multispectral_loss=multispec_loss,
            spectral_convergence=sc,
            l2_loss=l2_loss,
            l1_loss=l1_loss,
            linf_loss=linf_loss,
            commit_loss=commit_loss,
            signal_to_noise_ratio=signal_to_noise_ratio,
            pesq=pesq,
            **quantiser_metrics))

        for key, val in metrics.items():
            metrics[key] = val.detach()

        return x_out, loss, metrics
