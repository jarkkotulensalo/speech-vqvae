import numpy as np
import torch as t
import torch.nn as nn
import jukebox.utils.dist_adapter as dist

from jukebox.transformer.ops import LayerNorm
from jukebox.prior.autoregressive import ConditionalAutoregressive2D
from jukebox.prior.conditioners import Conditioner, LabelConditioner
from jukebox.data.labels import EmptyLabeller, Labeller

from jukebox.utils.torch_utils import assert_shape
from jukebox.utils.dist_utils import print_once
from jukebox.vqvae.vqvae import calculate_strides

from jukebox.utils.audio_utils import spectral_convergence, spectral_loss, multispectral_loss
from jukebox.vqvae.vqvae import _loss_fn

"""
Model the prior on vq codes conditioned on timing, artist, genre, lyrics and codes from levels above. 
To condition on the timing, genre and artist, we use the LabelConditioner class
To condition on the codes from the level above, we use the Conditioner class
To condition on lyrics, we allow two types of priors:
- Separate Encoder Decoder: This is the usual encoder-decoder style transformer. The encoder transformer autoregressively 
models the lyrics, and we use its last layer to produce keys/values that are attened to by the decoder transformer
- Single Encoder Decoder: This is a simplification where we combine them into a single model. We merge the text vocab 
and VQ vocab into a single large vocab, and the lyric tokens and VQ tokens into a single longer sequence of tokens which 
we autoregressively model together.
"""

class SimplePrior(nn.Module):
    def __init__(self, z_shapes, l_bins, encoder, decoder, level,
                 downs_t, strides_t, labels, prior_kwargs, x_cond_kwargs, y_cond_kwargs,
                 prime_kwargs, copy_input, labels_v3=False,
                 merged_decoder=False, single_enc_dec=False):
        super().__init__()

        self.use_tokens = prime_kwargs.pop('use_tokens')
        self.n_tokens = prime_kwargs.pop('n_tokens')
        self.prime_loss_fraction = prime_kwargs.pop('prime_loss_fraction')
        print(f"prime_loss_fraction {self.prime_loss_fraction}")

        self.copy_input = copy_input
        if self.copy_input:
            prime_kwargs['bins'] = l_bins

        self.z_shapes = z_shapes
        self.levels = len(self.z_shapes)
        print(f"self.levels {self.levels}")

        self.z_shape = self.z_shapes[level]

        self.level = level
        assert level < self.levels, f"Total levels {self.levels}, got level {level}"

        self.l_bins = l_bins

        # Passing functions instead of the vqvae module to avoid getting params
        self.encoder = encoder
        self.decoder = decoder

        # X conditioning
        self.x_cond = (level != (self.levels - 1))
        self.cond_level = level + 1

        # Y conditioning
        self.y_cond = labels
        self.y_cond = False
        print(f"SimplePrior self.y_cond = labels, force set self.y_cond=False")

        self.single_enc_dec = single_enc_dec
        # X conditioning
        """
        if self.x_cond:
            self.conditioner_blocks = nn.ModuleList()
            conditioner_block = lambda _level: Conditioner(input_shape=z_shapes[_level],
                                                          bins=l_bins,
                                                          down_t=downs_t[_level],
                                                          stride_t=strides_t[_level],
                                                          **x_cond_kwargs)
            if dist.get_rank() == 0: print(f"Conditioning on 1 above level(s)")
            self.conditioner_blocks.append(conditioner_block(self.cond_level))
        """
        # Y conditioning
        # if self.y_cond:
            # self.n_time = self.z_shape[0] # Assuming STFT=TF order and raw=T1 order, so T is first dim
        self.n_time = self.z_shape[0]  # (400, )
        print(f"y_cond_kwargs {y_cond_kwargs}")
        # self.y_emb = LabelConditioner(n_time=self.n_time,include_time_signal=not self.x_cond,**y_cond_kwargs)
        # print(f"prior.py LabelConditioner set include_time_signal=False. Check later.")
        self.y_emb = LabelConditioner(n_time=self.n_time, include_time_signal=False, **y_cond_kwargs)
        # Lyric conditioning
        if single_enc_dec:
            print(f"Prior with single_enc_dec")
            # Single encoder-decoder transformer
            self.prior_shapes = [(self.n_tokens,), prior_kwargs.pop('input_shape')]
            self.prior_bins = [prime_kwargs['bins'], prior_kwargs.pop('bins')]
            self.prior_dims = [np.prod(shape) for shape in self.prior_shapes]
            self.prior_bins_shift = np.cumsum([0, *self.prior_bins])[:-1]
            self.prior_width = prior_kwargs['width']
            print_once(f'Creating cond. autoregress with prior bins {self.prior_bins}, ')
            print_once(f'dims {self.prior_dims}, ')
            print_once(f'shift {self.prior_bins_shift}')
            print_once(f'input shape {sum(self.prior_dims)}')
            print_once(f'input bins {sum(self.prior_bins)}')
            print_once(f'Self copy is {self.copy_input}')

            self.prime_loss_dims, self.gen_loss_dims = self.prior_dims[0], self.prior_dims[1]
            self.total_loss_dims = self.prime_loss_dims + self.gen_loss_dims
            self.prior = ConditionalAutoregressive2D(input_shape=(sum(self.prior_dims),),
                                                     bins=sum(self.prior_bins),
                                                     x_cond=(self.x_cond or self.y_cond), y_cond=True,
                                                     prime_len=self.prime_loss_dims,
                                                     **prior_kwargs)

        else:
            # Separate encoder-decoder transformer
            print(f"Prior with Separate encoder-decoder transformer")
            if self.n_tokens != 0 and self.use_tokens:
                from jukebox.transformer.ops import Conv1D
                prime_input_shape = (self.n_tokens,)
                self.prime_loss_dims = np.prod(prime_input_shape)
                self.prime_acts_width, self.prime_state_width = prime_kwargs['width'], prior_kwargs['width']
                print(f"prime_input_shape {prime_input_shape}")
                self.prime_prior = ConditionalAutoregressive2D(input_shape=prime_input_shape, x_cond=False, y_cond=False,
                                                               only_encode=True,
                                                               **prime_kwargs)
                self.prime_state_proj = Conv1D(self.prime_acts_width, self.prime_state_width, init_scale=prime_kwargs['init_scale'])
                self.prime_state_ln = LayerNorm(self.prime_state_width)
                self.prime_bins = prime_kwargs['bins']
                print(f"self.prime_prior self.prime_bins {self.prime_bins}")
                self.prime_x_out = nn.Linear(self.prime_state_width, self.prime_bins, bias=False)
                nn.init.normal_(self.prime_x_out.weight, std=0.02 * prior_kwargs['init_scale'])
            else:
                self.prime_loss_dims = 0
            self.gen_loss_dims = np.prod(self.z_shape)
            self.total_loss_dims = self.prime_loss_dims + self.gen_loss_dims
            print(f"x_cond {(self.x_cond or self.y_cond)}, y_cond {self.y_cond}")
            #self.prior = ConditionalAutoregressive2D(x_cond=(self.x_cond or self.y_cond), y_cond=self.y_cond,
            #                                        encoder_dims = self.prime_loss_dims, merged_decoder=merged_decoder,
            #                                         **prior_kwargs)
            self.prior = ConditionalAutoregressive2D(x_cond=(self.x_cond or self.y_cond), y_cond=self.y_cond,
                                                     encoder_dims=self.prime_loss_dims, merged_decoder=merged_decoder,
                                                     **prior_kwargs)

        self.n_ctx = self.gen_loss_dims
        self.downsamples = calculate_strides(strides_t, downs_t)
        self.cond_downsample = self.downsamples[level+1] if level != self.levels - 1 else None
        self.raw_to_tokens = np.prod(self.downsamples[:level+1])
        self.sample_length = self.n_ctx*self.raw_to_tokens

        if labels:
            print(f"labels=True, setting Labeller.")
            self.labels_v3 = labels_v3
            self.labeller = Labeller(n_tokens=self.n_tokens, sample_length=self.sample_length, v3=self.labels_v3)
        else:
            print(f"EmptyLabeller")
            self.labeller = EmptyLabeller()

        print(f"Level:{level}, Cond downsample:{self.cond_downsample}, Raw to tokens:{self.raw_to_tokens}, Sample length:{self.sample_length}, self.n_ctx {self.n_ctx}")


    def get_y(self, labels, start, get_indices=False):
        if isinstance(self.labeller, EmptyLabeller):
            print(f"prior.py get_y self.labeller is EmptyLabeller, returning None.")
            return None
        y = labels['y'].clone()

        # Set sample_length to match this level
        # y[:, 2] = int(self.sample_length)

        # Set offset
        # y[:, 1:2] = y[:, 1:2] + int(start * self.raw_to_tokens)

        # Set lyric tokens
        # print(f"prior.py get_y y {y.shape} get_indices {get_indices}")
        if get_indices:
            return y, y
            #indices = self.labeller.set_y_lyric_tokens(y, labels)
            #print(f"prior.py get_y indices {indices}")
            #return y, indices
        else:
            return y

    def get_z_conds(self, zs, start, end):
        if self.level != self.levels - 1:
            assert start % self.cond_downsample == end % self.cond_downsample == 0
            z_cond = zs[self.level + 1][:,start//self.cond_downsample:end//self.cond_downsample]
            assert z_cond.shape[1] == self.n_ctx//self.cond_downsample
            z_conds = [z_cond]
        else:
            z_conds = None
        return z_conds

    def prior_preprocess(self, xs, conds):
        """

        :param xs:
        :param conds:
        :return: z, x_cond
        """
        N = xs[0].shape[0]
        print(f"prior_preprocess N {N}")
        print(f"prior_preprocess len(xs) {len(xs)}")
        for i in range(len(xs)):
            x, shape, dims = xs[i], self.prior_shapes[i], self.prior_dims[i]
            print(f"prior_preprocess x.shape {x.shape}")
            print(f"prior_preprocess shape {shape}")
            print(f"prior_preprocess dims {dims}")
            bins, bins_shift = int(self.prior_bins[i]), int(self.prior_bins_shift[i])
            print(f"prior_preprocess bins {bins}")
            print(f"prior_preprocess bins_shift {bins_shift}")

            assert isinstance(x, t.cuda.LongTensor), x
            assert (0 <= x).all() and (x < bins).all()
            #assert_shape(x, (N, *shape))
            print(f"(xs[i] + bins_shift).shape {(xs[i] + bins_shift).shape}")
            print(f"(xs[i] + bins_shift).view(N, -1).shape {(xs[i] + bins_shift).view(N, -1).shape}")
            xs[i] = (xs[i] + bins_shift).view(N, -1)

        print(f"prior_preprocess len(conds) {len(conds)}")
        for i in range(len(conds)):
            cond, shape, dims = conds[i], self.prior_shapes[i], self.prior_dims[i]
            if cond is not None:
                assert_shape(cond, (N, dims, self.prior_width))
            else:
                conds[i] = t.zeros((N, dims, self.prior_width), dtype=t.float, device='cuda')

        print(f"t.cat(xs, dim=1) {t.cat(xs, dim=1).shape}")
        print(f"t.cat(conds, dim=1) {t.cat(conds, dim=1).shape}")

        return t.cat(xs, dim=1), t.cat(conds, dim=1)

    def prior_postprocess(self, z):
        N = z.shape[0]
        dims = (self.prior_dims[0], z.shape[1] - self.prior_dims[0])
        # xs = list(t.split(z, self.prior_dims, dim=1))
        xs = list(t.split(z, dims, dim=1))

        for i in range(len(xs)):
            # x, shape, dims, bins, bins_shift = xs[i], self.prior_shapes[i], self.prior_dims[i], self.prior_bins[i], self.prior_bins_shift[i]
            # assert_shape(x, (N, dims))
            shape = self.prior_shapes[i]
            bins, bins_shift = int(self.prior_bins[i]), int(self.prior_bins_shift[i])
            # xs[i] = (xs[i] - bins_shift).view(N, *shape) #view(N, -1, *shape[1:])
            xs[i] = (xs[i] - bins_shift).view(N, -1, *shape[1:])
            xs[i] = t.clamp(xs[i], min=0)  # If not masking loss, model may have generated lyric/midi tokens which are now shifted <0 by bin_shift
            assert (xs[i] < bins).all(), f'rank: {dist.get_rank()}, bins: {bins}, dims {dims}, shape {shape}, prior_shape {self.prior_shapes}, bins_shift {bins_shift}, xs[i]: {xs[i]}'

        return xs[-1]

    """
    def x_emb(self, z_conds):
        z_conds = z_conds[:self.cond_level - self.level]
        assert len(z_conds) == len(self.conditioner_blocks) == self.cond_level - self.level, f"Expected {len(z_conds)} == {len(self.conditioner_blocks)} == {self.cond_level} - {self.level}"
        x_cond = None
        for z_cond, conditioner_block in reversed(list(zip(z_conds, self.conditioner_blocks))):
            x_cond = conditioner_block(z_cond, x_cond)
        return x_cond
    """

    def encode(self, x, start_level=None, end_level=None, bs_chunks=1):
        if start_level == None:
            start_level = self.level
        if end_level == None:
            end_level = self.levels
        # Get latents
        with t.no_grad():
            zs = self.encoder(x, start_level=start_level, end_level=end_level, bs_chunks=bs_chunks)
        return zs

    def decode(self, zs, start_level=None, end_level=None, bs_chunks=1):
        """

        :param zs: list of torch.tensors, [torch.LongTensor(batch_size, 1, code_length)]
        :param start_level: int
        :param end_level: int
        :param bs_chunks: int
        :return: list of torch.tensors
        """
        # print(f"len(zs) {len(zs)} == end_level {end_level} - start_level {start_level}")
        if start_level == None:
            start_level = self.level
        if end_level == None:
            end_level = self.levels
            #print(f"decode zs {len(zs)}")
            # print(f"decode zs[0] {zs[0].shape}")
        #print(f"decode len(zs) {len(zs)} == end_level {end_level} - start_level {start_level}")
        assert len(zs) == end_level - start_level
        with t.no_grad():
            x_out = self.decoder(zs, start_level=start_level, end_level=end_level, bs_chunks=bs_chunks)
        return x_out

    def get_cond(self, y):
        if y is not None:
            # n_labels = y.shape[1] - self.n_tokens
            # y, prime = y[:, :n_labels], y[:, n_labels:]
            y_cond = None
            prime = y
        else:
            y, prime = None, None
        x_cond = None
        # print(f"prior get_cond x_cond {x_cond}, y_cond {y_cond.shape} prime {prime.shape}")

        return x_cond, y_cond, prime

    def sample(self, n_samples, z=None, z_conds=None, y=None, fp16=False, temp=1.0, top_k=0, top_p=0.0,
               chunk_size=None, sample_tokens=None):
        N = n_samples
        if z is not None: assert z.shape[0] == N, f"Expected shape ({N},**), got shape {z.shape}"
        if y is not None: assert y.shape[0] == N, f"Expected shape ({N},**), got shape {y.shape}"
        if z_conds is not None:
            for z_cond in z_conds:
                assert z_cond.shape[0] == N,  f"Expected shape ({N},**), got shape {z_cond.shape}"

        no_past_context = (z is None or z.shape[1] == 0)
        if dist.get_rank() == 0:
            name = {True: 'Ancestral', False: 'Primed'}[no_past_context]
            print(f"{name} sampling {n_samples} samples with temp={temp}, top_k={top_k}, top_p={top_p}")

        with t.no_grad():
            x_cond, y_cond, prime = self.get_cond(y)
            # print(f"prior sample y_cond {y_cond}, prime {prime.shape}")

            if self.single_enc_dec:
                # assert chunk_size % self.prime_loss_dims == 0. TODO: Check if needed
                if no_past_context:
                    z, x_cond = self.prior_preprocess([prime], [None, x_cond])
                else:
                    z, x_cond = self.prior_preprocess([prime, z], [None, x_cond])
                if sample_tokens is not None:
                    sample_tokens += self.n_tokens
                z = self.prior.primed_sample(n_samples, z, x_cond, y_cond, fp16=fp16, temp=temp,
                                             top_k=top_k, top_p=top_p, chunk_size=chunk_size, sample_tokens=sample_tokens)
                z = self.prior_postprocess(z)

            else:
                encoder_kv = self.get_encoder_kv(prime, fp16=fp16, sample=True)
                
                if no_past_context:
                    print(f"prior sample with no_past_context")
                    z = self.prior.sample(n_samples, x_cond, y_cond, encoder_kv, fp16=fp16, temp=temp, top_k=top_k,
                                          top_p=top_p, sample_tokens=sample_tokens)
                else:
                    print(f"prior sample with past_context")
                    z = self.prior.primed_sample(n_samples, z, x_cond, y_cond, encoder_kv, fp16=fp16, temp=temp,
                                             top_k=top_k, top_p=top_p, chunk_size=chunk_size, sample_tokens=sample_tokens)
            if sample_tokens is None:
                assert_shape(z, (N, *self.z_shape))

        return z

    def get_encoder_kv(self, prime, fp16=False, sample=False):

        # print(f"get_encoder_kv self.n_tokens {self.n_tokens}, self.use_tokens {self.use_tokens}")
        if self.n_tokens != 0 and self.use_tokens:

            # if sample:
                # self.prime_prior.cuda()
            N = prime.shape[0]
            prime_acts = self.prime_prior(prime, None, None, None, fp16=fp16)  # torch.Size([16, 188, 128])
            # print(f"get_encoder_kv prime_acts {prime_acts.shape}")
            assert_shape(prime_acts, (N, self.prime_loss_dims, self.prime_acts_width))
            assert prime_acts.dtype == t.float, f'Expected t.float, got {prime_acts.dtype}'
            encoder_kv = self.prime_state_ln(self.prime_state_proj(prime_acts))  # torch.Size([16, 188, 1024])
            # print(f"get_encoder_kv encoder_kv {encoder_kv.shape}")
            assert encoder_kv.dtype == t.float, f'Expected t.float, got {encoder_kv.dtype}'
            if sample:
                # self.prime_prior.cpu()  # this induces CUDA error
                if fp16:
                    encoder_kv = encoder_kv.half()
        else:
            encoder_kv = None
        # print(f"prior.py get_encoder_kv returning encoder_kv keys and values {encoder_kv.shape}")
        return encoder_kv

    def get_prime_loss(self, encoder_kv, prime_t):
        # z = torch.LongTensor(batch_size, 7166)
        # prime_t = torch.LongTensor(batch_size, 2000)
        # gen_t = torch.LongTensor(batch_size, 5166)
        if self.use_tokens:
            encoder_kv = encoder_kv.float()
            encoder_kv = self.prime_x_out(encoder_kv)
            prime_loss = nn.functional.cross_entropy(encoder_kv.view(-1, self.prime_bins), prime_t.view(-1)) / np.log(2.)
        else:
            prime_loss = t.tensor(0.0, device='cuda')
        return prime_loss

    def z_forward(self, z, z_conds=[], y=None, fp16=False, get_preds=False, get_attn_weights=False, hps=None):
        """
        Arguments:
            get_attn_weights (bool or set): Makes forward prop dump
                self-attention softmaxes to self.prior.transformer.ws. Either a
                set of layer indices indicating which layers to store, or a
                boolean value indicating whether to dump all.
        """
        # print(f"z_forward z {z.shape}")
        # print(f"get_attn_weights {get_attn_weights}")
        assert isinstance(get_attn_weights, (bool, set))
        if get_attn_weights:
            self.prior.transformer.set_record_attn(get_attn_weights)
        # print(f"z_forward z.shape {z.shape}")
        # print(f"z_forward z_conds {z_conds}")
        # print(f"z_forward y.shape {y.shape}")

        x_cond, y_cond, prime = self.get_cond(y)
        # print(f"z_forward x_cond {x_cond}")
        # print(f"z_forward y_cond {y_cond}")
        # print(f"z_forward y {y.shape}, prime {prime.shape}")  y torch.Size([16, 188]), prime torch.Size([16, 188])

        if self.copy_input:
            prime = z[:, :self.n_tokens]
            print(f"self.copy_input prime values {prime.shape}")
        if self.single_enc_dec:
            z, x_cond = self.prior_preprocess([prime, z], [None, x_cond])

            (prime_loss, gen_loss), preds = self.prior(z, x_cond, y_cond, fp16=fp16, get_sep_loss=True, get_preds=get_preds)
        else:
            # print(f"z_forward get_encoder_kv")
            encoder_kv = self.get_encoder_kv(prime, fp16=fp16)
            # print(f"z_forward encoder_kv {encoder_kv.shape}")

            # print(f"z_forward get_prime_loss")
            prime_loss = self.get_prime_loss(encoder_kv, prime)

            #
            # print(f"z_forward self.prior")
            # gen_loss, preds = self.prior(z, x_cond, y_cond, encoder_kv, fp16=fp16, get_preds=get_preds)
            gen_loss, preds = self.prior(z, x_cond, y_cond, encoder_kv, fp16=fp16, get_preds=True)
            # print(f"prior.py z_forward get_preds={get_preds}, preds {preds}, z {z.shape}")

        # Prime loss is something between Transformer keys-values and sentence.

        ############## ADDED MULTISPECTRAL LOSS
        # https://arxiv.org/pdf/1910.11480.pdf
        # print(f"z {z.shape}, preds {preds.shape}") z torch.Size([32, 896]), preds torch.Size([32, 896, 512])
        x_target = self.decode([z, *z_conds])
        # print(f"x_target {x_target}")
        z_preds = t.argmax(preds, dim=2)
        x_pred = self.decode([z_preds, *z_conds])

        # print(f"x_target {x_target.shape}, x_pred {x_pred.shape}") x_target torch.Size([32, 1, 114688]), x_pred torch.Size([32, 1, 114688])
        def _multispectral_loss(x_target, x_out, hps):
            sl = multispectral_loss(x_target, x_out, hps) / hps.bandwidth['spec']
            sl = t.mean(sl)
            return sl

        def _spectral_loss(x_target, x_out, hps):
            #if hps.use_nonrelative_specloss:
            #    sl = spectral_loss(x_target, x_out, hps) / hps.bandwidth['spec']
            #else:
            sl = spectral_convergence(x_target, x_out, hps)
            sl = t.mean(sl)
            return sl

        recons_loss = _loss_fn('l2', x_target, x_pred, hps)
        spec_loss = _spectral_loss(x_target, x_pred, hps)
        multispec_loss = _multispectral_loss(x_target, x_pred, hps)
        vqvae_loss = recons_loss + hps.multispectral * multispec_loss
        # print(f"vqvae_loss {vqvae_loss}, recons_loss {recons_loss} + multispec_loss {multispec_loss}")
        ############## ADDED MULTISPECTRAL LOSS

        """"
        The lyrics encoder is a Transformer with an autoregressive modeling loss for lyrics, and its last level is
        used as features of the lyrics.
        """

        loss = (self.prime_loss_fraction*prime_loss*self.prime_loss_dims/self.total_loss_dims) + \
                   (gen_loss*self.gen_loss_dims/self.total_loss_dims) # + \
                    # (hps.multispectral_loss_fraction*multispec_loss)

        metrics=dict(bpd=gen_loss.clone().detach(),
                     prime_loss=prime_loss.clone().detach(),
                     gen_loss=gen_loss.clone().detach(),
                     recons_loss=recons_loss,
                     multispec_loss=multispec_loss,
                     vqvae_loss=vqvae_loss,
                     spectral_loss=spec_loss
                     )
        if get_preds:
            metrics["preds"] = preds.clone().detach()
        if get_attn_weights:
            ws = self.prior.transformer.ws
            self.prior.transformer.set_record_attn(False)
            return ws
        else:
            # return loss, metrics
            return loss, metrics

    def forward(self, x, y=None, fp16=False, decode=False, get_preds=False, hps=None):
        """
        Prior forward.
        Changed the forward to work with pre-encoded waveforms. Input x for this function is latent z.
        :param x: torch.LongTensor(batch size, n_ctx), f.e. torch.Size([16, 7166])
        :param y: torch (batch size, sequence length)
        :param fp16: boolean
        :param decode: boolean
        :param get_preds: boolean
        :return: x_out, loss, metrics
        """
        # print(f"prior.py forward starting")
        # bs = x.shape[0]
        # print(f"prior forward x {x.shape} {x.type()}")  # torch.Size([16, 1, 229320])
        # z, *z_conds = self.encode(x, bs_chunks=bs)  # (torch.Size([16, 7166]),  [])
        # print(f"prior forward z {z.shape}, z_conds {z_conds}")
        z = x
        z_conds = []
        # print(f"prior forward new z {z.shape}, {z.type()}, z_conds {z_conds}")
        # loss, metrics = self.z_forward(z=z, z_conds=z_conds, y=y, fp16=fp16, get_preds=get_preds)
        loss, metrics = self.z_forward(z=z, z_conds=z_conds, y=y, fp16=fp16, get_preds=get_preds, hps=hps)
        if decode:
            # x_out = self.decode([z, *z_conds])
            # print(f"prior decode z {z.type()}, {z.shape}")
            x_out = self.decode([z, *z_conds])

        else:
            x_out = None
        return x_out, loss, metrics
