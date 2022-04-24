import torch as t
import torch.nn as nn
from jukebox.vqvae.resnet import Resnet, Resnet1D, NoiseInjection
from jukebox.utils.torch_utils import assert_shape


class EncoderConvBlock(nn.Module):
    def __init__(self, input_emb_width, output_emb_width, down_t,
                 stride_t, width, depth, m_conv,
                 dilation_growth_rate=1, dilation_cycle=None, zero_out=False,
                 res_scale=False):
        super().__init__()
        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        if down_t > 0:
            for i in range(down_t):
                block = nn.Sequential(
                    nn.Conv1d(input_emb_width if i == 0 else width, width, filter_t, stride_t, pad_t),
                    Resnet1D(width, depth, m_conv, dilation_growth_rate, dilation_cycle, zero_out, res_scale),
                )
                blocks.append(block)
            block = nn.Conv1d(width, output_emb_width, 3, 1, 1)
            blocks.append(block)
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)

class DecoderConvBock(nn.Module):
    def __init__(self, input_emb_width, output_emb_width, down_t,
                 stride_t, width, depth, m_conv, dilation_growth_rate=1, dilation_cycle=None, zero_out=False,
                 res_scale=False, reverse_decoder_dilation=False, checkpoint_res=False, noise_layer_idx=0):
        super().__init__()
        blocks = []
        if down_t > 0:
            filter_t, pad_t = stride_t * 2, stride_t // 2
            block = nn.Conv1d(in_channels=output_emb_width,
                              out_channels=width,
                              kernel_size=3,
                              stride=1,
                              padding=1)
            blocks.append(block)
            for i in range(down_t):
                if i == (noise_layer_idx):
                    noise_injection = True
                else:
                    noise_injection = False
                # print(f"DecoderConvBock i={i} noise_injection={noise_injection}, noise_layer_idx = {noise_layer_idx}")
                block = nn.Sequential(
                    Resnet1D(n_in=width,
                             n_depth=depth,
                             m_conv=m_conv,
                             dilation_growth_rate=dilation_growth_rate,
                             dilation_cycle=dilation_cycle,
                             zero_out=zero_out,
                             res_scale=res_scale,
                             reverse_dilation=reverse_decoder_dilation,
                             checkpoint_res=checkpoint_res,
                             noise_injection=noise_injection),
                    nn.ConvTranspose1d(width, input_emb_width if i == (down_t - 1) else width, filter_t, stride_t, pad_t)
                    )
                """
                block = nn.Sequential(
                    Resnet1D(n_in=width,
                             n_depth=depth,
                             m_conv=m_conv,
                             dilation_growth_rate=dilation_growth_rate,
                             dilation_cycle=dilation_cycle,
                             zero_out=zero_out,
                             res_scale=res_scale,
                             reverse_dilation=reverse_decoder_dilation,
                             checkpoint_res=checkpoint_res),
                    NoiseInjection(noise_channels, True),
                    nn.ConvTranspose1d(width + noise_channels, input_emb_width if i == (down_t - 1) else width, filter_t, stride_t, pad_t)
                )
                """
                blocks.append(block)
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)

class Encoder(nn.Module):
    def __init__(self, input_emb_width, output_emb_width, levels, downs_t,
                 strides_t, **block_kwargs):
        super().__init__()
        self.input_emb_width = input_emb_width
        self.output_emb_width = output_emb_width
        self.levels = levels
        self.downs_t = downs_t
        self.strides_t = strides_t

        block_kwargs_copy = dict(**block_kwargs)
        if 'reverse_decoder_dilation' in block_kwargs_copy:
            del block_kwargs_copy['reverse_decoder_dilation']
        level_block = lambda level, down_t, stride_t: EncoderConvBlock(input_emb_width if level == 0 else output_emb_width,
                                                           output_emb_width,
                                                           down_t, stride_t,
                                                           **block_kwargs_copy)
        self.level_blocks = nn.ModuleList()
        iterator = zip(list(range(self.levels)), downs_t, strides_t)
        for level, down_t, stride_t in iterator:
            self.level_blocks.append(level_block(level, down_t, stride_t))

    def forward(self, x):
        N, T = x.shape[0], x.shape[-1]
        emb = self.input_emb_width
        assert_shape(x, (N, emb, T))
        xs = []

        # print(f"Input to encdec encoder x.shape {x.shape}")
        # 64, 32, ...
        iterator = zip(list(range(self.levels)), self.downs_t, self.strides_t)
        for level, down_t, stride_t in iterator:
            level_block = self.level_blocks[level]
            x = level_block(x)
            emb, T = self.output_emb_width, T // (stride_t ** down_t)
            assert_shape(x, (N, emb, T))
            # print(f"encdec encoder level {level} x.shape {x.shape}")
            xs.append(x)

        return xs

class Decoder(nn.Module):
    def __init__(self, input_emb_width, output_emb_width, levels, downs_t,
                 strides_t, noise_layer_idx, **block_kwargs):
        super().__init__()
        self.input_emb_width = input_emb_width
        self.output_emb_width = output_emb_width
        self.levels = levels
        self.downs_t = downs_t
        self.strides_t = strides_t
        self.noise_layer_idx = noise_layer_idx

        # self.noise_injection1 = NoiseInjection(noise_channels)
        level_block = lambda level, down_t, stride_t: DecoderConvBock(output_emb_width,
                                                                      output_emb_width,
                                                                      down_t,
                                                                      stride_t,
                                                                      noise_layer_idx=noise_layer_idx,
                                                                      **block_kwargs)
        """
        level_block = lambda level, down_t, stride_t: DecoderConvBock(output_emb_width,
                                                          output_emb_width,
                                                          down_t, stride_t,
                                                          **block_kwargs)
        """
        self.level_blocks = nn.ModuleList()
        iterator = zip(list(range(self.levels)), downs_t, strides_t)
        for level, down_t, stride_t in iterator:
            self.level_blocks.append(level_block(level, down_t, stride_t))

        if noise_layer_idx == 5:
            self.noise_injection2 = NoiseInjection()
            self.out = nn.Conv1d(output_emb_width + 1, input_emb_width, 3, 1, 1)
        # self.out = nn.Conv1d(output_emb_width, input_emb_width, 3, 1, 1)
        # self.noise_injection2 = NoiseInjection(noise_channels)
        else:
            self.out = nn.Conv1d(output_emb_width, input_emb_width, 3, 1, 1)

    def forward(self, xs, all_levels=True):
        if all_levels:
            assert len(xs) == self.levels
        else:
            assert len(xs) == 1
        x = xs[-1]
        N, T = x.shape[0], x.shape[-1]
        emb = self.output_emb_width
        assert_shape(x, (N, emb, T))

        # x = self.noise_injection1(x)
        # print(f"Input to decoder xs.shape {x.shape}")
        # 32, 64 ...
        iterator = reversed(list(zip(list(range(self.levels)), self.downs_t, self.strides_t)))
        for level, down_t, stride_t in iterator:
            level_block = self.level_blocks[level]
            x = level_block(x)
            # print(f"level_block(x) x.shape {x.shape}")
            emb, T = self.output_emb_width, T * (stride_t ** down_t)
            assert_shape(x, (N, emb, T))
            if level != 0 and all_levels:
                x = x + xs[level - 1]
                # print(f" x = x + xs[level - 1] x.shape {x.shape}")

            # print(f"encdec decoder level {level} x.shape {x.shape}")
        if self.noise_layer_idx == 5:
            x = self.noise_injection2(x)
        x = self.out(x)
        # print(f"x = self.out(x) x.shape {x.shape}")
        return x
