import math
import torch
import torch.nn as nn
import jukebox.utils.dist_adapter as dist
from jukebox.utils.checkpoint import checkpoint


class NoiseInjectionImages(nn.Module):
    """Additive noise with per-channel scaling weights set initially to zero."""
    def __init__(self, channel):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channel, 1, 1))

    def forward(self, image, noise):
        return image + self.weight * noise


class NoiseInjection(nn.Module):
    """Additive noise with per-channel scaling weights set initially to zero."""
    def __init__(self, inject=True):
        super().__init__()
        self.noise_channels = 1
        self.inject = inject

    def forward(self, x):
        if self.noise_channels > 0 and self.inject:
            z = torch.randn(x.size(0), self.noise_channels, x.size(2), device='cuda')
            x = torch.cat([x, z], dim=1)
        return x

"""
https://version.aalto.fi/gitlab/ljkarkk2/deeplearn/-/blob/06c6dfd4e0cc28a382af789e857527485ad5ab9b/stylegan.py
self.noise1 = equal_lr(NoiseInjection(out_channel))
# Random z for generator
gen_in1 = torch.randn(b_size, params.code_size, device=device)

https://discuss.pytorch.org/t/add-gaussian-noise-to-parameters-while-training/109260

self.noise_conv2 = torch.randn(self.conv1.weight.size())*0.6 + 0  

add_noise(self.conv2.weight, self.noise_conv2)

def add_noise(weights, noise):                                                                                                                                                                                                                                              
    with torch.no_grad():                                                                                                                                                                                                                                                   
        weights.add_(noise)  
"""


class ResConvBlock(nn.Module):
    def __init__(self, n_in, n_state):
        super().__init__()
        self.model = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(n_in, n_state, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(n_state, n_in, 1, 1, 0),
        )

    def forward(self, x):
        return x + self.model(x)

class Resnet(nn.Module):
    def __init__(self, n_in, n_depth, m_conv=1.0):
        super().__init__()
        self.model = nn.Sequential(*[ResConvBlock(n_in, int(m_conv * n_in)) for _ in range(n_depth)])

    def forward(self, x):
        return self.model(x)

class ResConv1DBlock(nn.Module):
    def __init__(self, n_in, n_state, dilation=1, zero_out=False, res_scale=1.0, noise_injection=False):
        super().__init__()
        padding = dilation
        # torch.randn
        # slice to one channel
        # concatenate to inputs
        # gen_in1 = torch.randn(b_size, params.code_size, device=device)

        if noise_injection:
            self.model = nn.Sequential(
                NoiseInjection(),
                nn.ReLU(),
                nn.Conv1d(n_in + 1, n_state, 3, 1, padding, dilation),
                nn.ReLU(),
                nn.Conv1d(n_state, n_in, 1, 1, 0),
            )
        else:
            self.model = nn.Sequential(
                nn.ReLU(),
                nn.Conv1d(n_in, n_state, 3, 1, padding, dilation),
                nn.ReLU(),
                nn.Conv1d(n_state, n_in, 1, 1, 0),
            )

        if zero_out:
            out = self.model[-1]
            nn.init.zeros_(out.weight)
            nn.init.zeros_(out.bias)
        self.res_scale = res_scale

    def forward(self, x):
        return x + self.res_scale * self.model(x)


class ResConv1DGatedBlock(nn.Module):
    def __init__(self, n_in, n_state, dilation=1, zero_out=False, res_scale=1.0):
        super().__init__()
        padding = dilation
        self.n_state = n_state
        self.conv_gates = nn.Conv1d(n_in, 2*n_state, 3, 1, padding, dilation)
        self.conv_out = nn.Conv1d(n_state, n_in, 1, 1, 0)
        if zero_out:
            out = self.model[-1]
            nn.init.zeros_(out.weight)
            nn.init.zeros_(out.bias)
        self.res_scale = res_scale

    def forward(self, x):
        y = torch.tanh(x)
        y = self.conv_gates(y)
        y = torch.sigmoid(y[:, :self.n_state, :]) * torch.tanh(y[:, self.n_state:, :])
        y = self.conv_out(y)
        return x + self.res_scale * y


class Resnet1D(nn.Module):
    def __init__(self, n_in, n_depth, m_conv=1.0, dilation_growth_rate=1, dilation_cycle=None, zero_out=False, res_scale=False, reverse_dilation=False, checkpoint_res=False, noise_injection=False):
        super().__init__()
        def _get_depth(depth):
            if dilation_cycle is None:
                return depth
            else:
                return depth % dilation_cycle
        blocks = [ResConv1DBlock(n_in=n_in,
                                 n_state=int(m_conv * n_in),
                                 dilation=dilation_growth_rate ** _get_depth(depth),
                                 zero_out=zero_out,
                                 res_scale=1.0 if not res_scale else 1.0 / math.sqrt(n_depth),
                                 noise_injection=noise_injection if depth==(n_depth-1) else False)
                  for depth in range(n_depth)]
        if reverse_dilation:
            blocks = blocks[::-1]
        self.checkpoint_res = checkpoint_res
        if self.checkpoint_res == 1:
            if dist.get_rank() == 0:
                print("Checkpointing convs")
            self.blocks = nn.ModuleList(blocks)
        else:
            self.model = nn.Sequential(*blocks)

    def forward(self, x):
        if self.checkpoint_res == 1:
            for block in self.blocks:
                x = checkpoint(block, (x, ), block.parameters(), True)
            return x
        else:
            return self.model(x)


"""
class Resnet1D(nn.Module):
    def __init__(self, n_in, n_depth, m_conv=1.0, dilation_growth_rate=1, dilation_cycle=None, zero_out=False, res_scale=False, reverse_dilation=False, checkpoint_res=False):
        super().__init__()
        def _get_depth(depth):
            if dilation_cycle is None:
                return depth
            else:
                return depth % dilation_cycle
        blocks = [ResConv1DGatedBlock(n_in, int(m_conv * n_in),
                                 dilation=dilation_growth_rate ** _get_depth(depth),
                                 zero_out=zero_out,
                                 res_scale=1.0 if not res_scale else 1.0 / math.sqrt(n_depth))
                  for depth in range(n_depth)]
        if reverse_dilation:
            blocks = blocks[::-1]
        self.checkpoint_res = checkpoint_res
        if self.checkpoint_res == 1:
            if dist.get_rank() == 0:
                print("Checkpointing convs")
            self.blocks = nn.ModuleList(blocks)
        else:
            self.model = nn.Sequential(*blocks)

    def forward(self, x):
        if self.checkpoint_res == 1:
            for block in self.blocks:
                x = checkpoint(block, (x, ), block.parameters(), True)
            return x
        else:
            return self.model(x)
"""