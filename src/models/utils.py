import torch
import torch.nn.functional as F
from audiolm_pytorch.version import __version__
from einops import rearrange
from packaging import version
from torch import nn
from torch.autograd import grad as torch_grad
from torch.linalg import vector_norm

parsed_version = version.parse(__version__)


# helper functions

def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def cast_tuple(t, l=1):
    return ((t,) * l) if not isinstance(t, tuple) else t


def filter_by_keys(fn, d):
    return {k: v for k, v in d.items() if fn(k)}


def map_keys(fn, d):
    return {fn(k): v for k, v in d.items()}


# gan losses

def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def hinge_discr_loss(fake, real):
    return (F.relu(1 + fake) + F.relu(1 - real)).mean()


def hinge_gen_loss(fake):
    # return -fake.mean()
    return (F.relu(1 - fake)).mean()


def leaky_relu(p=0.1):
    return nn.LeakyReLU(p)


def gradient_penalty(wave, output, weight=10):
    batch_size, device = wave.shape[0], wave.device

    gradients = torch_grad(
        outputs=output,
        inputs=wave,
        grad_outputs=torch.ones_like(output),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = rearrange(gradients, 'b ... -> b (...)')
    return weight * ((vector_norm(gradients, dim=1) - 1) ** 2).mean()


# better sequential

def Sequential(*mods):
    return nn.Sequential(*filter(exists, mods))


class SqueezeExcite(nn.Module):
    def __init__(self, dim, reduction_factor=4, dim_minimum=8):
        super().__init__()
        dim_inner = max(dim_minimum, dim // reduction_factor)
        self.net = nn.Sequential(
            nn.Conv1d(dim, dim_inner, 1),
            nn.SiLU(),
            nn.Conv1d(dim_inner, dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        seq, device = x.shape[-2], x.device

        # cumulative mean - since it is autoregressive

        cum_sum = x.cumsum(dim=-2)
        denom = torch.arange(1, seq + 1, device=device).float()
        cum_mean = cum_sum / rearrange(denom, 'n -> n 1')

        # glu gate

        gate = self.net(cum_mean)

        return x * gate


class ModReLU(nn.Module):
    """
    https://arxiv.org/abs/1705.09792
    https://github.com/pytorch/pytorch/issues/47052#issuecomment-718948801
    """

    def __init__(self):
        super().__init__()
        self.b = nn.Parameter(torch.tensor(0.))

    def forward(self, x):
        return F.relu(torch.abs(x) + self.b) * torch.exp(1.j * torch.angle(x))


class ComplexConv2d(nn.Module):
    def __init__(
            self,
            dim,
            dim_out,
            kernel_size,
            stride=1,
            padding=0
    ):
        super().__init__()
        conv = nn.Conv2d(dim, dim_out, kernel_size, dtype=torch.complex64)
        self.weight = nn.Parameter(torch.view_as_real(conv.weight))
        self.bias = nn.Parameter(torch.view_as_real(conv.bias))

        self.stride = stride
        self.padding = padding

    def forward(self, x):
        weight, bias = map(torch.view_as_complex, (self.weight, self.bias))

        x = x.to(weight.dtype)
        return F.conv2d(x, weight, bias, stride=self.stride, padding=self.padding)


def ComplexSTFTResidualUnit(chan_in, chan_out, strides):
    kernel_sizes = tuple(map(lambda t: t + 2, strides))
    paddings = tuple(map(lambda t: t // 2, kernel_sizes))

    return nn.Sequential(
        Residual(Sequential(
            ComplexConv2d(chan_in, chan_in, 3, padding=1),
            ModReLU(),
            ComplexConv2d(chan_in, chan_in, 3, padding=1)
        )),
        ComplexConv2d(chan_in, chan_out, kernel_sizes, stride=strides, padding=paddings)
    )


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class CausalConv1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, pad_mode='reflect', **kwargs):
        super().__init__()
        kernel_size = kernel_size
        dilation = kwargs.get('dilation', 1)
        stride = kwargs.get('stride', 1)
        self.pad_mode = pad_mode
        self.causal_padding = dilation * (kernel_size - 1) + (1 - stride)

        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, **kwargs)

    def forward(self, x):
        x = F.pad(x, (self.causal_padding, 0), mode=self.pad_mode)
        return self.conv(x)


class CausalConvTranspose1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, stride, **kwargs):
        super().__init__()
        self.upsample_factor = stride
        self.padding = kernel_size - 1
        self.conv = nn.ConvTranspose1d(chan_in, chan_out, kernel_size, stride, **kwargs)

    def forward(self, x):
        n = x.shape[-1]

        out = self.conv(x)
        out = out[..., :(n * self.upsample_factor)]

        return out


def ResidualUnit(chan_in, chan_out, dilation, kernel_size=7, squeeze_excite=False, pad_mode='reflect'):
    return Residual(Sequential(
        CausalConv1d(chan_in, chan_out, kernel_size, dilation=dilation, pad_mode=pad_mode),
        nn.ELU(),
        CausalConv1d(chan_out, chan_out, 1, pad_mode=pad_mode),
        nn.ELU(),
        SqueezeExcite(chan_out) if squeeze_excite else None
    ))
