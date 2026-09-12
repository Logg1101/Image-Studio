import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any


class ResidualDenseBlock_5C(nn.Module):
    """5-convolution Residual Dense Block for RRDBNet."""

    def __init__(self, nf: int = 64, gc: int = 32, bias: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block."""

    def __init__(self, nf: int = 64, gc: int = 32):
        super().__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """
    RealESRGAN / UltraSharp / NMKD Superscale RRDBNet Architecture.
    Zero external dependencies, pure PyTorch implementation.
    """

    def __init__(
        self,
        in_nc: int = 3,
        out_nc: int = 3,
        nf: int = 64,
        nb: int = 23,
        gc: int = 32,
        scale: int = 4,
    ):
        super().__init__()
        self.scale = scale
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.body = nn.Sequential(*[RRDB(nf, gc) for _ in range(nb)])
        self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_up1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_up2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fea = self.conv_first(x)
        trunk = self.conv_body(self.body(fea))
        fea = fea + trunk

        fea = self.lrelu(self.conv_up1(F.interpolate(fea, scale_factor=2, mode="nearest")))
        fea = self.lrelu(self.conv_up2(F.interpolate(fea, scale_factor=2, mode="nearest")))
        out = self.conv_last(self.lrelu(self.conv_hr(fea)))
        return out

    def load_esrgan_state(self, state_dict: Dict[str, Any]) -> None:
        """
        Remaps ESRGAN/UltraSharp/NMKD checkpoint keys and loads state dict strictly.
        """
        if "params_ema" in state_dict:
            state_dict = state_dict["params_ema"]
        elif "params" in state_dict:
            state_dict = state_dict["params"]
        elif "model" in state_dict:
            state_dict = state_dict["model"]

        key_mapping = {}
        for k, v in state_dict.items():
            if k.startswith("model.0."):
                new_k = k.replace("model.0.", "conv_first.")
            elif k.startswith("model.1.sub.23."):
                new_k = k.replace("model.1.sub.23.", "conv_body.")
            elif k.startswith("model.1.sub."):
                clean_k = k.replace("model.1.sub.", "body.")
                for c in ["conv1", "conv2", "conv3", "conv4", "conv5"]:
                    clean_k = clean_k.replace(f".{c}.0.", f".{c}.")
                new_k = clean_k
            elif k.startswith("model.3."):
                new_k = k.replace("model.3.", "conv_up1.")
            elif k.startswith("model.6."):
                new_k = k.replace("model.6.", "conv_up2.")
            elif k.startswith("model.8."):
                new_k = k.replace("model.8.", "conv_hr.")
            elif k.startswith("model.10."):
                new_k = k.replace("model.10.", "conv_last.")
            else:
                new_k = k
            key_mapping[new_k] = v

        self.load_state_dict(key_mapping, strict=True)
