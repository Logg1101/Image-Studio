import torch
import torch.nn.functional as F
import numpy as np
from typing import Callable, Optional


class TiledUpscaler:
    """
    Seamless Overlap-Aware Tiled Inference Processor.
    Splits large images into overlapping tiles, runs upscaling inference sequentially,
    and blends overlapping seams with 2D cosine weight windows to guarantee 0 boundary artifacts.
    """

    def __init__(
        self,
        tile_size: int = 512,
        tile_overlap: int = 48,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.device = device

    def _build_tile_weights(
        self,
        tile_h: int,
        tile_w: int,
        overlap_h: int,
        overlap_w: int,
        ramp_top: bool,
        ramp_bottom: bool,
        ramp_left: bool,
        ramp_right: bool,
    ) -> torch.Tensor:
        """
        Constructs a 2D cosine feathering window for seamless directional overlap blending.
        """
        wy = torch.ones(tile_h, dtype=torch.float32)
        if ramp_top and overlap_h > 0:
            wy[:overlap_h] = 0.5 - 0.5 * torch.cos(torch.linspace(0, torch.pi, overlap_h))
        if ramp_bottom and overlap_h > 0:
            wy[-overlap_h:] = 0.5 + 0.5 * torch.cos(torch.linspace(0, torch.pi, overlap_h))

        wx = torch.ones(tile_w, dtype=torch.float32)
        if ramp_left and overlap_w > 0:
            wx[:overlap_w] = 0.5 - 0.5 * torch.cos(torch.linspace(0, torch.pi, overlap_w))
        if ramp_right and overlap_w > 0:
            wx[-overlap_w:] = 0.5 + 0.5 * torch.cos(torch.linspace(0, torch.pi, overlap_w))

        w2d = wy[:, None] * wx[None, :]
        return w2d[None, None, :, :]  # [1, 1, H, W]

    def upscale_tiled(
        self,
        image_tensor: torch.Tensor,
        upscale_fn: Callable[[torch.Tensor, int], torch.Tensor],
        scale: int = 4,
    ) -> torch.Tensor:
        """
        Performs overlap-aware tiled inference on image_tensor [1, 3, H, W].

        Args:
            image_tensor (torch.Tensor): Input image tensor in [0.0, 1.0].
            upscale_fn (Callable): Upscale callback `(tile_tensor, scale) -> upscaled_tile`.
            scale (int): Upscale factor (1, 2, 4).

        Returns:
            torch.Tensor: Seamlessly stitched high-resolution output [1, 3, H*scale, W*scale].
        """
        _, _, h, w = image_tensor.shape

        # If image is small enough to process in a single pass without tiling, process directly
        if h <= self.tile_size and w <= self.tile_size:
            return upscale_fn(image_tensor, scale)

        out_h, out_w = h * scale, w * scale
        out_overlap = self.tile_overlap * scale

        # Accumulator buffers
        output_acc = torch.zeros(
            (1, 3, out_h, out_w), dtype=torch.float32, device="cpu"
        )
        weight_acc = torch.zeros(
            (1, 1, out_h, out_w), dtype=torch.float32, device="cpu"
        )

        stride = self.tile_size - self.tile_overlap

        # Iterate over 2D grid
        for y0 in range(0, h, stride):
            y1 = min(y0 + self.tile_size, h)
            # Adjust y0 if at boundary
            actual_y0 = max(0, y1 - self.tile_size) if y1 == h else y0
            actual_h = y1 - actual_y0

            for x0 in range(0, w, stride):
                x1 = min(x0 + self.tile_size, w)
                actual_x0 = max(0, x1 - self.tile_size) if x1 == w else x0
                actual_w = x1 - actual_x0

                # Crop input tile
                tile = image_tensor[:, :, actual_y0:y1, actual_x0:x1]

                # Upscale tile
                out_tile = upscale_fn(tile, scale).to("cpu")

                # Compute output window weights
                cur_out_h, cur_out_w = actual_h * scale, actual_w * scale
                weight_tile = self._build_tile_weights(
                    tile_h=cur_out_h,
                    tile_w=cur_out_w,
                    overlap_h=out_overlap,
                    overlap_w=out_overlap,
                    ramp_top=(actual_y0 > 0),
                    ramp_bottom=(y1 < h),
                    ramp_left=(actual_x0 > 0),
                    ramp_right=(x1 < w),
                )

                # Accumulate
                out_y0, out_y1 = actual_y0 * scale, y1 * scale
                out_x0, out_x1 = actual_x0 * scale, x1 * scale

                output_acc[:, :, out_y0:out_y1, out_x0:out_x1] += (
                    out_tile * weight_tile
                )
                weight_acc[:, :, out_y0:out_y1, out_x0:out_x1] += weight_tile

        # Normalize by accumulated weights
        weight_acc = torch.clamp(weight_acc, min=1e-5)
        final_output = output_acc / weight_acc
        return torch.clamp(final_output, 0.0, 1.0)


tiled_upscaler = TiledUpscaler()
