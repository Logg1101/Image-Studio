import numpy as np
from typing import Dict, Any, Tuple, Optional


class ColorTemperatureEstimator:
    """
    Color Temperature & Chromaticity Estimator.
    Computes Correlated Color Temperature (CCT) in Kelvin and chromaticity balance
    using weighted illuminant estimation on neutral and diffuse image regions.
    """

    @staticmethod
    def estimate_temperature(
        image_np: np.ndarray,
        neutral_mask: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Estimates correlated color temperature (CCT) in Kelvin and color bias.

        Args:
            image_np (np.ndarray): RGB float array [H, W, 3] in [0.0, 1.0].
            neutral_mask (Optional[np.ndarray]): Spatial mask weighting neutral/diffuse zones [H, W].

        Returns:
            Dict[str, Any]: CCT in Kelvin, color bias category, and normalized RGB tint.
        """
        h, w, _ = image_np.shape

        if neutral_mask is None:
            # Automatic neutral zone estimation: pixels with low saturation and moderate luminance
            r, g, b = image_np[..., 0], image_np[..., 1], image_np[..., 2]
            max_c = np.maximum(np.maximum(r, g), b)
            min_c = np.minimum(np.minimum(r, g), b)
            saturation = (max_c - min_c) / (max_c + 1e-5)
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            weight = (1.0 - saturation) * ((luminance > 0.2) & (luminance < 0.9)).astype(np.float32)
        else:
            weight = neutral_mask

        total_weight = np.sum(weight) + 1e-6
        mean_r = float(np.sum(image_np[..., 0] * weight) / total_weight)
        mean_g = float(np.sum(image_np[..., 1] * weight) / total_weight)
        mean_b = float(np.sum(image_np[..., 2] * weight) / total_weight)

        # Convert sRGB to linear RGB
        def srgb_to_linear(c):
            return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

        lin_r = srgb_to_linear(mean_r)
        lin_g = srgb_to_linear(mean_g)
        lin_b = srgb_to_linear(mean_b)

        # Convert to CIE XYZ
        X = lin_r * 0.4124564 + lin_g * 0.3575761 + lin_b * 0.1804375
        Y = lin_r * 0.2126729 + lin_g * 0.7151522 + lin_b * 0.0721750
        Z = lin_r * 0.0193339 + lin_g * 0.1191920 + lin_b * 0.9503041

        denom = X + Y + Z + 1e-6
        x = X / denom
        y = Y / denom

        # McCamy's CCT formula: CCT = 449n^3 + 3525n^2 + 6823.3n + 5520.33, where n = (x - 0.3320)/(0.1858 - y)
        n = (x - 0.3320) / (0.1858 - y + 1e-6)
        cct = 449.0 * (n ** 3) + 3525.0 * (n ** 2) + 6823.3 * n + 5520.33
        cct = float(np.clip(cct, 2200.0, 10000.0))

        # Determine bias
        if cct < 4800.0:
            bias = "warm"
        elif cct > 6500.0:
            bias = "cool"
        else:
            bias = "neutral"

        # Normalized tint
        total_rgb = mean_r + mean_g + mean_b + 1e-6
        tint_rgb = [mean_r / total_rgb * 3.0, mean_g / total_rgb * 3.0, mean_b / total_rgb * 3.0]

        return {
            "color_temperature_k": cct,
            "color_bias": bias,
            "color_tint_rgb": tint_rgb,
            "mean_rgb": [mean_r, mean_g, mean_b],
        }


color_temperature_estimator = ColorTemperatureEstimator()
