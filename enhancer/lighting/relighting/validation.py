import numpy as np
import scipy.ndimage as ndi
from typing import Dict, Any, Tuple


class RelightingValidator:
    """
    Quality & Geometric Invariance Validator for Relighted Output Images.
    """

    @staticmethod
    def validate(
        base_np: np.ndarray,
        relit_np: np.ndarray,
        expected_shape: Tuple[int, int, int],
    ) -> Dict[str, Any]:
        """
        Validates output array properties and verifies zero geometric distortion.
        """
        errors = []
        warnings = []

        if relit_np.shape != expected_shape:
            errors.append(f"Shape mismatch: expected {expected_shape}, got {relit_np.shape}")

        if np.any(np.isnan(relit_np)):
            errors.append("Relit image contains NaN values.")

        if np.any(np.isinf(relit_np)):
            errors.append("Relit image contains Infinite values.")

        cur_min = float(np.min(relit_np))
        cur_max = float(np.max(relit_np))

        if cur_min < -1e-4 or cur_max > 1.0 + 1e-4:
            warnings.append(f"Pixel values slightly out of bounds: [{cur_min:.4f}, {cur_max:.4f}]")

        # Geometric Invariance Check: High-pass edge location comparison
        base_gray = 0.299 * base_np[..., 0] + 0.587 * base_np[..., 1] + 0.114 * base_np[..., 2]
        relit_gray = 0.299 * relit_np[..., 0] + 0.587 * relit_np[..., 1] + 0.114 * relit_np[..., 2]

        b_gy, b_gx = np.gradient(base_gray)
        r_gy, r_gx = np.gradient(relit_gray)

        b_edge = np.sqrt(b_gx ** 2 + b_gy ** 2)
        r_edge = np.sqrt(r_gx ** 2 + r_gy ** 2)

        # Correlation between gradient magnitude fields
        flat_b = b_edge.flatten()
        flat_r = r_edge.flatten()
        corr = float(np.corrcoef(flat_b, flat_r)[0, 1])

        if corr < 0.60:
            errors.append(f"Structural edge correlation too low ({corr:.3f}), indicating geometric distortion.")

        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "edge_correlation": round(corr, 4),
            "min": cur_min,
            "max": cur_max,
            "mean_delta": float(np.mean(np.abs(relit_np - base_np))),
        }


relighting_validator = RelightingValidator()
