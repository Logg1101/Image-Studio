import numpy as np
import torch
from typing import Dict, Any, Tuple


class OutputValidator:
    """
    Quality & Anti-Bleed Validator for upscaled image arrays.
    """

    @staticmethod
    def validate_output(
        image_np: np.ndarray,
        expected_shape: Tuple[int, int, int],
        min_val: float = 0.0,
        max_val: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Validates output array properties.
        """
        errors = []
        warnings = []

        if image_np.shape != expected_shape:
            errors.append(f"Shape mismatch: Expected {expected_shape}, got {image_np.shape}")

        if np.any(np.isnan(image_np)):
            errors.append("Output contains NaN values.")

        if np.any(np.isinf(image_np)):
            errors.append("Output contains Infinite values.")

        cur_min = float(np.min(image_np))
        cur_max = float(np.max(image_np))

        if cur_min < min_val - 1e-4 or cur_max > max_val + 1e-4:
            warnings.append(f"Values out of expected range [{min_val}, {max_val}]: [{cur_min:.4f}, {cur_max:.4f}]")

        is_valid = len(errors) == 0
        return {
            "is_valid": is_valid,
            "errors": errors,
            "warnings": warnings,
            "shape": list(image_np.shape),
            "min": cur_min,
            "max": cur_max,
            "mean": float(np.mean(image_np)),
        }


output_validator = OutputValidator()
