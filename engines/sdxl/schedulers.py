from typing import Tuple, List
from diffusers import (
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    HeunDiscreteScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    KDPM2DiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    DDIMScheduler,
    UniPCMultistepScheduler,
    LCMScheduler,
    DEISMultistepScheduler,
    PNDMScheduler
)

# 12 Available Samplers
AVAILABLE_SAMPLERS: List[str] = [
    "Euler a",
    "Euler",
    "Heun",
    "DPM++ 2M",
    "DPM++ 2M SDE",
    "DPM++ 2S a",
    "DPM++ SDE",
    "DPM2",
    "DPM2 a",
    "DDIM",
    "UniPC",
    "LCM"
]

# 5 Available Schedulers (Noise Curves)
AVAILABLE_SCHEDULERS: List[str] = [
    "Normal",
    "Karras",
    "Exponential",
    "SGM Uniform",
    "Simple"
]

def build_sdxl_scheduler(sampler_name: str, scheduler_name: str, base_config: dict):
    """
    Constructs the appropriate Diffusers scheduler class and applies 
    the selected sigma / timestep schedule curve.
    """
    config = dict(base_config)
    
    # 1. Apply Schedule Curve flags
    if scheduler_name == "Karras":
        config["use_karras_sigmas"] = True
    elif scheduler_name == "Exponential":
        config["use_exponential_sigmas"] = True
    elif scheduler_name == "SGM Uniform":
        config["timestep_spacing"] = "trailing"
    elif scheduler_name == "Simple":
        config["timestep_spacing"] = "linspace"
    elif scheduler_name == "Normal":
        config["use_karras_sigmas"] = False
        config["use_exponential_sigmas"] = False
        config["timestep_spacing"] = "leading"

    # 2. Instantiate the target Sampler Algorithm
    if sampler_name == "Euler a":
        # Clean non-applicable flags for Euler a
        config.pop("use_exponential_sigmas", None)
        return EulerAncestralDiscreteScheduler.from_config(config)

    elif sampler_name == "Euler":
        return EulerDiscreteScheduler.from_config(config)

    elif sampler_name == "Heun":
        config.pop("use_exponential_sigmas", None)
        return HeunDiscreteScheduler.from_config(config)

    elif sampler_name == "DPM++ 2M":
        config["algorithm_type"] = "dpmsolver++"
        config["solver_order"] = 2
        return DPMSolverMultistepScheduler.from_config(config)

    elif sampler_name == "DPM++ 2M SDE":
        config["algorithm_type"] = "sde-dpmsolver++"
        config["solver_order"] = 2
        return DPMSolverMultistepScheduler.from_config(config)

    elif sampler_name == "DPM++ 2S a":
        return DPMSolverSinglestepScheduler.from_config(config)

    elif sampler_name == "DPM++ SDE":
        config["algorithm_type"] = "sde-dpmsolver++"
        config["solver_order"] = 1
        return DPMSolverMultistepScheduler.from_config(config)

    elif sampler_name == "DPM2":
        config.pop("use_exponential_sigmas", None)
        return KDPM2DiscreteScheduler.from_config(config)

    elif sampler_name == "DPM2 a":
        config.pop("use_exponential_sigmas", None)
        return KDPM2AncestralDiscreteScheduler.from_config(config)

    elif sampler_name == "DDIM":
        return DDIMScheduler.from_config(config)

    elif sampler_name == "UniPC":
        return UniPCMultistepScheduler.from_config(config)

    elif sampler_name == "LCM":
        return LCMScheduler.from_config(config)

    # Fallback to Euler a
    return EulerAncestralDiscreteScheduler.from_config(base_config)