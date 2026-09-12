from dataclasses import dataclass

@dataclass
class FluxVariantConfig:
    steps: int
    guidance_scale: float
    max_sequence_length: int

# Schnell requires 4 steps and 0 CFG
SCHNELL_CONFIG = FluxVariantConfig(
    steps=4,
    guidance_scale=0.0,
    max_sequence_length=512
)

# Dev requires ~25 steps and ~3.5 CFG
DEV_CONFIG = FluxVariantConfig(
    steps=25,
    guidance_scale=3.5,
    max_sequence_length=512
)

def get_config_for_variant(variant: str) -> FluxVariantConfig:
    """Returns the correct defaults based on the Flux variant."""
    if variant.lower() == "schnell":
        return SCHNELL_CONFIG
    elif variant.lower() == "dev":
        return DEV_CONFIG
    else:
        raise ValueError(f"Unknown Flux variant: {variant}")