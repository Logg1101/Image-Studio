from enhancer.upscaling.adapters.realcugan import RealCUGANAdapter
from enhancer.upscaling.adapters.tiled_sdxl import TiledSDXLAdapter
from enhancer.upscaling.adapters.ultrasharp import UltraSharpAdapter
from enhancer.upscaling.adapters.nmkd import NMKDSuperscaleAdapter
from enhancer.upscaling.adapters.supir import SUPIRAdapter
from enhancer.upscaling.adapters.lanczos import LanczosAdapter

__all__ = [
    "RealCUGANAdapter",
    "TiledSDXLAdapter",
    "UltraSharpAdapter",
    "NMKDSuperscaleAdapter",
    "SUPIRAdapter",
    "LanczosAdapter",
]
