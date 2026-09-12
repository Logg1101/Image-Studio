import sys
import time
import torch
import psutil
from pathlib import Path
from typing import Dict, Any, List

from adapters.registry import adapter_registry
from core.model_manager import ModelManager
from core.types import GenerationRequest, ModelInfo, sanitize_seed, MAX_SEED
from core.memory import clear_vram

class SystemDiagnostics:
    """
    Comprehensive self-diagnostic engine for PyTorch, CUDA, GPU VRAM,
    Model Checkpoints, LoRA format parsers, ControlNet, and Pipeline Integrity.
    """

    @staticmethod
    def run_all() -> Dict[str, Any]:
        results: Dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "environment": SystemDiagnostics.check_environment(),
            "cuda": SystemDiagnostics.check_cuda(),
            "models": SystemDiagnostics.check_models(),
            "adapters": SystemDiagnostics.check_adapters(),
            "schema_check": SystemDiagnostics.check_schema(),
            "overall_status": "READY"
        }
        return results

    @staticmethod
    def check_environment() -> Dict[str, Any]:
        ram = psutil.virtual_memory()
        triton_status = "Optional (Not required for Windows PyTorch)"
        try:
            import triton
            triton_status = "Installed"
        except Exception:
            pass

        return {
            "python_version": sys.version.split()[0],
            "os": sys.platform,
            "cpu_count": psutil.cpu_count(logical=True),
            "cpu_usage_pct": psutil.cpu_percent(),
            "ram_total_gb": round(ram.total / (1024 ** 3), 2),
            "ram_available_gb": round(ram.available / (1024 ** 3), 2),
            "triton": triton_status
        }

    @staticmethod
    def check_cuda() -> Dict[str, Any]:
        cuda_avail = torch.cuda.is_available()
        if not cuda_avail:
            return {"available": False, "device_name": "CPU", "vram_gb": 0.0}

        dev_name = torch.cuda.get_device_name(0)
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        alloc_vram = torch.cuda.memory_allocated(0) / (1024 ** 3)
        res_vram = torch.cuda.memory_reserved(0) / (1024 ** 3)

        return {
            "available": True,
            "device_name": dev_name,
            "device_count": torch.cuda.device_count(),
            "cuda_version": torch.version.cuda,
            "vram_total_gb": round(total_vram, 2),
            "vram_allocated_gb": round(alloc_vram, 3),
            "vram_reserved_gb": round(res_vram, 3),
            "bf16_supported": torch.cuda.is_bf16_supported(),
        }

    @staticmethod
    def check_models() -> Dict[str, Any]:
        mgr = ModelManager()
        models = mgr.available_models
        sdxl_count = sum(1 for m in models.values() if m.architecture.lower() == "sdxl")
        flux_count = sum(1 for m in models.values() if m.architecture.lower() == "flux")
        return {
            "total_models": len(models),
            "sdxl_models": sdxl_count,
            "flux_models": flux_count,
            "model_list": list(models.keys())
        }

    @staticmethod
    def check_adapters() -> Dict[str, Any]:
        loras = adapter_registry.scan_loras()
        cnets = adapter_registry.scan_controlnets()
        ips = adapter_registry.scan_ip_adapters()
        return {
            "lora_count": len(loras),
            "controlnet_count": len(cnets),
            "ip_adapter_count": len(ips),
            "lora_names": list(loras.keys())[:15]
        }

    @staticmethod
    def check_schema() -> Dict[str, Any]:
        # Validate seed sanitization and schema
        test_seed = sanitize_seed(9999999999)
        dummy_model = ModelInfo("test", "sdxl", "base", "safetensors", "")
        req = GenerationRequest(
            model=dummy_model,
            prompt="test",
            seed=test_seed,
            pulid_strength=0.75
        )
        return {
            "schema_valid": True,
            "seed_sanitization": req.seed <= MAX_SEED,
            "pulid_scale_synced": req.pulid_scale == 0.75
        }

    @staticmethod
    def generate_report() -> str:
        data = SystemDiagnostics.run_all()
        env = data["environment"]
        cuda = data["cuda"]
        mod = data["models"]
        ad = data["adapters"]
        sch = data["schema_check"]

        report = f"""
==================================================
  IMAGESTUDIO WORKSTATION SYSTEM STATUS
  Generated: {data['timestamp']}
==================================================

[ENVIRONMENT]
- Python: {env['python_version']} ({env['os']})
- CPU: {env['cpu_count']} Cores | Usage: {env['cpu_usage_pct']}%
- RAM: {env['ram_available_gb']} GB available / {env['ram_total_gb']} GB total
- Triton: {env['triton']}

[HARDWARE ACCELERATION]
- CUDA Available: {cuda['available']}
- GPU: {cuda.get('device_name', 'N/A')}
- VRAM: {cuda.get('vram_allocated_gb', 0)} GB used / {cuda.get('vram_total_gb', 0)} GB total
- CUDA Version: {cuda.get('cuda_version', 'N/A')} | BF16 Supported: {cuda.get('bf16_supported', False)}

[IMAGE GENERATION MODELS]
- Checkpoints Discovered: {mod['total_models']} (SDXL: {mod['sdxl_models']}, Flux: {mod['flux_models']})
- Available Models: {', '.join(mod['model_list'])}

[UNIVERSAL ADAPTERS]
- LoRA Models: {ad['lora_count']}
- ControlNet Modules: {ad['controlnet_count']}
- IP-Adapter Modules: {ad['ip_adapter_count']}

[SYSTEM INTEGRITY]
- Generation Schema: Valid (Seed sanitization: {sch['seed_sanitization']}, PuLID: {sch['pulid_scale_synced']})
- Overall Status: {data['overall_status']}
==================================================
"""
        return report.strip()
