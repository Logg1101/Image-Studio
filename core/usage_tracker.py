import json
import time
from pathlib import Path
from typing import Dict, Any
import config.paths as paths

class UsageTracker:
    """
    Lightweight local telemetry tracker for generation metrics,
    VRAM peaks, model frequencies, and productivity statistics.
    """
    STATS_FILE = paths.OUTPUTS_DIR / "usage_stats.json"

    @classmethod
    def _load_data(cls) -> Dict[str, Any]:
        if cls.STATS_FILE.exists():
            try:
                return json.loads(cls.STATS_FILE.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {
            "today_date": time.strftime("%Y-%m-%d"),
            "today": {
                "generated": 0,
                "edited": 0,
                "img2img": 0,
                "lora_runs": 0,
                "upscaled": 0,
                "peak_vram_gb": 0.0,
                "total_time_sec": 0.0,
                "avg_time_sec": 0.0
            },
            "all_time": {
                "total_generations": 0,
                "total_edits": 0,
                "total_time_sec": 0.0,
                "checkpoints": {},
                "loras": {},
                "samplers": {}
            }
        }

    @classmethod
    def _save_data(cls, data: Dict[str, Any]):
        paths.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        try:
            cls.STATS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            pass

    @classmethod
    def log_generation(
        cls,
        checkpoint: str,
        sampler: str,
        loras_count: int,
        is_img2img: bool,
        gen_time_sec: float,
        vram_peak_gb: float
    ):
        data = cls._load_data()
        current_date = time.strftime("%Y-%m-%d")

        # Roll over today stats if new day
        if data.get("today_date") != current_date:
            data["today_date"] = current_date
            data["today"] = {
                "generated": 0,
                "edited": 0,
                "img2img": 0,
                "lora_runs": 0,
                "upscaled": 0,
                "peak_vram_gb": 0.0,
                "total_time_sec": 0.0,
                "avg_time_sec": 0.0
            }

        t = data["today"]
        a = data["all_time"]

        t["generated"] += 1
        if is_img2img:
            t["img2img"] += 1
        if loras_count > 0:
            t["lora_runs"] += 1
        t["peak_vram_gb"] = max(t["peak_vram_gb"], round(vram_peak_gb, 2))
        t["total_time_sec"] = round(t["total_time_sec"] + gen_time_sec, 2)
        t["avg_time_sec"] = round(t["total_time_sec"] / max(t["generated"], 1), 2)

        a["total_generations"] += 1
        a["total_time_sec"] = round(a["total_time_sec"] + gen_time_sec, 2)

        # Increment checkpoint frequency
        if checkpoint:
            a["checkpoints"][checkpoint] = a["checkpoints"].get(checkpoint, 0) + 1
        # Increment sampler frequency
        if sampler:
            a["samplers"][sampler] = a["samplers"].get(sampler, 0) + 1

        cls._save_data(data)

    @classmethod
    def log_edit(cls, action_type: str = "edit"):
        data = cls._load_data()
        t = data["today"]
        a = data["all_time"]
        if action_type == "upscale":
            t["upscaled"] += 1
        else:
            t["edited"] += 1
            a["total_edits"] += 1
        cls._save_data(data)

    @classmethod
    def get_summary(cls) -> Dict[str, Any]:
        return cls._load_data()
