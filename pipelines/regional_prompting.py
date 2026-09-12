import logging
from typing import List, Tuple, Optional, Any
from pathlib import Path
import torch
import numpy as np
from PIL import Image

from core.types import CharacterRegion, GenerationRequest

logger = logging.getLogger(__name__)


def _disable_active_loras(pipeline: Any) -> None:
    """
    Safely disables active LoRA layers and resets active adapters to prevent
    concept bleed-through across regional zones or future generation runs.
    """
    if pipeline is None:
        return

    if hasattr(pipeline, "disable_lora"):
        try:
            pipeline.disable_lora()
        except (RuntimeError, ValueError, TypeError, AttributeError, KeyError) as e:
            logger.warning("[RegionalPrompting] pipeline.disable_lora() failed: %s", e)

    if hasattr(pipeline, "set_adapters"):
        try:
            pipeline.set_adapters([], adapter_weights=[])
        except (RuntimeError, ValueError, TypeError, AttributeError, KeyError):
            try:
                if hasattr(pipeline, "get_active_adapters"):
                    active = pipeline.get_active_adapters()
                    if active:
                        pipeline.set_adapters(active, adapter_weights=[0.0] * len(active))
            except (RuntimeError, ValueError, TypeError, AttributeError, KeyError) as e:
                logger.warning("[RegionalPrompting] Failed resetting adapters to zero weight: %s", e)


def _activate_single_lora(pipeline: Any, char_lora_key: str, strength: float) -> None:
    """
    Enables LoRA layers and activates precisely one LoRA adapter for a regional character zone.
    """
    if pipeline is None:
        return

    if hasattr(pipeline, "enable_lora"):
        try:
            pipeline.enable_lora()
        except (RuntimeError, ValueError, TypeError, AttributeError, KeyError) as e:
            logger.warning("[RegionalPrompting] pipeline.enable_lora() failed: %s", e)

    if hasattr(pipeline, "set_adapters"):
        try:
            pipeline.set_adapters([char_lora_key], adapter_weights=[float(strength)])
        except (RuntimeError, ValueError, TypeError, AttributeError, KeyError) as e:
            logger.warning("[RegionalPrompting] pipeline.set_adapters([%s]) failed: %s", char_lora_key, e)


def get_default_regions(count: int) -> List[Tuple[float, float, float, float]]:
    """
    Computes balanced spatial regions [x0, y0, x1, y1] for 2, 3, or 4 characters.
    Coordinates are normalized in [0.0, 1.0].
    """
    if count <= 1:
        return [(0.0, 0.0, 1.0, 1.0)]
    elif count == 2:
        return [
            (0.0, 0.0, 0.5, 1.0),
            (0.5, 0.0, 1.0, 1.0)
        ]
    elif count == 3:
        return [
            (0.0, 0.0, 0.3333, 1.0),
            (0.3333, 0.0, 0.6667, 1.0),
            (0.6667, 0.0, 1.0, 1.0)
        ]
    else:  # 4 characters: 4 vertical columns
        return [
            (0.0, 0.0, 0.25, 1.0),
            (0.25, 0.0, 0.5, 1.0),
            (0.5, 0.0, 0.75, 1.0),
            (0.75, 0.0, 1.0, 1.0)
        ]

def create_regional_masks(
    latent_h: int,
    latent_w: int,
    characters: List[CharacterRegion],
    device: str = "cuda",
    dtype: torch.dtype = torch.float16
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """
    Constructs normalized, soft-feathered spatial masks of shape (1, 1, H, W)
    for each character region, along with a residual background/base mask.
    Modular design ensures arbitrary shapes or bounding boxes can be accommodated.
    """
    ys = torch.linspace(0.0, 1.0, latent_h, device=device, dtype=dtype).view(-1, 1)
    xs = torch.linspace(0.0, 1.0, latent_w, device=device, dtype=dtype).view(1, -1)

    raw_masks = []
    for char in characters:
        x0, y0, x1, y1 = char.box
        x0, x1 = min(x0, x1), max(x0, x1)
        y0, y1 = min(y0, y1), max(y0, y1)
        x0 = max(0.0, min(1.0, float(x0)))
        x1 = max(0.0, min(1.0, float(x1)))
        y0 = max(0.0, min(1.0, float(y0)))
        y1 = max(0.0, min(1.0, float(y1)))

        f = max(float(char.feather), 1e-4)
        left = torch.clamp((xs - x0) / f, 0.0, 1.0)
        right = torch.clamp((x1 - xs) / f, 0.0, 1.0)
        top = torch.clamp((ys - y0) / f, 0.0, 1.0)
        bottom = torch.clamp((y1 - ys) / f, 0.0, 1.0)

        # 2D soft mask
        mask = left * right * top * bottom
        raw_masks.append(mask.unsqueeze(0).unsqueeze(0))  # (1, 1, H, W)

    if not raw_masks:
        full = torch.ones((1, 1, latent_h, latent_w), device=device, dtype=dtype)
        return [full], torch.zeros((1, 1, latent_h, latent_w), device=device, dtype=dtype)

    stacked = torch.cat(raw_masks, dim=0)  # (K, 1, H, W)
    total_mask = torch.sum(stacked, dim=0, keepdim=True)  # (1, 1, H, W)

    # Normalize character masks where they overlap or exceed 1.0
    norm_factor = torch.clamp(total_mask, min=1.0)
    normalized_masks = [m / norm_factor for m in raw_masks]

    # Background residual mask (for unassigned canvas space)
    base_mask = torch.clamp(1.0 - total_mask, min=0.0, max=1.0)

    return normalized_masks, base_mask

def generate_regional_sdxl(
    pipeline: Any,
    compel_processor: Any,
    request: GenerationRequest,
    generator: torch.Generator,
    adapter_registry: Any
) -> Image.Image:
    """
    Samples a multi-character SDXL image using per-character regional prompting,
    isolated LoRA adapter routing, and smooth latent noise composite.
    """
    from core.device import get_torch_device, get_torch_dtype
    from core.memory import clear_vram
    device = get_torch_device()
    dtype = get_torch_dtype(device)

    clear_vram()

    # 1. Sync and activate any needed LoRAs across characters
    needed_loras = {}
    for char in request.characters:
        if char.lora_name and str(char.lora_name).strip() and str(char.lora_name).lower() != "none":
            needed_loras[char.lora_name] = float(char.lora_strength)

    if needed_loras:
        adapter_registry.sync_loras(pipeline, needed_loras, target_architecture="sdxl")
    else:
        adapter_registry.unload_all_loras(pipeline)

    # 2. Build spatial masks
    latent_h = int(request.height) // 8
    latent_w = int(request.width) // 8
    char_masks, base_mask = create_regional_masks(
        latent_h, latent_w, request.characters, device=device, dtype=dtype
    )
    has_base = bool(torch.max(base_mask) > 0.01)

    # 3. Prompt Conditioning via Compel
    neg_prompt = request.negative_prompt or ""
    with torch.no_grad():
        neg_cond, neg_pool = compel_processor(neg_prompt)
        base_cond, base_pool = compel_processor(request.prompt or "")

        char_conds = []
        char_pools = []
        for char in request.characters:
            char_p = char.prompt.strip()
            if char_p:
                comb_p = f"{char_p}, {request.prompt.strip()}" if request.prompt and request.prompt.strip() else char_p
            else:
                comb_p = request.prompt.strip() or "masterpiece"
            c_cond, c_pool = compel_processor(comb_p)
            char_conds.append(c_cond)
            char_pools.append(c_pool)

        # Pad conditionings to matching token sequence lengths
        all_conds = [neg_cond, base_cond] + char_conds
        empty_c, _ = compel_processor("")
        padded_conds = compel_processor.pad_conditioning_tensors_to_same_length(
            all_conds, precomputed_padding=empty_c
        )

        neg_cond = padded_conds[0].contiguous().to(device=device, dtype=dtype)
        base_cond = padded_conds[1].contiguous().to(device=device, dtype=dtype)
        char_conds_padded = [p.contiguous().to(device=device, dtype=dtype) for p in padded_conds[2:]]

        neg_pool = neg_pool.contiguous().to(device=device, dtype=dtype)
        base_pool = base_pool.contiguous().to(device=device, dtype=dtype)
        char_pools_c = [p.contiguous().to(device=device, dtype=dtype) for p in char_pools]

    # 4. Prepare SDXL Added Time IDs
    add_time_ids = pipeline._get_add_time_ids(
        (int(request.height), int(request.width)),
        (0, 0),
        (int(request.height), int(request.width)),
        dtype=dtype,
        text_encoder_projection_dim=pipeline.text_encoder_2.config.projection_dim
    ).to(device=device)

    do_cfg = float(request.guidance_scale) > 1.0
    add_time_ids_input = torch.cat([add_time_ids, add_time_ids], dim=0) if do_cfg else add_time_ids

    # 5. Prepare Initial Latents & Timesteps under inference_mode
    with torch.inference_mode():
        latents = pipeline.prepare_latents(
            batch_size=1,
            num_channels_latents=4,
            height=int(request.height),
            width=int(request.width),
            dtype=dtype,
            device=device,
            generator=generator
        )

        pipeline.scheduler.set_timesteps(int(request.steps), device=device)
        timesteps = pipeline.scheduler.timesteps

        # 6. Regional Multi-Character Denoising Loop
        try:
            for i, t in enumerate(timesteps):
                if request.step_callback:
                    request.step_callback(i + 1, request.steps)

                latent_model_input = pipeline.scheduler.scale_model_input(latents, t)
                if do_cfg:
                    latent_model_input = torch.cat([latent_model_input] * 2)

                noise_pred_composite = torch.zeros_like(latents)

                # A. Character UNet passes with isolated LoRAs
                for k, char in enumerate(request.characters):
                    # Select LoRA for character k only
                    char_lora_key = None
                    if char.lora_name and str(char.lora_name).strip() and str(char.lora_name).lower() != "none":
                        cand_stem = Path(char.lora_name).stem
                        if cand_stem in adapter_registry.active_loras:
                            char_lora_key = cand_stem
                        elif char.lora_name in adapter_registry.active_loras:
                            char_lora_key = char.lora_name

                    if char_lora_key:
                        _activate_single_lora(pipeline, char_lora_key, float(char.lora_strength))
                    else:
                        _disable_active_loras(pipeline)

                    # Prepare text and pooled embeds for character k
                    if do_cfg:
                        c_text = torch.cat([neg_cond, char_conds_padded[k]], dim=0)
                        c_pool = torch.cat([neg_pool, char_pools_c[k]], dim=0)
                    else:
                        c_text = char_conds_padded[k]
                        c_pool = char_pools_c[k]

                    added_cond = {"text_embeds": c_pool, "time_ids": add_time_ids_input}

                    pred = pipeline.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=c_text,
                        added_cond_kwargs=added_cond,
                        return_dict=False
                    )[0]

                    if do_cfg:
                        pred_uncond, pred_text = pred.chunk(2)
                        eps_k = pred_uncond + float(request.guidance_scale) * (pred_text - pred_uncond)
                    else:
                        eps_k = pred

                    # Add to composite weighted by character's spatial region mask
                    noise_pred_composite = noise_pred_composite + (char_masks[k] * eps_k)

                # B. Shared scene background pass (if unassigned region exists)
                if has_base:
                    _disable_active_loras(pipeline)

                    if do_cfg:
                        c_text = torch.cat([neg_cond, base_cond], dim=0)
                        c_pool = torch.cat([neg_pool, base_pool], dim=0)
                    else:
                        c_text = base_cond
                        c_pool = base_pool

                    added_cond = {"text_embeds": c_pool, "time_ids": add_time_ids_input}

                    pred = pipeline.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=c_text,
                        added_cond_kwargs=added_cond,
                        return_dict=False
                    )[0]

                    if do_cfg:
                        pred_uncond, pred_text = pred.chunk(2)
                        eps_base = pred_uncond + float(request.guidance_scale) * (pred_text - pred_uncond)
                    else:
                        eps_base = pred

                    noise_pred_composite = noise_pred_composite + (base_mask * eps_base)

                # C. Single synchronized scheduler step for all regions
                latents = pipeline.scheduler.step(noise_pred_composite, t, latents, return_dict=False)[0]

            # 7. Decode Latents via VAE
            latents = latents / pipeline.vae.config.scaling_factor
            image_tensor = pipeline.vae.decode(latents, return_dict=False)[0]
            # Convert tensor (-1..1) to uint8 PIL Image
            image_np = (image_tensor / 2 + 0.5).clamp(0, 1).squeeze(0)
            image_np = image_np.cpu().permute(1, 2, 0).float().numpy()
            image_np = (image_np * 255.0).round().astype(np.uint8)
            pil_image = Image.fromarray(image_np)
        finally:
            # Guarantee adapters are completely reset back to clean state
            _disable_active_loras(pipeline)

    # Free conditioning tensors
    del neg_cond, neg_pool, base_cond, base_pool, char_conds_padded, char_pools_c
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return pil_image
