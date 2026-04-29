"""
Generate upsampled (150x280) frames using both denoiser + upsampler.

Shows the full DIAMOND pipeline: low-res prediction -> super-resolution.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from tqdm import tqdm

SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from hydra.utils import instantiate
from omegaconf import OmegaConf

from agent import Agent
from models.diffusion import DiffusionSampler, DiffusionSamplerConfig
from data import Dataset, CS2Hdf5Dataset
from data.segment import SegmentId

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results" / "upsampled"


def tensor_to_image(t):
    img = t.permute(1, 2, 0).numpy()
    img = ((img + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "results" / "cloud_training" / "agent_epoch_00149.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--denoising-steps", type=int, default=3)
    parser.add_argument("--upsampler-steps", type=int, default=10)
    args = parser.parse_args()

    device = torch.device(args.device)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load config
    cfg = OmegaConf.load(ROOT / "config" / "trainer.yaml")
    agent_cfg = OmegaConf.load(ROOT / "config" / "agent" / "cs2.yaml")
    env_cfg = OmegaConf.load(ROOT / "config" / "env" / "cs2.yaml")
    cfg = OmegaConf.merge(cfg, {"agent": agent_cfg, "env": env_cfg})
    OmegaConf.resolve(cfg)

    # Build and load model
    agent_cfg_inst = instantiate(cfg.agent, num_actions=cfg.env.num_actions)
    agent = Agent(agent_cfg_inst).to(device)
    agent.load(args.checkpoint, load_denoiser=True, load_upsampler=True)
    agent.eval()

    # Load test data
    low_res_path = ROOT / "data" / "processed" / "low_res"
    full_res_path = ROOT / "data" / "processed" / "full_res"
    dataset_full_res = CS2Hdf5Dataset(full_res_path)
    test_dataset = Dataset(low_res_path / "test", dataset_full_res, "test_dataset", cache_in_ram=True)
    test_dataset.load_from_default_path()

    # Setup samplers
    denoiser_sampler = DiffusionSampler(
        agent.denoiser.to(device),
        DiffusionSamplerConfig(num_steps_denoising=args.denoising_steps, sigma_min=2e-3, sigma_max=5.0),
    )

    upsampler_sampler = DiffusionSampler(
        agent.upsampler.to(device),
        DiffusionSamplerConfig(num_steps_denoising=args.upsampler_steps, sigma_min=2e-3, sigma_max=5.0),
    )

    n_cond = 4
    episode = test_dataset.load_episode(0)
    num_frames = len(episode.obs)
    upsampling_factor = 5  # 30x56 -> 150x280

    # Get full-res episode for ground truth
    full_res_ep = dataset_full_res.load_episode(
        episode.info.get("original_file_id", 0) if isinstance(episode.info, dict) else 0
    )

    positions = np.linspace(0, num_frames - n_cond - 2, args.num_samples, dtype=int)

    print(f"Generating {args.num_samples} upsampled predictions...")
    for idx, pos in enumerate(tqdm(positions)):
        pos = int(pos)
        # Low-res conditioning
        cond_obs = episode.obs[pos:pos + n_cond].unsqueeze(0).to(device)
        cond_act = episode.act[pos:pos + n_cond].unsqueeze(0).to(device)

        # 1. Denoiser: predict low-res next frame
        with torch.no_grad():
            low_res_pred, _ = denoiser_sampler.sample(cond_obs, cond_act)

        # 2. Upsampler: upscale to full res
        # The upsampler needs: 1 previous full-res frame + bicubic-upsampled low-res prediction
        # Get previous full-res frame from the dataset
        prev_full_res_idx = pos + n_cond - 1
        if prev_full_res_idx < len(full_res_ep.obs):
            prev_full_res = full_res_ep.obs[prev_full_res_idx].unsqueeze(0).unsqueeze(0).to(device)
        else:
            # Fallback: bicubic upsample the last low-res frame
            prev_full_res = F.interpolate(
                cond_obs[:, -1:].reshape(1, 3, 30, 56),
                scale_factor=upsampling_factor, mode="bicubic"
            ).unsqueeze(1)

        # Bicubic upsample the low-res prediction
        low_res_upsampled = F.interpolate(
            low_res_pred, scale_factor=upsampling_factor, mode="bicubic"
        )

        # Upsampler expects prev_obs = (prev_full_res, bicubic_upsampled) concatenated
        # DiffusionSampler.sample reshapes (B, T, C, H, W) -> (B, T*C, H, W)
        # So pass T=2, C=3 to get 6 channels for obs, then +3 noisy = 9 total
        upsampler_input = torch.cat([
            prev_full_res,  # (1, 1, 3, 150, 280)
            low_res_upsampled.unsqueeze(1),  # (1, 1, 3, 150, 280)
        ], dim=1)  # (1, 2, 3, 150, 280)

        with torch.no_grad():
            high_res_pred, _ = upsampler_sampler.sample(upsampler_input, None)

        # Get ground truth full-res
        gt_full_res_idx = pos + n_cond
        gt_full_res = full_res_ep.obs[gt_full_res_idx] if gt_full_res_idx < len(full_res_ep.obs) else None

        # Save images
        lr_img = tensor_to_image(low_res_pred.squeeze(0).cpu())
        lr_up_img = tensor_to_image(low_res_upsampled.squeeze(0).cpu())
        hr_img = tensor_to_image(high_res_pred.squeeze(0).cpu())

        # Upscale low-res for visual comparison (nearest neighbor to see pixels)
        lr_nn = lr_img.resize((280, 150), Image.NEAREST)

        # Save individual images
        lr_nn.save(RESULTS_DIR / f"low_res_{idx:03d}.png")
        lr_up_img.save(RESULTS_DIR / f"bicubic_{idx:03d}.png")
        hr_img.save(RESULTS_DIR / f"upsampled_{idx:03d}.png")

        if gt_full_res is not None:
            gt_img = tensor_to_image(gt_full_res.cpu())
            gt_img.save(RESULTS_DIR / f"ground_truth_{idx:03d}.png")

            # Comparison: low-res(nn) | bicubic | upsampled | ground truth
            w, h = 280, 150
            comp = Image.new("RGB", (4 * w + 12, h + 20), (0, 0, 0))
            draw = ImageDraw.Draw(comp)
            labels = ["Low-res (NN upscale)", "Bicubic upscale", "Upsampler output", "Ground Truth"]
            for j, (img, label) in enumerate(zip([lr_nn, lr_up_img, hr_img, gt_img], labels)):
                draw.text((j * (w + 4) + 2, 1), label, fill=(200, 200, 200))
                comp.paste(img, (j * (w + 4), 20))
            comp.save(RESULTS_DIR / f"comparison_{idx:03d}.png")

    print(f"\nDone! Results in {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
