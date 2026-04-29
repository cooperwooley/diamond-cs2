"""
Single-step frame prediction evaluation.

Instead of autoregressive rollouts (which degrade without actions),
this evaluates the model's ability to predict the NEXT frame given
4 real conditioning frames. This is the fairest evaluation for a
model trained with zero-action labels.

Generates:
  - Single-step predictions vs ground truth across multiple test episodes
  - Quality comparison at different test set positions
  - Upsampled (high-res) frame comparisons
"""

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
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
RESULTS_DIR = ROOT / "results" / "single_step"


def load_all():
    cfg = OmegaConf.load(ROOT / "config" / "trainer.yaml")
    agent_cfg = OmegaConf.load(ROOT / "config" / "agent" / "cs2.yaml")
    env_cfg = OmegaConf.load(ROOT / "config" / "env" / "cs2.yaml")
    cfg = OmegaConf.merge(cfg, {"agent": agent_cfg, "env": env_cfg})
    OmegaConf.resolve(cfg)
    return cfg


def tensor_to_image(t):
    img = t.permute(1, 2, 0).numpy()
    img = ((img + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img)


def make_labeled_row(images, labels, cell_w, cell_h, label_h=18):
    """Create a row of images with labels above each."""
    row = Image.new("RGB", (len(images) * cell_w, cell_h + label_h), (0, 0, 0))
    draw = ImageDraw.Draw(row)
    for i, (img, label) in enumerate(zip(images, labels)):
        # Label
        draw.text((i * cell_w + 2, 1), label, fill=(255, 255, 255))
        # Image
        row.paste(img.resize((cell_w, cell_h), Image.NEAREST), (i * cell_w, label_h))
    return row


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "results" / "cloud_training" / "agent_epoch_00149.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-samples", type=int, default=12, help="Number of test samples to evaluate")
    parser.add_argument("--denoising-steps", type=int, default=3)
    args = parser.parse_args()

    device = torch.device(args.device)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    cfg = load_all()

    # Build and load model
    agent_cfg = instantiate(cfg.agent, num_actions=cfg.env.num_actions)
    agent = Agent(agent_cfg).to(device)
    agent.load(args.checkpoint, load_denoiser=True, load_upsampler=True)
    agent.eval()

    # Load test data
    low_res_path = ROOT / "data" / "processed" / "low_res"
    full_res_path = ROOT / "data" / "processed" / "full_res"
    dataset_full_res = CS2Hdf5Dataset(full_res_path)
    test_dataset = Dataset(low_res_path / "test", dataset_full_res, "test_dataset", cache_in_ram=True)
    test_dataset.load_from_default_path()

    # Setup sampler
    sampler = DiffusionSampler(
        agent.denoiser.to(device),
        DiffusionSamplerConfig(num_steps_denoising=args.denoising_steps, sigma_min=2e-3, sigma_max=5.0),
    )

    n_cond = 4
    episode = test_dataset.load_episode(0)
    num_frames = len(episode.obs)

    # Sample positions spread across the episode
    positions = np.linspace(0, num_frames - n_cond - 2, args.num_samples, dtype=int)

    all_pred_imgs = []
    all_gt_imgs = []
    all_cond_imgs = []

    print(f"Generating {args.num_samples} single-step predictions...")
    for pos in tqdm(positions):
        # Get conditioning frames and ground truth
        cond_obs = episode.obs[pos:pos + n_cond].unsqueeze(0).to(device)
        cond_act = episode.act[pos:pos + n_cond].unsqueeze(0).to(device)
        gt_obs = episode.obs[pos + n_cond]

        # Predict next frame
        with torch.no_grad():
            pred_obs, _ = sampler.sample(cond_obs, cond_act)

        pred_img = tensor_to_image(pred_obs.squeeze(0).cpu())
        gt_img = tensor_to_image(gt_obs.cpu())
        cond_img = tensor_to_image(cond_obs[0, -1].cpu())  # Last conditioning frame

        all_pred_imgs.append(pred_img)
        all_gt_imgs.append(gt_img)
        all_cond_imgs.append(cond_img)

    # === Save results ===

    # 1. Big comparison grid: condition | predicted | ground truth
    w, h = all_pred_imgs[0].size
    scale = 4  # Upscale for visibility
    cell_w, cell_h = w * scale, h * scale
    label_h = 18
    row_h = cell_h + label_h

    grid = Image.new("RGB", (3 * cell_w, len(positions) * row_h + label_h), (30, 30, 30))
    draw = ImageDraw.Draw(grid)

    # Column headers
    headers = ["Last Context Frame", "Predicted (1-step)", "Ground Truth"]
    for i, header in enumerate(headers):
        draw.text((i * cell_w + 2, 1), header, fill=(200, 200, 200))

    for row, (cond, pred, gt) in enumerate(zip(all_cond_imgs, all_pred_imgs, all_gt_imgs)):
        y = label_h + row * row_h
        for col, img in enumerate([cond, pred, gt]):
            grid.paste(img.resize((cell_w, cell_h), Image.NEAREST), (col * cell_w, y))

    grid.save(RESULTS_DIR / "single_step_comparison.png")
    print(f"Saved single-step comparison grid to {RESULTS_DIR / 'single_step_comparison.png'}")

    # 2. Side-by-side strips (predicted vs ground truth)
    strip_h = cell_h
    n = min(8, len(all_pred_imgs))

    pred_strip = Image.new("RGB", (n * cell_w, strip_h))
    gt_strip = Image.new("RGB", (n * cell_w, strip_h))
    for i in range(n):
        pred_strip.paste(all_pred_imgs[i].resize((cell_w, cell_h), Image.NEAREST), (i * cell_w, 0))
        gt_strip.paste(all_gt_imgs[i].resize((cell_w, cell_h), Image.NEAREST), (i * cell_w, 0))

    # Combined with labels
    combined = Image.new("RGB", (n * cell_w, 2 * strip_h + 2 * label_h), (0, 0, 0))
    draw = ImageDraw.Draw(combined)
    draw.text((2, 1), "Predicted (single-step, 3 denoising steps)", fill=(200, 200, 200))
    combined.paste(pred_strip, (0, label_h))
    draw.text((2, label_h + strip_h + 1), "Ground Truth", fill=(200, 200, 200))
    combined.paste(gt_strip, (0, 2 * label_h + strip_h))
    combined.save(RESULTS_DIR / "pred_vs_gt_strip.png")
    print(f"Saved prediction vs GT strip to {RESULTS_DIR / 'pred_vs_gt_strip.png'}")

    # 3. Individual high-quality pairs for the report
    for i in [0, len(positions) // 4, len(positions) // 2, 3 * len(positions) // 4]:
        if i < len(all_pred_imgs):
            pair = Image.new("RGB", (2 * cell_w + 4, cell_h), (0, 0, 0))
            pair.paste(all_pred_imgs[i].resize((cell_w, cell_h), Image.NEAREST), (0, 0))
            pair.paste(all_gt_imgs[i].resize((cell_w, cell_h), Image.NEAREST), (cell_w + 4, 0))
            pair.save(RESULTS_DIR / f"pair_{i:03d}.png")

    # 4. Save raw individual frames at original resolution too
    for i in range(min(4, len(all_pred_imgs))):
        all_pred_imgs[i].save(RESULTS_DIR / f"pred_{i:03d}.png")
        all_gt_imgs[i].save(RESULTS_DIR / f"gt_{i:03d}.png")

    print(f"\nDone! All single-step evaluation results in {RESULTS_DIR}/")
    print(f"Key files:")
    print(f"  single_step_comparison.png  — Full grid (context | predicted | ground truth)")
    print(f"  pred_vs_gt_strip.png        — Side-by-side strip for report")
    print(f"  pair_*.png                  — Individual comparison pairs")


if __name__ == "__main__":
    main()
