"""Compute FID between an autoregressive rollout from the trained DIAMOND CS2
denoiser and the matching ground-truth segment of the test episode.

Outputs are written to /tmp/diamond_fid_{gen,real}/ as individual PNGs and the
FID score is printed at the end.

Usage:
    python scripts/compute_fid.py \
        --checkpoint results/cloud_training/agent_epoch_00149.pt \
        --num-frames 256 --denoising-steps 3
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hydra.utils import instantiate
from omegaconf import OmegaConf

from agent import Agent
from data import Dataset, CS2Hdf5Dataset
from models.diffusion import DiffusionSampler, DiffusionSamplerConfig
from pytorch_fid.fid_score import calculate_fid_given_paths


def load_cfg():
    cfg = OmegaConf.load(ROOT / "config" / "trainer.yaml")
    cfg = OmegaConf.merge(
        cfg,
        {
            "agent": OmegaConf.load(ROOT / "config" / "agent" / "cs2.yaml"),
            "env": OmegaConf.load(ROOT / "config" / "env" / "cs2.yaml"),
        },
    )
    OmegaConf.resolve(cfg)
    return cfg


def load_test_dataset():
    full_res = CS2Hdf5Dataset(ROOT / "data" / "processed" / "full_res")
    return Dataset(
        ROOT / "data" / "processed" / "low_res" / "test",
        full_res,
        "test_dataset",
        cache_in_ram=True,
    )


def tensor_to_pil(t):
    """[-1, 1] (C, H, W) tensor -> PIL Image."""
    img = t.permute(1, 2, 0).cpu().numpy()
    img = ((img + 1) / 2 * 255).clip(0, 255).astype("uint8")
    return Image.fromarray(img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--num-frames", type=int, default=256)
    parser.add_argument("--denoising-steps", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--gen-dir", type=Path, default=Path("/tmp/diamond_fid_gen"))
    parser.add_argument("--real-dir", type=Path, default=Path("/tmp/diamond_fid_real"))
    parser.add_argument("--batch-size", type=int, default=64, help="pytorch-fid Inception batch size")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"[setup] device={device}")

    for d in (args.gen_dir, args.real_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    cfg = load_cfg()
    agent_cfg = instantiate(cfg.agent, num_actions=cfg.env.num_actions)
    agent = Agent(agent_cfg).to(device)
    agent.load(args.checkpoint, load_denoiser=True, load_upsampler=False)
    agent.to(device).eval()
    print(f"[setup] loaded checkpoint {args.checkpoint}")

    test_dataset = load_test_dataset()
    episode = test_dataset.load_episode(0)
    n_cond = 4
    init_obs = episode.obs[:n_cond].unsqueeze(0).to(device)
    init_act = episode.act[:n_cond].unsqueeze(0).to(device)

    available_gt = len(episode.obs) - n_cond
    num_frames = min(args.num_frames, available_gt)
    print(f"[setup] generating {num_frames} frames "
          f"(test episode has {len(episode.obs)} frames, {available_gt} usable as GT)")

    sampler = DiffusionSampler(
        agent.denoiser.to(device),
        DiffusionSamplerConfig(
            num_steps_denoising=args.denoising_steps,
            sigma_min=2e-3,
            sigma_max=5.0,
            rho=7,
            order=1,
            s_churn=0.0,
        ),
    )

    obs = init_obs.clone()
    act = init_act.clone()
    t0 = time.time()
    with torch.no_grad():
        for i in tqdm(range(num_frames), desc="rolling out"):
            next_obs, _ = sampler.sample(obs, act)
            tensor_to_pil(next_obs.squeeze(0)).save(args.gen_dir / f"gen_{i:04d}.png")
            tensor_to_pil(episode.obs[n_cond + i]).save(args.real_dir / f"real_{i:04d}.png")
            obs = torch.cat([obs[:, 1:], next_obs.unsqueeze(1)], dim=1)
            next_act = torch.zeros(1, 1, act.size(-1), device=device)
            act = torch.cat([act[:, 1:], next_act], dim=1)
    rollout_s = time.time() - t0
    print(f"[rollout] {num_frames} frames in {rollout_s:.1f}s "
          f"({rollout_s / num_frames:.2f}s/frame)")

    print("[fid] computing FID via pytorch-fid (Inception v3)...")
    t1 = time.time()
    fid = calculate_fid_given_paths(
        [str(args.gen_dir), str(args.real_dir)],
        batch_size=args.batch_size,
        device=device,
        dims=2048,
        num_workers=2,
    )
    print(f"[fid] FID computed in {time.time() - t1:.1f}s")

    print()
    print("=" * 60)
    print(f"  FID (gen vs real, {num_frames} samples each): {fid:.3f}")
    print(f"  Resolution: 30x56 (denoiser native, Inception resizes to 299x299)")
    print(f"  Sampling:   {args.denoising_steps} denoising steps, autoregressive")
    print(f"  Test source: episode 0 of test split, frames {n_cond}..{n_cond + num_frames - 1}")
    print("=" * 60)


if __name__ == "__main__":
    main()
