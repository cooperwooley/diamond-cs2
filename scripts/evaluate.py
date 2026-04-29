"""
Evaluation script for DIAMOND CS2 trained model.

Generates all deliverables for the final report:
  1. Test loss evaluation across checkpoints (loss curve)
  2. Frame rollouts from the trained model
  3. Side-by-side comparison (generated vs ground truth)
  4. Temporal consistency visualization

Usage:
    # Full evaluation pipeline
    python scripts/evaluate.py --checkpoint results/cloud_training/agent_epoch_00149.pt

    # Just generate frames
    python scripts/evaluate.py --checkpoint results/cloud_training/agent_epoch_00149.pt --only frames

    # Evaluate loss across all checkpoints
    python scripts/evaluate.py --loss-curve

    # All of the above
    python scripts/evaluate.py --checkpoint results/cloud_training/agent_epoch_00149.pt --loss-curve
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# Add src to path for imports
SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from hydra.utils import instantiate
from omegaconf import OmegaConf

from agent import Agent, AgentConfig
from models.diffusion import (
    Denoiser,
    DenoiserConfig,
    DiffusionSampler,
    DiffusionSamplerConfig,
    InnerModelConfig,
    SigmaDistributionConfig,
)
from data import Dataset, CS2Hdf5Dataset, DatasetTraverser
from data.segment import SegmentId


ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"
CLOUD_DIR = RESULTS_DIR / "cloud_training"


def load_config():
    """Load the training config used for cloud training."""
    cfg = OmegaConf.load(ROOT / "config" / "trainer.yaml")

    # Load agent config
    agent_cfg = OmegaConf.load(ROOT / "config" / "agent" / "cs2.yaml")
    env_cfg = OmegaConf.load(ROOT / "config" / "env" / "cs2.yaml")

    # Merge
    cfg = OmegaConf.merge(cfg, {"agent": agent_cfg, "env": env_cfg})
    OmegaConf.resolve(cfg)
    return cfg


def build_agent(cfg, device):
    """Instantiate the agent from config."""
    agent_cfg = instantiate(cfg.agent, num_actions=cfg.env.num_actions)
    agent = Agent(agent_cfg).to(device)
    return agent


def load_checkpoint(agent, checkpoint_path, device):
    """Load weights from a checkpoint file."""
    agent.load(checkpoint_path, load_denoiser=True, load_upsampler=True)
    agent.to(device)
    agent.eval()
    return agent


def load_test_dataset(cfg):
    """Load the test dataset."""
    low_res_path = ROOT / "data" / "processed" / "low_res"
    full_res_path = ROOT / "data" / "processed" / "full_res"

    dataset_full_res = CS2Hdf5Dataset(full_res_path)
    test_dataset = Dataset(
        low_res_path / "test",
        dataset_full_res,
        "test_dataset",
        cache_in_ram=True,
    )
    test_dataset.load_from_default_path()
    return test_dataset


def evaluate_test_loss(agent, test_dataset, cfg, device):
    """Compute test loss (denoiser + upsampler) on the test set."""
    # Setup sigma distribution for training loss computation
    sigma_dist_cfg = instantiate(cfg.denoiser.sigma_distribution)
    sigma_dist_cfg_up = instantiate(cfg.upsampler.sigma_distribution) if agent.upsampler is not None else None
    agent.setup_training(sigma_dist_cfg, sigma_dist_cfg_up)

    agent.eval()

    # Denoiser test loss
    n_cond = cfg.agent.denoiser.inner_model.num_steps_conditioning
    n_ar = cfg.denoiser.training.num_autoregressive_steps
    seq_length = n_cond + 1 + n_ar
    batch_size = cfg.denoiser.training.batch_size

    traverser = DatasetTraverser(test_dataset, batch_size, seq_length)

    denoiser_losses = []
    for batch in tqdm(traverser, desc="Evaluating denoiser"):
        batch = batch.to(device)
        with torch.no_grad():
            loss, metrics = agent.denoiser(batch)
        denoiser_losses.append(metrics["loss_denoising"])

    avg_denoiser_loss = np.mean(denoiser_losses)

    # Upsampler test loss
    upsampler_losses = []
    if agent.upsampler is not None:
        n_cond_up = cfg.agent.upsampler.inner_model.num_steps_conditioning
        n_ar_up = cfg.upsampler.training.num_autoregressive_steps
        seq_length_up = n_cond_up + 1 + n_ar_up
        batch_size_up = cfg.upsampler.training.batch_size

        traverser_up = DatasetTraverser(test_dataset, batch_size_up, seq_length_up)
        for batch in tqdm(traverser_up, desc="Evaluating upsampler"):
            batch = batch.to(device)
            with torch.no_grad():
                loss, metrics = agent.upsampler(batch)
            upsampler_losses.append(metrics["loss_denoising"])

    avg_upsampler_loss = np.mean(upsampler_losses) if upsampler_losses else None

    return avg_denoiser_loss, avg_upsampler_loss


def generate_rollout(agent, test_dataset, num_frames=64, num_denoising_steps=3, device=torch.device("cpu")):
    """Generate a rollout of predicted frames using the diffusion sampler."""
    n_cond = 4  # num_steps_conditioning for denoiser

    # Get initial frames from the test set
    episode = test_dataset.load_episode(0)
    init_obs = episode.obs[:n_cond].unsqueeze(0).to(device)  # (1, 4, C, H, W)
    init_act = episode.act[:n_cond].unsqueeze(0).to(device)  # (1, 4, A)

    # Setup diffusion sampler
    sampler_cfg = DiffusionSamplerConfig(
        num_steps_denoising=num_denoising_steps,
        sigma_min=2e-3,
        sigma_max=5.0,
        rho=7,
        order=1,
        s_churn=0.0,
    )
    sampler = DiffusionSampler(agent.denoiser.to(device), sampler_cfg)

    generated_frames = []
    ground_truth_frames = []

    obs = init_obs.clone()
    act = init_act.clone()

    for i in tqdm(range(num_frames), desc="Generating frames"):
        # Generate next frame
        next_obs, trajectory = sampler.sample(obs, act)
        generated_frames.append(next_obs.squeeze(0).cpu())

        # Get ground truth for comparison
        gt_idx = n_cond + i
        if gt_idx < len(episode.obs):
            ground_truth_frames.append(episode.obs[gt_idx].cpu())

        # Shift observation window
        obs = torch.cat([obs[:, 1:], next_obs.unsqueeze(1)], dim=1)

        # Use zero actions (we don't have real actions)
        next_act = torch.zeros(1, 1, act.size(-1), device=device)
        act = torch.cat([act[:, 1:], next_act], dim=1)

    return generated_frames, ground_truth_frames


def tensor_to_image(t):
    """Convert a [-1, 1] tensor (C, H, W) to a PIL Image."""
    img = t.permute(1, 2, 0).numpy()
    img = ((img + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img)


def save_frame_grid(frames, output_path, cols=8, title=None):
    """Save a grid of frames as an image."""
    images = [tensor_to_image(f) for f in frames]
    h, w = images[0].size[1], images[0].size[0]
    rows = (len(images) + cols - 1) // cols

    grid = Image.new("RGB", (cols * w, rows * h), (0, 0, 0))
    for i, img in enumerate(images):
        r, c = i // cols, i % cols
        grid.paste(img, (c * w, r * h))

    grid.save(output_path)
    print(f"Saved frame grid ({len(frames)} frames, {rows}x{cols}) to {output_path}")


def save_comparison(generated, ground_truth, output_path, num_pairs=8):
    """Save side-by-side comparison: top row = generated, bottom row = ground truth."""
    n = min(num_pairs, len(generated), len(ground_truth))
    gen_imgs = [tensor_to_image(generated[i]) for i in range(n)]
    gt_imgs = [tensor_to_image(ground_truth[i]) for i in range(n)]

    h, w = gen_imgs[0].size[1], gen_imgs[0].size[0]
    label_h = 20
    grid = Image.new("RGB", (n * w, 2 * h + label_h), (0, 0, 0))

    for i in range(n):
        grid.paste(gen_imgs[i], (i * w, 0))
        grid.paste(gt_imgs[i], (i * w, h + label_h))

    grid.save(output_path)
    print(f"Saved comparison ({n} pairs) to {output_path}")


def save_temporal_strip(frames, output_path, every=4):
    """Save every Nth frame in a horizontal strip to show temporal progression."""
    selected = frames[::every]
    images = [tensor_to_image(f) for f in selected]
    h, w = images[0].size[1], images[0].size[0]

    strip = Image.new("RGB", (len(images) * w, h))
    for i, img in enumerate(images):
        strip.paste(img, (i * w, 0))

    strip.save(output_path)
    print(f"Saved temporal strip ({len(images)} frames, every {every}th) to {output_path}")


def plot_loss_curve(losses_by_epoch, output_path):
    """Plot and save the loss curve."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = sorted(losses_by_epoch.keys())
    denoiser_losses = [losses_by_epoch[e]["denoiser"] for e in epochs]

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(epochs, denoiser_losses, "b-o", label="Denoiser Test Loss", markersize=8, linewidth=2)

    if any(losses_by_epoch[e].get("upsampler") is not None for e in epochs):
        upsampler_losses = [losses_by_epoch[e]["upsampler"] for e in epochs if losses_by_epoch[e].get("upsampler") is not None]
        up_epochs = [e for e in epochs if losses_by_epoch[e].get("upsampler") is not None]
        ax.plot(up_epochs, upsampler_losses, "r-s", label="Upsampler Test Loss", markersize=8, linewidth=2)

    ax.set_xlabel("Epoch", fontsize=14)
    ax.set_ylabel("Test Loss (MSE)", fontsize=14)
    ax.set_title("DIAMOND CS2 — Training Loss Curve (150 epochs on L4 GPU)", fontsize=15)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved loss curve to {output_path}")


def run_loss_curve(cfg, device):
    """Evaluate test loss across all available checkpoints."""
    checkpoint_dirs = [
        CLOUD_DIR / "checkpoints",
        CLOUD_DIR,
    ]

    checkpoints = {}
    for d in checkpoint_dirs:
        if not d.exists():
            continue
        for f in d.glob("agent_epoch_*.pt"):
            epoch = int(f.stem.split("_")[-1])
            checkpoints[epoch] = f

    if not checkpoints:
        print("No checkpoints found for loss curve evaluation.")
        return {}

    print(f"Found {len(checkpoints)} checkpoints: epochs {sorted(checkpoints.keys())}")

    test_dataset = load_test_dataset(cfg)
    losses_by_epoch = {}

    for epoch in sorted(checkpoints.keys()):
        print(f"\n--- Evaluating epoch {epoch} ---")
        # Build fresh agent each time (setup_training can only be called once)
        agent = build_agent(cfg, device)
        load_checkpoint(agent, checkpoints[epoch], device)

        denoiser_loss, upsampler_loss = evaluate_test_loss(agent, test_dataset, cfg, device)
        losses_by_epoch[epoch] = {
            "denoiser": denoiser_loss,
            "upsampler": upsampler_loss,
        }
        print(f"  Denoiser loss: {denoiser_loss:.6f}")
        if upsampler_loss is not None:
            print(f"  Upsampler loss: {upsampler_loss:.6f}")

        # Free memory
        del agent
        torch.cuda.empty_cache()

    # Save results
    output_path = RESULTS_DIR / "cloud_training_losses.json"
    serializable = {str(k): v for k, v in losses_by_epoch.items()}
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nSaved loss data to {output_path}")

    # Plot
    if len(losses_by_epoch) >= 2:
        plot_loss_curve(losses_by_epoch, RESULTS_DIR / "cloud_loss_curve.png")

    return losses_by_epoch


def run_frame_generation(cfg, checkpoint_path, device):
    """Generate frames and save all visualizations."""
    print(f"\nLoading model from {checkpoint_path}...")
    agent = build_agent(cfg, device)
    load_checkpoint(agent, checkpoint_path, device)

    test_dataset = load_test_dataset(cfg)

    # Generate rollout
    print("\nGenerating 64-frame rollout (3 denoising steps)...")
    generated, ground_truth = generate_rollout(
        agent, test_dataset, num_frames=64, num_denoising_steps=3, device=device,
    )

    output_dir = RESULTS_DIR / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Frame grid of generated frames
    save_frame_grid(generated, output_dir / "generated_grid.png", cols=8)

    # 2. Frame grid of ground truth
    if ground_truth:
        save_frame_grid(ground_truth[:len(generated)], output_dir / "ground_truth_grid.png", cols=8)

    # 3. Side-by-side comparison
    if ground_truth:
        save_comparison(generated, ground_truth, output_dir / "comparison.png", num_pairs=8)

    # 4. Temporal strips
    save_temporal_strip(generated, output_dir / "temporal_strip_generated.png", every=4)
    if ground_truth:
        save_temporal_strip(ground_truth, output_dir / "temporal_strip_ground_truth.png", every=4)

    # 5. Also generate a fast rollout (1 step) for comparison
    print("\nGenerating 32-frame rollout (1 denoising step, fast mode)...")
    generated_fast, _ = generate_rollout(
        agent, test_dataset, num_frames=32, num_denoising_steps=1, device=device,
    )
    save_frame_grid(generated_fast, output_dir / "generated_fast_grid.png", cols=8)

    # 6. Save individual frames for the report
    for i in [0, 7, 15, 31]:
        if i < len(generated):
            tensor_to_image(generated[i]).save(output_dir / f"generated_frame_{i:03d}.png")
        if i < len(ground_truth):
            tensor_to_image(ground_truth[i]).save(output_dir / f"ground_truth_frame_{i:03d}.png")

    print(f"\nAll generated frames saved to {output_dir}/")
    return generated, ground_truth


def main():
    parser = argparse.ArgumentParser(description="Evaluate DIAMOND CS2 model")
    parser.add_argument("--checkpoint", type=Path, help="Path to agent checkpoint for frame generation")
    parser.add_argument("--loss-curve", action="store_true", help="Evaluate loss across all checkpoints")
    parser.add_argument("--only", choices=["frames", "loss"], help="Only run one evaluation type")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-frames", type=int, default=64)
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    cfg = load_config()

    if args.only == "loss" or args.loss_curve:
        run_loss_curve(cfg, device)

    if args.only == "frames" or args.checkpoint:
        if args.checkpoint is None:
            # Default to the final checkpoint
            args.checkpoint = CLOUD_DIR / "agent_epoch_00149.pt"
        if not args.checkpoint.exists():
            print(f"Checkpoint not found: {args.checkpoint}")
            return
        run_frame_generation(cfg, args.checkpoint, device)

    if not args.checkpoint and not args.loss_curve and not args.only:
        print("No action specified. Use --checkpoint, --loss-curve, or --only.")
        print("Example: python scripts/evaluate.py --checkpoint results/cloud_training/agent_epoch_00149.pt --loss-curve")


if __name__ == "__main__":
    main()
