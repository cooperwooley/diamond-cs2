"""
Build the CS2 HDF5 dataset from captured frames and parsed actions.

Usage:
    python scripts/build_dataset.py \
        --frames data/frames/ \
        --actions data/parsed/ \
        --output data/processed/ \
        --target-size 150 280

Creates HDF5 files with 1000 frames each (matching DIAMOND format),
then generates low-resolution (30x56) copies for training.
"""

import argparse
import random
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
import torchvision.transforms.functional as T
from tqdm import tqdm

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.dataset import Dataset, CS2Hdf5Dataset
from data.episode import Episode
from data.segment import SegmentId


FRAMES_PER_EPISODE = 1000


def load_frames_from_directory(frame_dir: Path, target_h: int, target_w: int) -> list:
    """Load and resize frames from a directory of image files."""
    extensions = {".tga", ".png", ".jpg", ".jpeg", ".bmp"}
    frame_files = sorted([
        f for f in frame_dir.iterdir()
        if f.suffix.lower() in extensions
    ])

    frames = []
    for f in frame_files:
        img = cv2.imread(str(f))
        if img is None:
            continue
        # Resize to target resolution (height, width)
        img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
        frames.append(img)  # BGR format, matching DIAMOND convention

    return frames


def create_hdf5_episode(
    frames: list,
    actions: np.ndarray,
    output_path: Path,
    episode_idx: int,
) -> bool:
    """Create a single HDF5 file with 1000 frames."""
    n_frames = min(len(frames), len(actions), FRAMES_PER_EPISODE)
    if n_frames < FRAMES_PER_EPISODE:
        print(f"  Warning: only {n_frames} frames available for episode {episode_idx}")
        if n_frames < 100:
            return False

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as f:
        for i in range(n_frames):
            # Store frame as BGR uint8 (matching DIAMOND CS:GO format)
            f.create_dataset(f"frame_{i}_x", data=frames[i], dtype=np.uint8)
            # Store action as float32 (51-dim vector)
            f.create_dataset(f"frame_{i}_y", data=actions[i].astype(np.float32))

    return True


def create_low_res_dataset(
    full_res_dir: Path,
    low_res_dir: Path,
    test_ratio: float = 0.1,
):
    """Create low-resolution dataset from full-res HDF5 files and split train/test."""
    print("\nCreating low-resolution dataset...")

    cs2_dataset = CS2Hdf5Dataset(full_res_dir)
    print(f"Found {cs2_dataset.num_episodes} episodes")

    if cs2_dataset.num_episodes == 0:
        print("No episodes found!")
        return

    # Determine train/test split
    all_ids = list(cs2_dataset._filenames.keys())
    random.shuffle(all_ids)
    n_test = max(1, int(len(all_ids) * test_ratio))
    test_ids = set(all_ids[:n_test])

    train_dataset = Dataset(low_res_dir / "train", None)
    test_dataset = Dataset(low_res_dir / "test", None)

    for file_id in tqdm(cs2_dataset._filenames, desc="Creating low_res"):
        episode = Episode(
            **{
                k: v
                for k, v in cs2_dataset[SegmentId(file_id, 0, FRAMES_PER_EPISODE)].__dict__.items()
                if k not in ("mask_padding", "id")
            }
        )
        # Downsample to 30x56 (matching DIAMOND)
        episode.obs = T.resize(
            episode.obs, (30, 56), interpolation=T.InterpolationMode.BICUBIC
        )
        filename = cs2_dataset._filenames[file_id]
        episode.info = {"original_file_id": file_id}
        dataset = test_dataset if file_id in test_ids else train_dataset
        dataset.add_episode(episode)

    train_dataset.save_to_default_path()
    test_dataset.save_to_default_path()

    print(f"Train: {train_dataset.num_episodes} episodes ({train_dataset.num_steps} steps)")
    print(f"Test: {test_dataset.num_episodes} episodes ({test_dataset.num_steps} steps)")

    return low_res_dir


def main():
    parser = argparse.ArgumentParser(description="Build CS2 HDF5 dataset")
    parser.add_argument("--frames", type=Path, required=True, help="Directory with captured frames (subdirs per demo)")
    parser.add_argument("--actions", type=Path, required=True, help="Directory with parsed action data")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for processed dataset")
    parser.add_argument("--target-height", type=int, default=150, help="Target frame height")
    parser.add_argument("--target-width", type=int, default=280, help="Target frame width")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Fraction of episodes for test set")
    args = parser.parse_args()

    full_res_dir = args.output / "full_res"
    low_res_dir = args.output / "low_res"
    full_res_dir.mkdir(parents=True, exist_ok=True)

    # Find frame directories (one per demo/player)
    frame_dirs = sorted([d for d in args.frames.iterdir() if d.is_dir()])
    if not frame_dirs:
        # Check if frames are directly in the directory
        frame_files = list(args.frames.glob("*.tga")) + list(args.frames.glob("*.png"))
        if frame_files:
            frame_dirs = [args.frames]
        else:
            print(f"No frame directories found in {args.frames}")
            return

    # Find action files
    action_dirs = sorted([d for d in args.actions.iterdir() if d.is_dir()])

    print(f"Frame directories: {len(frame_dirs)}")
    print(f"Action directories: {len(action_dirs)}")
    print(f"Target resolution: {args.target_height}x{args.target_width}")

    episode_idx = 0
    total_frames = 0

    for frame_dir in tqdm(frame_dirs, desc="Building episodes"):
        demo_name = frame_dir.name

        # Find matching actions
        action_dir = args.actions / demo_name
        if not action_dir.exists():
            print(f"  No actions found for {demo_name}, skipping")
            continue

        # Load frames
        frames = load_frames_from_directory(frame_dir, args.target_height, args.target_width)
        if len(frames) < 100:
            print(f"  Too few frames ({len(frames)}) in {demo_name}, skipping")
            continue

        # Find player action files
        player_dirs = sorted([d for d in action_dir.iterdir() if d.is_dir()])
        if not player_dirs:
            # Try loading actions directly
            action_file = action_dir / "actions.npy"
            if action_file.exists():
                player_dirs = [action_dir]
            else:
                print(f"  No action files found for {demo_name}")
                continue

        for player_dir in player_dirs:
            action_file = player_dir / "actions.npy"
            if not action_file.exists():
                continue

            actions = np.load(action_file)

            # Split into episodes of FRAMES_PER_EPISODE
            n_episodes = min(len(frames), len(actions)) // FRAMES_PER_EPISODE

            for ep in range(n_episodes):
                start = ep * FRAMES_PER_EPISODE
                end = start + FRAMES_PER_EPISODE

                ep_frames = frames[start:end]
                ep_actions = actions[start:end]

                # Create episode subdirectory (matching DIAMOND structure)
                subdir_name = f"ep_{episode_idx:05d}"
                ep_dir = full_res_dir / subdir_name
                ep_dir.mkdir(exist_ok=True)

                hdf5_path = ep_dir / f"cs2_episode_{episode_idx}.hdf5"

                if create_hdf5_episode(ep_frames, ep_actions, hdf5_path, episode_idx):
                    episode_idx += 1
                    total_frames += FRAMES_PER_EPISODE

    print(f"\n{'='*60}")
    print(f"Created {episode_idx} episodes ({total_frames} total frames)")
    print(f"Full-res HDF5 files in: {full_res_dir}")

    if episode_idx > 0:
        # Create low-res dataset
        create_low_res_dataset(full_res_dir, low_res_dir, args.test_ratio)

        print(f"\nDataset ready! Update config/env/cs2.yaml:")
        print(f"  path_data_low_res: {low_res_dir.absolute()}")
        print(f"  path_data_full_res: {full_res_dir.absolute()}")
    else:
        print("\nNo episodes created. Check your frame/action directories.")


if __name__ == "__main__":
    main()
