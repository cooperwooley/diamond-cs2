"""
Generate CS2 console commands for automated frame capture from demo playback.

Usage:
    python scripts/capture_frames.py --demos data/demos/ --output data/frames/ --fps 16

This script generates .cfg files that can be executed in CS2's console to
automate demo playback and frame capture using the `startmovie` command.

Steps:
    1. Run this script to generate .cfg files
    2. Copy the .cfg files to your CS2/cfg directory
    3. In CS2 console, run: exec capture_demo_0
    4. The game will play the demo and save frames as TGA files
    5. Convert TGA files to the dataset format with build_dataset.py

Frame capture requires CS2 running on Windows with a display.
"""

import argparse
from pathlib import Path


def generate_cfg(demo_path: Path, output_dir: Path, fps: int, demo_idx: int) -> str:
    """Generate a CS2 .cfg file for automated demo playback and frame capture."""
    demo_name = demo_path.stem
    frame_output = output_dir / demo_name

    cfg_lines = [
        f"// Auto-generated capture config for {demo_name}",
        f"// Demo index: {demo_idx}",
        "",
        "// Set framerate to match target FPS",
        f"host_framerate {fps}",
        "",
        "// Disable HUD elements that might interfere",
        "// (Comment these out if you want HUD in the dataset)",
        "// cl_drawhud 0",
        "",
        "// Set resolution to match DIAMOND dataset (4:3 crop will be done in post)",
        "// Recommended: play at 1280x960 or 1024x768 for 4:3",
        "",
        "// Start demo playback",
        f'playdemo "{demo_path.name}"',
        "",
        "// Wait for demo to load (adjust if needed)",
        "wait 100",
        "",
        f'// Start recording frames to: {frame_output}',
        f'startmovie "{frame_output}" tga',
        "",
        "// The demo will play through and frames will be captured",
        "// When done, run 'endmovie' in console or it will auto-stop",
        "",
        f"// After capture, run: python scripts/build_dataset.py",
        f"// to process the TGA files into HDF5 format",
    ]

    return "\n".join(cfg_lines)


def main():
    parser = argparse.ArgumentParser(description="Generate CS2 frame capture configs")
    parser.add_argument("--demos", type=Path, required=True, help="Directory containing .dem files")
    parser.add_argument("--output", type=Path, default=Path("data/frames"), help="Output directory for captured frames")
    parser.add_argument("--fps", type=int, default=16, help="Target FPS for capture")
    parser.add_argument("--cfg-output", type=Path, default=None, help="Directory to save .cfg files (default: next to demos)")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    demo_files = sorted(args.demos.glob("*.dem"))
    if not demo_files:
        print(f"No .dem files found in {args.demos}")
        return

    cfg_dir = args.cfg_output or args.demos / "capture_configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(demo_files)} demo files")
    print(f"Target FPS: {args.fps}")
    print(f"Frame output: {args.output}")
    print()

    for i, demo_path in enumerate(demo_files):
        cfg_content = generate_cfg(demo_path, args.output, args.fps, i)
        cfg_path = cfg_dir / f"capture_demo_{i}.cfg"

        with open(cfg_path, "w") as f:
            f.write(cfg_content)

        print(f"  [{i+1}] {demo_path.name} -> {cfg_path.name}")

    # Generate master config that runs all demos
    master_lines = [
        "// Master capture config - runs all demos in sequence",
        "// Execute with: exec capture_all",
        "",
    ]
    for i in range(len(demo_files)):
        master_lines.append(f"exec capture_demo_{i}")
        master_lines.append("wait 5000")  # Wait between demos
        master_lines.append("")

    master_path = cfg_dir / "capture_all.cfg"
    with open(master_path, "w") as f:
        f.write("\n".join(master_lines))

    print(f"\nGenerated {len(demo_files)} capture configs in {cfg_dir}")
    print(f"Master config: {master_path}")
    print()
    print("Instructions:")
    print(f"  1. Copy {cfg_dir}/*.cfg to your CS2 cfg directory")
    print("     Usually: C:\\Program Files (x86)\\Steam\\steamapps\\common\\Counter-Strike Global Offensive\\game\\csgo\\cfg\\")
    print(f"  2. Copy demo files to CS2 directory or use full paths")
    print(f"  3. In CS2 console: exec capture_demo_0")
    print(f"  4. Wait for demo to finish, then: endmovie")
    print(f"  5. Frames will be saved as TGA files in {args.output}")
    print()
    print("Alternative (faster): Use HLAE (advancedfx.org) for automated batch capture")
    print()
    print("For screen capture alternative (no CS2 cfg needed):")
    print("  Use OBS Studio or similar to record demo playback at 16 FPS")
    print("  Then extract frames: ffmpeg -i recording.mp4 -vf fps=16 frames/%06d.png")


if __name__ == "__main__":
    main()
