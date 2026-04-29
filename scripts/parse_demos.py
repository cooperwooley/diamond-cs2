"""
Parse CS2 demo files to extract per-tick game state and actions using demoparser2.

Usage:
    python scripts/parse_demos.py --input data/demos/ --output data/parsed/

Extracts tick-by-tick player state (position, view angles, buttons) and
encodes actions into the 51-dim format matching DIAMOND's action space.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

try:
    from demoparser2 import DemoParser
except ImportError:
    print("demoparser2 not installed. Run: pip install demoparser2")
    exit(1)


# DIAMOND action space discretization
MOUSE_X_POSSIBLES = [
    -1000, -500, -300, -200, -100, -60, -30, -20, -10, -4, -2,
    0,
    2, 4, 10, 20, 30, 60, 100, 200, 300, 500, 1000,
]
MOUSE_Y_POSSIBLES = [
    -200, -100, -50, -20, -10, -4, -2,
    0,
    2, 4, 10, 20, 50, 100, 200,
]

N_KEYS = 11
N_CLICKS = 2
N_MOUSE_X = len(MOUSE_X_POSSIBLES)
N_MOUSE_Y = len(MOUSE_Y_POSSIBLES)
N_TOTAL = N_KEYS + N_CLICKS + N_MOUSE_X + N_MOUSE_Y  # 51


def discretize_mouse(value: float, possibles: list) -> int:
    """Find the index of the closest discretized value."""
    value = np.clip(value, possibles[0], possibles[-1])
    return min(range(len(possibles)), key=lambda i: abs(possibles[i] - value))


def encode_tick_action(row: dict, prev_yaw: float, prev_pitch: float) -> np.ndarray:
    """Encode a single tick's game state into the 51-dim action vector."""
    action = np.zeros(N_TOTAL, dtype=np.float32)

    # Keyboard keys (11 multi-hot)
    button_map = {
        0: "FORWARD",
        1: "LEFT",
        2: "BACK",
        3: "RIGHT",
        4: "JUMP",    # maps to SPACE
        5: "DUCK",    # maps to CTRL
        6: "WALK",    # maps to SHIFT
    }

    for idx, btn_name in button_map.items():
        if row.get(btn_name, False):
            action[idx] = 1.0

    # Weapon slots (indices 7, 8, 9) - inferred from active weapon changes
    # These would need to be derived from weapon switch events
    # For now, leave as 0 (handled in build_dataset.py if needed)

    # Reload (index 10)
    if row.get("RELOAD", False):
        action[10] = 1.0

    # Mouse clicks (indices 11, 12)
    if row.get("FIRE", False) or row.get("ATTACK", False):
        action[N_KEYS] = 1.0
    if row.get("RIGHTCLICK", False) or row.get("ATTACK2", False):
        action[N_KEYS + 1] = 1.0

    # Mouse movement from view angle deltas
    yaw = row.get("yaw", 0.0)
    pitch = row.get("pitch", 0.0)

    yaw_delta = yaw - prev_yaw
    pitch_delta = pitch - prev_pitch

    # Handle yaw wraparound (0-360)
    if yaw_delta > 180:
        yaw_delta -= 360
    elif yaw_delta < -180:
        yaw_delta += 360

    # Scale to mouse units (approximate conversion)
    mouse_x_raw = yaw_delta * 10.0
    mouse_y_raw = pitch_delta * 10.0

    # Discretize
    mx_idx = discretize_mouse(mouse_x_raw, MOUSE_X_POSSIBLES)
    my_idx = discretize_mouse(mouse_y_raw, MOUSE_Y_POSSIBLES)

    action[N_KEYS + N_CLICKS + mx_idx] = 1.0
    action[N_KEYS + N_CLICKS + N_MOUSE_X + my_idx] = 1.0

    return action


def parse_demo(demo_path: Path, target_fps: int = 16) -> Dict:
    """Parse a single demo file and extract player data at target FPS."""
    print(f"  Parsing: {demo_path.name}")

    parser = DemoParser(str(demo_path))

    # Fields to extract
    fields = [
        "X", "Y", "Z",
        "pitch", "yaw",
        "health", "is_alive",
        "FORWARD", "BACK", "LEFT", "RIGHT",
        "FIRE", "RIGHTCLICK", "RELOAD", "WALK",
    ]

    # Try to parse - some fields may not be available
    try:
        df = parser.parse_ticks(fields)
    except Exception as e:
        print(f"    Error parsing ticks: {e}")
        # Try with minimal fields
        try:
            fields = ["X", "Y", "Z", "pitch", "yaw", "health", "is_alive"]
            df = parser.parse_ticks(fields)
        except Exception as e2:
            print(f"    Failed to parse demo: {e2}")
            return None

    if df is None or len(df) == 0:
        print("    No data extracted")
        return None

    # Get unique player names/steamids
    if "name" in df.columns:
        players = df["name"].unique().tolist()
    elif "steamid" in df.columns:
        players = df["steamid"].unique().tolist()
    else:
        print("    No player identifier column found")
        return None

    # Get tick rate from demo
    try:
        header = parser.parse_header()
        tickrate = header.get("tickrate", 64)
    except Exception:
        tickrate = 64

    # Subsample to target FPS
    tick_interval = max(1, tickrate // target_fps)

    print(f"    Tick rate: {tickrate}, subsampling every {tick_interval} ticks -> {target_fps} fps")
    print(f"    Players found: {len(players)}")
    print(f"    Total rows: {len(df)}")

    result = {
        "tickrate": tickrate,
        "target_fps": target_fps,
        "tick_interval": tick_interval,
        "players": {},
    }

    for player in players:
        player_df = df[df["name" if "name" in df.columns else "steamid"] == player]

        if len(player_df) < 100:
            continue

        # Get unique ticks and subsample
        if "tick" in player_df.columns:
            ticks = sorted(player_df["tick"].unique())
        else:
            ticks = list(range(len(player_df)))

        sampled_ticks = ticks[::tick_interval]

        # Extract data at sampled ticks
        actions = []
        positions = []
        prev_yaw = 0.0
        prev_pitch = 0.0

        for tick in sampled_ticks:
            if "tick" in player_df.columns:
                tick_data = player_df[player_df["tick"] == tick]
            else:
                tick_data = player_df.iloc[tick:tick+1]

            if len(tick_data) == 0:
                continue

            row = tick_data.iloc[0].to_dict() if hasattr(tick_data, 'iloc') else tick_data

            # Skip if dead
            if row.get("is_alive", True) == False:
                prev_yaw = row.get("yaw", prev_yaw)
                prev_pitch = row.get("pitch", prev_pitch)
                continue

            # Encode action
            action = encode_tick_action(row, prev_yaw, prev_pitch)
            actions.append(action)

            # Store position for metadata
            positions.append({
                "x": float(row.get("X", 0)),
                "y": float(row.get("Y", 0)),
                "z": float(row.get("Z", 0)),
                "yaw": float(row.get("yaw", 0)),
                "pitch": float(row.get("pitch", 0)),
                "tick": int(tick),
            })

            prev_yaw = row.get("yaw", prev_yaw)
            prev_pitch = row.get("pitch", prev_pitch)

        if len(actions) > 100:
            result["players"][str(player)] = {
                "actions": np.array(actions),
                "positions": positions,
                "num_frames": len(actions),
            }
            print(f"    Player {player}: {len(actions)} frames extracted")

    return result


def main():
    parser = argparse.ArgumentParser(description="Parse CS2 demos for action extraction")
    parser.add_argument("--input", type=Path, required=True, help="Directory containing .dem files")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for parsed data")
    parser.add_argument("--fps", type=int, default=16, help="Target frames per second (default: 16)")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    demo_files = list(args.input.glob("*.dem"))
    if not demo_files:
        # Also check for compressed files
        gz_files = list(args.input.glob("*.dem.gz"))
        if gz_files:
            print(f"Found {len(gz_files)} compressed demos. Decompress first:")
            print(f"  cd {args.input} && gunzip *.dem.gz")
            return
        print(f"No .dem files found in {args.input}")
        return

    print(f"Found {len(demo_files)} demo files")

    total_frames = 0
    for i, demo_path in enumerate(sorted(demo_files)):
        print(f"\n[{i+1}/{len(demo_files)}] {demo_path.name}")

        result = parse_demo(demo_path, target_fps=args.fps)
        if result is None:
            continue

        # Save parsed data
        demo_name = demo_path.stem
        demo_output_dir = args.output / demo_name
        demo_output_dir.mkdir(parents=True, exist_ok=True)

        for player_id, player_data in result["players"].items():
            player_dir = demo_output_dir / f"player_{player_id}"
            player_dir.mkdir(exist_ok=True)

            # Save actions as numpy array
            np.save(player_dir / "actions.npy", player_data["actions"])

            # Save positions/metadata as JSON
            with open(player_dir / "metadata.json", "w") as f:
                json.dump({
                    "player_id": player_id,
                    "num_frames": player_data["num_frames"],
                    "tickrate": result["tickrate"],
                    "target_fps": result["target_fps"],
                    "positions": player_data["positions"],
                }, f, indent=2)

            total_frames += player_data["num_frames"]

    print(f"\n{'='*60}")
    print(f"Parsing complete!")
    print(f"Total frames extracted: {total_frames}")
    print(f"Output saved to: {args.output}")


if __name__ == "__main__":
    main()
