"""
Automated CS2 dataset builder from YouTube POV videos + HLTV demos.

Pipeline:
1. Scrape YouTube POV channel for Dust 2 videos
2. Parse video titles for player/team/tournament metadata
3. Download videos with yt-dlp, extract frames at 16fps with ffmpeg
4. Download matching demos from HLTV
5. Parse demos with demoparser2 for the POV player's actions
6. Align frames to actions and build HDF5 dataset

Usage:
    python scripts/auto_dataset.py --max-videos 10 --output data/
    python scripts/auto_dataset.py --list-only  # just show available videos
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


def get_ffmpeg_path() -> str:
    """Find ffmpeg binary - prefer system install, fall back to imageio-ffmpeg."""
    # Check system PATH first
    for path in os.environ.get("PATH", "").split(os.pathsep):
        ffmpeg = os.path.join(path, "ffmpeg")
        if os.path.isfile(ffmpeg) and os.access(ffmpeg, os.X_OK):
            return ffmpeg
    # Fall back to imageio-ffmpeg bundled binary
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"  # hope for the best

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ──────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────

@dataclass
class VideoMeta:
    video_id: str
    title: str
    duration: float
    player: str = ""
    team1: str = ""
    team2: str = ""
    map_name: str = ""
    tournament: str = ""
    score: str = ""  # e.g. "23/10"
    url: str = ""

    def __post_init__(self):
        self.url = f"https://www.youtube.com/watch?v={self.video_id}"


# ──────────────────────────────────────────────
# Step 1: Find Dust 2 POV videos
# ──────────────────────────────────────────────

YOUTUBE_POV_CHANNELS = [
    "https://www.youtube.com/@CSGOPOVDemosHighlights/videos",
]


def parse_pov_title(title: str) -> dict:
    """Parse a POV video title like:
    'CS2 POV Demo NaVi b1t (23/10) vs G2 (de_dust2) @ ESL Pro League Season 23'
    """
    result = {
        "player": "",
        "team1": "",
        "team2": "",
        "map_name": "",
        "tournament": "",
        "score": "",
    }

    # Extract map name
    map_match = re.search(r'\(?(de_\w+|dust2)\)?', title, re.IGNORECASE)
    if map_match:
        result["map_name"] = map_match.group(0).strip("()")

    # Extract tournament (after @)
    tourney_match = re.search(r'@\s*(.+)$', title)
    if tourney_match:
        result["tournament"] = tourney_match.group(1).strip()

    # Extract score like (23/10)
    score_match = re.search(r'\((\d+/\d+)\)', title)
    if score_match:
        result["score"] = score_match.group(1)

    # Extract player and teams
    # Pattern: "CS2 POV Demo <Team> <Player> (<score>) vs <Team2>"
    team_player_match = re.search(
        r'(?:CS2?\s+)?POV\s+Demo\s+(\S+)\s+(\S+)\s+\(\d+/\d+\)\s+vs\s+(\S+)',
        title, re.IGNORECASE
    )
    if team_player_match:
        result["team1"] = team_player_match.group(1)
        result["player"] = team_player_match.group(2)
        result["team2"] = team_player_match.group(3)
    else:
        # Try alternate pattern without score
        alt_match = re.search(
            r'(?:CS2?\s+)?POV\s+Demo\s+(\S+)\s+(\S+)\s+vs\s+(\S+)',
            title, re.IGNORECASE
        )
        if alt_match:
            result["team1"] = alt_match.group(1)
            result["player"] = alt_match.group(2)
            result["team2"] = alt_match.group(3)

    return result


def find_dust2_videos(max_videos: int = 50, max_scan: int = 500) -> List[VideoMeta]:
    """Scan YouTube POV channels for Dust 2 videos."""
    videos = []

    for channel_url in YOUTUBE_POV_CHANNELS:
        print(f"Scanning: {channel_url}")
        try:
            result = subprocess.run(
                [
                    "yt-dlp", "--flat-playlist",
                    "--print", "%(id)s|||%(title)s|||%(duration)s",
                    channel_url,
                    "--playlist-end", str(max_scan),
                ],
                capture_output=True, text=True, timeout=120,
            )

            for line in result.stdout.strip().split("\n"):
                if not line.strip():
                    continue

                parts = line.split("|||")
                if len(parts) != 3:
                    continue

                video_id, title, duration = parts
                duration = float(duration) if duration != "NA" else 0

                # Filter for Dust 2
                if not re.search(r'dust\s*2|de_dust2', title, re.IGNORECASE):
                    continue

                # Parse metadata from title
                meta = parse_pov_title(title)
                video = VideoMeta(
                    video_id=video_id,
                    title=title,
                    duration=duration,
                    **meta,
                )
                videos.append(video)

                if len(videos) >= max_videos:
                    break

        except subprocess.TimeoutExpired:
            print("  Timeout scanning channel")
        except Exception as e:
            print(f"  Error: {e}")

    print(f"Found {len(videos)} Dust 2 POV videos")
    return videos


# ──────────────────────────────────────────────
# Step 2: Download video and extract frames
# ──────────────────────────────────────────────

def download_and_extract_frames(
    video: VideoMeta,
    output_dir: Path,
    fps: int = 16,
    target_height: int = 150,
    target_width: int = 280,
) -> Optional[Path]:
    """Download a YouTube video and extract frames at target FPS and resolution."""
    frames_dir = output_dir / "frames" / sanitize_filename(video)
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Check if already processed
    existing = list(frames_dir.glob("*.png"))
    if len(existing) > 100:
        print(f"  Frames already exist ({len(existing)} frames), skipping download")
        return frames_dir

    video_path = output_dir / "videos" / f"{video.video_id}.mp4"
    video_path.parent.mkdir(parents=True, exist_ok=True)

    # Download video - check for existing files (yt-dlp may add format codes)
    existing_videos = list(video_path.parent.glob(f"{video.video_id}*"))
    existing_videos = [f for f in existing_videos if f.suffix in ('.mp4', '.mkv', '.webm') and '.part' not in f.name]

    if existing_videos:
        video_path = existing_videos[0]
        print(f"  Video already downloaded: {video_path.name}")
    else:
        print(f"  Downloading: {video.title}")
        try:
            subprocess.run(
                [
                    "yt-dlp",
                    "-f", "bestvideo[height<=1080]+bestaudio/best[height<=1080]/best",
                    "-o", str(video_path),
                    "--merge-output-format", "mp4",
                    "--no-playlist",
                    video.url,
                ],
                check=True, timeout=900,
                capture_output=True, text=True,
            )
            # yt-dlp may save with different name, find it
            existing_videos = list(video_path.parent.glob(f"{video.video_id}*"))
            existing_videos = [f for f in existing_videos if f.suffix in ('.mp4', '.mkv', '.webm') and '.part' not in f.name]
            if existing_videos:
                video_path = existing_videos[0]
            elif not video_path.exists():
                print("  Download succeeded but can't find output file")
                return None
        except subprocess.CalledProcessError as e:
            print(f"  Download failed: {e.stderr[:200] if e.stderr else 'unknown error'}")
            return None
        except subprocess.TimeoutExpired:
            print("  Download timed out")
            return None

    # Extract frames with ffmpeg
    ffmpeg = get_ffmpeg_path()
    print(f"  Extracting frames at {fps}fps, {target_width}x{target_height}")
    try:
        result = subprocess.run(
            [
                ffmpeg, "-i", str(video_path),
                "-vf", f"fps={fps},scale={target_width}:{target_height}",
                "-q:v", "2",  # high quality PNG
                str(frames_dir / "frame_%06d.png"),
                "-y",  # overwrite
            ],
            timeout=1800,
            capture_output=True, text=True,
        )
        # ffmpeg prints info to stderr even on success, so check return code
        if result.returncode != 0:
            print(f"  Frame extraction failed (exit code {result.returncode})")
            # Print last few lines of stderr for actual errors
            err_lines = result.stderr.strip().split("\n")
            for line in err_lines[-5:]:
                print(f"    {line}")
            return None
    except subprocess.TimeoutExpired:
        print("  Frame extraction timed out")
        return None
    except FileNotFoundError:
        print("  ERROR: ffmpeg not found. Install with: sudo apt install ffmpeg")
        print("  Or: pip install imageio-ffmpeg")
        return None

    num_frames = len(list(frames_dir.glob("*.png")))
    print(f"  Extracted {num_frames} frames")
    return frames_dir


# ──────────────────────────────────────────────
# Step 3: Download and parse matching demo
# ──────────────────────────────────────────────

def find_hltv_demo(video: VideoMeta, output_dir: Path) -> Optional[Path]:
    """Try to find and download the matching demo from HLTV.

    This searches HLTV for the match based on team names and tournament.
    Returns path to .dem file if successful.
    """
    import requests

    demo_dir = output_dir / "demos"
    demo_dir.mkdir(parents=True, exist_ok=True)

    # Check if we already have a demo for this video
    demo_marker = demo_dir / f"{video.video_id}.json"
    if demo_marker.exists():
        with open(demo_marker) as f:
            info = json.load(f)
            dem_path = Path(info.get("dem_path", ""))
            if dem_path.exists():
                return dem_path

    # Search HLTV for the match
    # HLTV search is rate-limited and may require careful handling
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }

    team1 = video.team1.lower()
    team2 = video.team2.lower()
    search_query = f"{team1} vs {team2} dust2"

    print(f"  Searching HLTV for: {search_query}")

    try:
        # Try HLTV results page with team filter
        url = f"https://www.hltv.org/results?content=demo&map=de_dust2&query={team1}+{team2}"
        resp = requests.get(url, headers=headers, timeout=30)

        if resp.status_code != 200:
            print(f"  HLTV returned status {resp.status_code}")
            return None

        # Look for match links
        match_links = re.findall(r'href="(/matches/\d+/[^"]+)"', resp.text)

        if not match_links:
            print(f"  No HLTV matches found for {team1} vs {team2}")
            return None

        # Try the first match
        match_url = f"https://www.hltv.org{match_links[0]}"
        print(f"  Found match: {match_url}")

        # Get demo download link
        time.sleep(2)  # Be respectful
        resp = requests.get(match_url, headers=headers, timeout=30)

        demo_links = re.findall(r'href="(/download/demo/\d+)"', resp.text)
        if not demo_links:
            print("  No demo download link found on match page")
            return None

        demo_url = f"https://www.hltv.org{demo_links[0]}"
        print(f"  Downloading demo...")

        resp = requests.get(demo_url, headers=headers, timeout=300, stream=True, allow_redirects=True)
        if resp.status_code != 200:
            print(f"  Demo download failed: {resp.status_code}")
            return None

        # Save demo
        cd = resp.headers.get("Content-Disposition", "")
        if "filename=" in cd:
            filename = re.findall(r'filename="?([^";\n]+)"?', cd)[0]
        else:
            filename = f"{video.video_id}_demo.dem.gz"

        dem_gz_path = demo_dir / filename
        with open(dem_gz_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        # Decompress if needed
        if dem_gz_path.suffix == ".gz":
            import gzip
            dem_path = dem_gz_path.with_suffix("")
            if dem_path.suffix != ".dem":
                dem_path = dem_path.with_suffix(".dem")
            with gzip.open(dem_gz_path, "rb") as f_in:
                with open(dem_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            dem_gz_path.unlink()
        else:
            dem_path = dem_gz_path

        # Save marker
        with open(demo_marker, "w") as f:
            json.dump({"dem_path": str(dem_path), "match_url": match_url}, f)

        print(f"  Demo saved: {dem_path.name}")
        return dem_path

    except Exception as e:
        print(f"  Error finding demo: {e}")
        return None


def parse_demo_actions(
    demo_path: Path,
    player_name: str,
    fps: int = 16,
) -> Optional[np.ndarray]:
    """Parse a demo file and extract actions for a specific player at target FPS."""
    from demoparser2 import DemoParser
    from cs2.action_processing import (
        MOUSE_X_POSSIBLES, MOUSE_Y_POSSIBLES,
        N_KEYS, N_CLICKS, N_MOUSE_X, N_MOUSE_Y,
    )

    print(f"  Parsing demo for player: {player_name}")

    try:
        parser = DemoParser(str(demo_path))
    except Exception as e:
        print(f"  Failed to open demo: {e}")
        return None

    # Get available fields
    fields = ["X", "Y", "Z", "pitch", "yaw", "health", "is_alive"]

    # Try adding button fields
    button_fields = ["FORWARD", "BACK", "LEFT", "RIGHT", "FIRE", "RIGHTCLICK", "RELOAD", "WALK"]
    all_fields = fields + button_fields

    try:
        df = parser.parse_ticks(all_fields)
    except Exception:
        try:
            df = parser.parse_ticks(fields)
            button_fields = []
        except Exception as e:
            print(f"  Failed to parse demo: {e}")
            return None

    if df is None or len(df) == 0:
        print("  No data in demo")
        return None

    # Find the player (case-insensitive partial match)
    player_col = "name" if "name" in df.columns else None
    if player_col is None:
        print("  No player name column found")
        return None

    # Get unique players and find our target
    all_players = df[player_col].unique().tolist()
    target_player = None
    player_lower = player_name.lower()
    for p in all_players:
        if player_lower in str(p).lower() or str(p).lower() in player_lower:
            target_player = p
            break

    if target_player is None:
        print(f"  Player '{player_name}' not found. Available: {all_players}")
        # Try just the first few characters
        for p in all_players:
            if player_lower[:4] in str(p).lower():
                target_player = p
                print(f"  Fuzzy matched to: {target_player}")
                break

    if target_player is None:
        print(f"  Could not match player. Skipping.")
        return None

    player_df = df[df[player_col] == target_player].copy()
    print(f"  Found {len(player_df)} ticks for {target_player}")

    # Get tick rate
    try:
        header = parser.parse_header()
        tickrate = header.get("tickrate", 64)
    except Exception:
        tickrate = 64

    tick_interval = max(1, tickrate // fps)

    # Get ticks and subsample
    if "tick" in player_df.columns:
        ticks = sorted(player_df["tick"].unique().tolist())
    else:
        ticks = list(range(len(player_df)))

    sampled_ticks = ticks[::tick_interval]

    # Build action vectors
    actions = []
    prev_yaw = 0.0
    prev_pitch = 0.0

    for tick in sampled_ticks:
        if "tick" in player_df.columns:
            tick_rows = player_df[player_df["tick"] == tick]
        else:
            tick_rows = player_df.iloc[tick:tick+1]

        if len(tick_rows) == 0:
            continue

        row = tick_rows.iloc[0].to_dict() if hasattr(tick_rows, 'iloc') else dict(tick_rows)

        # Build 51-dim action vector
        action = np.zeros(N_KEYS + N_CLICKS + N_MOUSE_X + N_MOUSE_Y, dtype=np.float32)

        # Keys
        if row.get("FORWARD", False): action[0] = 1.0
        if row.get("LEFT", False): action[1] = 1.0
        if row.get("BACK", False): action[2] = 1.0
        if row.get("RIGHT", False): action[3] = 1.0
        # jump/duck/walk may not be available as direct buttons
        if row.get("WALK", False): action[6] = 1.0
        if row.get("RELOAD", False): action[10] = 1.0

        # Clicks
        if row.get("FIRE", False) or row.get("ATTACK", False): action[N_KEYS] = 1.0
        if row.get("RIGHTCLICK", False) or row.get("ATTACK2", False): action[N_KEYS + 1] = 1.0

        # Mouse from view angle deltas
        yaw = float(row.get("yaw", 0))
        pitch = float(row.get("pitch", 0))

        yaw_delta = yaw - prev_yaw
        pitch_delta = pitch - prev_pitch
        if yaw_delta > 180: yaw_delta -= 360
        elif yaw_delta < -180: yaw_delta += 360

        mouse_x = np.clip(yaw_delta * 10.0, MOUSE_X_POSSIBLES[0], MOUSE_X_POSSIBLES[-1])
        mouse_y = np.clip(pitch_delta * 10.0, MOUSE_Y_POSSIBLES[0], MOUSE_Y_POSSIBLES[-1])

        mx_idx = min(range(len(MOUSE_X_POSSIBLES)), key=lambda i: abs(MOUSE_X_POSSIBLES[i] - mouse_x))
        my_idx = min(range(len(MOUSE_Y_POSSIBLES)), key=lambda i: abs(MOUSE_Y_POSSIBLES[i] - mouse_y))

        action[N_KEYS + N_CLICKS + mx_idx] = 1.0
        action[N_KEYS + N_CLICKS + N_MOUSE_X + my_idx] = 1.0

        actions.append(action)
        prev_yaw = yaw
        prev_pitch = pitch

    if not actions:
        print("  No actions extracted")
        return None

    actions = np.array(actions)
    print(f"  Extracted {len(actions)} action frames at {fps}fps")
    return actions


# ──────────────────────────────────────────────
# Step 4: Build HDF5 dataset
# ──────────────────────────────────────────────

def build_hdf5_episodes(
    frames_dir: Path,
    actions: np.ndarray,
    output_dir: Path,
    episode_length: int = 1000,
    video_meta: Optional[VideoMeta] = None,
) -> int:
    """Combine frames and actions into HDF5 episodes."""
    import cv2
    import h5py

    full_res_dir = output_dir / "full_res"
    full_res_dir.mkdir(parents=True, exist_ok=True)

    # Load frame file list
    frame_files = sorted(frames_dir.glob("*.png"))
    if not frame_files:
        frame_files = sorted(frames_dir.glob("*.jpg"))
    if not frame_files:
        print("  No frame files found")
        return 0

    # Align: use minimum of frames and actions
    n_usable = min(len(frame_files), len(actions))
    if n_usable < episode_length:
        print(f"  Only {n_usable} aligned frame-action pairs (need {episode_length})")
        if n_usable < 100:
            return 0

    n_episodes = n_usable // episode_length
    episodes_created = 0

    # Count existing episodes to avoid overwriting
    existing = list(full_res_dir.glob("*/*.hdf5"))
    episode_offset = len(existing)

    for ep in range(n_episodes):
        start = ep * episode_length
        end = start + episode_length

        ep_idx = episode_offset + ep
        subdir = full_res_dir / f"ep_{ep_idx:05d}"
        subdir.mkdir(exist_ok=True)
        hdf5_path = subdir / f"cs2_episode_{ep_idx}.hdf5"

        if hdf5_path.exists():
            episodes_created += 1
            continue

        with h5py.File(hdf5_path, "w") as f:
            for i, frame_idx in enumerate(range(start, end)):
                # Load and store frame as BGR uint8 (matching DIAMOND format)
                img = cv2.imread(str(frame_files[frame_idx]))
                if img is None:
                    # Use black frame as fallback
                    img = np.zeros((150, 280, 3), dtype=np.uint8)
                f.create_dataset(f"frame_{i}_x", data=img, dtype=np.uint8)
                f.create_dataset(f"frame_{i}_y", data=actions[frame_idx], dtype=np.float32)

        episodes_created += 1

    # Save metadata
    if video_meta:
        meta_path = full_res_dir / f"meta_ep{episode_offset}-{episode_offset + n_episodes - 1}.json"
        with open(meta_path, "w") as f:
            json.dump({
                "video": asdict(video_meta),
                "n_episodes": n_episodes,
                "episode_length": episode_length,
                "n_frames_total": n_usable,
            }, f, indent=2)

    print(f"  Created {episodes_created} episodes ({episodes_created * episode_length} frames)")
    return episodes_created


# ──────────────────────────────────────────────
# Step 5: Create low-res dataset
# ──────────────────────────────────────────────

def create_low_res(output_dir: Path, test_ratio: float = 0.1):
    """Create low-resolution copies and train/test split."""
    import torch
    import torchvision.transforms.functional as T

    from data.dataset import Dataset, CS2Hdf5Dataset
    from data.episode import Episode
    from data.segment import SegmentId

    full_res_dir = output_dir / "full_res"
    low_res_dir = output_dir / "low_res"

    cs2_dataset = CS2Hdf5Dataset(full_res_dir)
    if cs2_dataset.num_episodes == 0:
        print("No episodes found for low-res creation")
        return

    print(f"\nCreating low-res dataset from {cs2_dataset.num_episodes} episodes...")

    import random
    all_ids = list(cs2_dataset._filenames.keys())
    random.shuffle(all_ids)
    n_test = max(1, int(len(all_ids) * test_ratio))
    test_ids = set(all_ids[:n_test])

    train_dataset = Dataset(low_res_dir / "train", None)
    test_dataset = Dataset(low_res_dir / "test", None)

    from tqdm import tqdm
    for file_id in tqdm(cs2_dataset._filenames, desc="Creating low_res"):
        seg = cs2_dataset[SegmentId(file_id, 0, 1000)]
        episode = Episode(
            obs=seg.obs, act=seg.act, rew=seg.rew,
            end=seg.end, trunc=seg.trunc, info={},
        )
        episode.obs = T.resize(episode.obs, (30, 56), interpolation=T.InterpolationMode.BICUBIC)
        episode.info = {"original_file_id": file_id}
        dataset = test_dataset if file_id in test_ids else train_dataset
        dataset.add_episode(episode)

    train_dataset.save_to_default_path()
    test_dataset.save_to_default_path()
    print(f"Train: {train_dataset.num_episodes} episodes, Test: {test_dataset.num_episodes} episodes")


# ──────────────────────────────────────────────
# Main orchestrator
# ──────────────────────────────────────────────

def sanitize_filename(video: VideoMeta) -> str:
    """Create a clean directory name from video metadata."""
    name = f"{video.team1}-vs-{video.team2}_{video.player}_{video.video_id}"
    return re.sub(r'[^\w\-.]', '_', name)


def process_video(video: VideoMeta, output_dir: Path, fps: int = 16) -> int:
    """Process a single video through the full pipeline. Returns episodes created."""
    print(f"\n{'='*60}")
    print(f"Processing: {video.title}")
    print(f"  Player: {video.player} | Teams: {video.team1} vs {video.team2}")
    print(f"  Duration: {video.duration:.0f}s (~{video.duration/60:.0f} min)")
    print(f"{'='*60}")

    # Step 1: Download video and extract frames
    frames_dir = download_and_extract_frames(video, output_dir, fps=fps)
    if frames_dir is None:
        print("  FAILED: Could not extract frames")
        return 0

    # Step 2: Find and download matching demo
    demo_path = find_hltv_demo(video, output_dir)

    # Step 3: Parse demo for actions (or generate placeholder actions)
    if demo_path is not None:
        actions = parse_demo_actions(demo_path, video.player, fps=fps)
    else:
        actions = None

    if actions is None:
        # Generate zero actions as placeholder
        # The model can still learn visual dynamics even with placeholder actions
        num_frames = len(list(frames_dir.glob("*.png")))
        print(f"  WARNING: No demo actions available. Using zero-actions for {num_frames} frames.")
        print(f"  (The model will still learn visual patterns but won't condition on actions)")
        n_actions = 51  # 11 keys + 2 clicks + 23 mouse_x + 15 mouse_y
        actions = np.zeros((num_frames, n_actions), dtype=np.float32)
        # Set mouse to center (zero movement) one-hot
        actions[:, 11 + 2 + 11] = 1.0  # mouse_x = 0 (index 11 in MOUSE_X_POSSIBLES)
        actions[:, 11 + 2 + 23 + 7] = 1.0  # mouse_y = 0 (index 7 in MOUSE_Y_POSSIBLES)

    # Step 4: Build HDF5 episodes
    processed_dir = output_dir / "processed"
    episodes = build_hdf5_episodes(frames_dir, actions, processed_dir, video_meta=video)

    return episodes


def main():
    parser = argparse.ArgumentParser(description="Automated CS2 dataset builder")
    parser.add_argument("--output", type=Path, default=Path("data"), help="Output directory")
    parser.add_argument("--max-videos", type=int, default=10, help="Maximum videos to process")
    parser.add_argument("--max-scan", type=int, default=500, help="Max videos to scan from channel")
    parser.add_argument("--fps", type=int, default=16, help="Target FPS")
    parser.add_argument("--list-only", action="store_true", help="Just list available videos")
    parser.add_argument("--skip-demos", action="store_true", help="Skip HLTV demo download (use zero actions)")
    parser.add_argument("--video-ids", nargs="*", help="Process specific video IDs")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    # Find videos
    print("Scanning YouTube for Dust 2 POV videos...\n")
    videos = find_dust2_videos(max_videos=args.max_videos, max_scan=args.max_scan)

    if not videos:
        print("No Dust 2 POV videos found!")
        return

    # Filter by specific IDs if provided
    if args.video_ids:
        videos = [v for v in videos if v.video_id in args.video_ids]

    # Display found videos
    print(f"\n{'='*60}")
    print(f"Found {len(videos)} Dust 2 POV videos:")
    print(f"{'='*60}")
    for i, v in enumerate(videos):
        mins = v.duration / 60
        print(f"  [{i+1}] {v.player:15s} | {v.team1:12s} vs {v.team2:12s} | {mins:.0f}min | {v.video_id}")
    print()

    total_minutes = sum(v.duration for v in videos) / 60
    est_frames = int(total_minutes * 60 * args.fps)
    est_episodes = est_frames // 1000
    print(f"Total footage: ~{total_minutes:.0f} minutes")
    print(f"Estimated frames: ~{est_frames:,} ({est_episodes} episodes)")
    print()

    if args.list_only:
        return

    # Process each video
    total_episodes = 0
    results = []

    for i, video in enumerate(videos):
        print(f"\n[{i+1}/{len(videos)}]")
        try:
            episodes = process_video(video, args.output, fps=args.fps)
            total_episodes += episodes
            results.append({"video": video.title, "episodes": episodes, "status": "ok"})
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"video": video.title, "episodes": 0, "status": str(e)})

        # Be nice to YouTube
        if i < len(videos) - 1:
            time.sleep(2)

    # Create low-res dataset if we have episodes
    if total_episodes > 0:
        processed_dir = args.output / "processed"
        create_low_res(processed_dir)

    # Summary
    print(f"\n{'='*60}")
    print(f"DATASET BUILD COMPLETE")
    print(f"{'='*60}")
    print(f"Videos processed: {len(results)}")
    print(f"Total episodes: {total_episodes}")
    print(f"Total frames: {total_episodes * 1000:,}")
    print()
    for r in results:
        status = "OK" if r["status"] == "ok" else "FAIL"
        print(f"  [{status}] {r['video'][:60]:60s} -> {r['episodes']} episodes")

    if total_episodes > 0:
        processed = args.output / "processed"
        print(f"\nDataset ready at: {processed}")
        print(f"  Low-res: {processed / 'low_res'}")
        print(f"  Full-res: {processed / 'full_res'}")
        print(f"\nTo train:")
        print(f"  cd src && python3 main.py \\")
        print(f"    env.path_data_low_res={processed / 'low_res'} \\")
        print(f"    env.path_data_full_res={processed / 'full_res'}")

    # Save results log
    log_path = args.output / "dataset_build_log.json"
    with open(log_path, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
