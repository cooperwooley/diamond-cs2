"""
Download CS2 professional match demos from HLTV.org for Dust 2.

Usage:
    python scripts/download_demos.py --output data/demos/ --max-demos 40

This script downloads demo files from HLTV match pages. You'll need to
provide match URLs or use the built-in search for Dust 2 matches.

Note: HLTV may rate-limit requests. Be respectful with download frequency.
"""

import argparse
import os
import re
import time
from pathlib import Path

import requests


def get_demo_links_from_page(url: str, session: requests.Session) -> list:
    """Extract demo download link from an HLTV match page."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "https://www.hltv.org/",
    }
    try:
        resp = session.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        # Look for demo download link pattern
        matches = re.findall(r'href="(/download/demo/\d+)"', resp.text)
        return [f"https://www.hltv.org{m}" for m in matches]
    except Exception as e:
        print(f"  Error fetching {url}: {e}")
        return []


def download_demo(url: str, output_dir: Path, session: requests.Session) -> bool:
    """Download a demo file from HLTV."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "https://www.hltv.org/",
    }
    try:
        resp = session.get(url, headers=headers, timeout=120, stream=True, allow_redirects=True)
        resp.raise_for_status()

        # Extract filename from Content-Disposition or URL
        cd = resp.headers.get("Content-Disposition", "")
        if "filename=" in cd:
            filename = re.findall(r'filename="?([^";\n]+)"?', cd)[0]
        else:
            filename = url.split("/")[-1] + ".dem.gz"

        filepath = output_dir / filename
        if filepath.exists():
            print(f"  Already exists: {filename}")
            return True

        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        with open(filepath, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded / total * 100
                    print(f"\r  Downloading {filename}: {pct:.1f}%", end="", flush=True)

        print(f"\n  Saved: {filename} ({downloaded / 1024 / 1024:.1f} MB)")
        return True
    except Exception as e:
        print(f"  Error downloading {url}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download CS2 demos from HLTV")
    parser.add_argument("--output", type=Path, default=Path("data/demos"), help="Output directory")
    parser.add_argument("--max-demos", type=int, default=40, help="Maximum number of demos to download")
    parser.add_argument("--match-urls", type=str, nargs="*", help="Specific HLTV match URLs to download")
    parser.add_argument("--url-file", type=Path, help="File containing HLTV match URLs (one per line)")
    parser.add_argument("--delay", type=float, default=3.0, help="Delay between requests (seconds)")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    session = requests.Session()

    # Collect URLs
    urls = []
    if args.match_urls:
        urls.extend(args.match_urls)
    if args.url_file and args.url_file.exists():
        with open(args.url_file) as f:
            urls.extend([line.strip() for line in f if line.strip() and not line.startswith("#")])

    if not urls:
        print("No match URLs provided.")
        print()
        print("To use this script, you need to provide HLTV match URLs.")
        print("Steps:")
        print("  1. Go to https://www.hltv.org/results")
        print("  2. Filter by map: Dust 2")
        print("  3. Copy match page URLs")
        print("  4. Save them to a file (one per line)")
        print("  5. Run: python scripts/download_demos.py --url-file match_urls.txt")
        print()
        print("Or provide URLs directly:")
        print("  python scripts/download_demos.py --match-urls URL1 URL2 ...")
        print()
        print("Creating example URL file at data/demos/example_urls.txt")

        example_file = args.output / "example_urls.txt"
        with open(example_file, "w") as f:
            f.write("# Add HLTV match URLs here, one per line\n")
            f.write("# Example:\n")
            f.write("# https://www.hltv.org/matches/2371234/team1-vs-team2-tournament-name\n")
        return

    downloaded = 0
    for i, url in enumerate(urls):
        if downloaded >= args.max_demos:
            print(f"Reached max demos ({args.max_demos})")
            break

        print(f"\n[{i+1}/{len(urls)}] Processing: {url}")

        # Get demo download links from match page
        demo_links = get_demo_links_from_page(url, session)

        if not demo_links:
            print("  No demo link found on this page")
            time.sleep(args.delay)
            continue

        for link in demo_links:
            if downloaded >= args.max_demos:
                break
            if download_demo(link, args.output, session):
                downloaded += 1
            time.sleep(args.delay)

    print(f"\nDone! Downloaded {downloaded} demos to {args.output}")

    # Check for .gz files and decompress
    gz_files = list(args.output.glob("*.gz"))
    if gz_files:
        print(f"\nFound {len(gz_files)} compressed files. Decompress with:")
        print(f"  cd {args.output} && gunzip *.gz")


if __name__ == "__main__":
    main()
