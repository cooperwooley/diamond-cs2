# CS2 Frame Capture Guide

## Quick Start

### Step 1: Get Demo Files
1. Go to https://www.hltv.org/results?map=de_dust2
2. Click on any match played on Dust 2
3. Look for the "GOTV Demo" download button on the match page
4. Download 5-10 demos to start (each is ~100-300MB compressed)
5. Extract the .dem files and place them in `data/demos/`

Alternatively, download your own match demos from CS2:
- In CS2: Settings > Game > Enable "Download My Replays"
- Or use FACEIT/Leetify demo links

### Step 2: Parse Demos for Actions (WSL terminal)
```bash
cd /home/cooper/mst/spring2026/cs6406/project/diamond-cs2
source .venv/bin/activate
python3 scripts/parse_demos.py --input data/demos/ --output data/parsed/ --fps 16
```

### Step 3: Capture Frames in CS2 (Windows side)

1. **Launch CS2** with `-console` launch option
   - Steam > CS2 > Properties > Launch Options: `-console -windowed -w 1024 -h 768`

2. **Copy demo files** to your CS2 directory:
   ```
   C:\Program Files (x86)\Steam\steamapps\common\Counter-Strike Global Offensive\game\csgo\
   ```

3. **Open console** in CS2 (press ~) and run:
   ```
   host_framerate 16
   ```

4. **For each demo**:
   ```
   playdemo "DEMONAME.dem"
   ```
   Wait for it to load, then:
   ```
   startmovie "frames_DEMONAME" tga
   ```
   
   Let the demo play through (or at least 10-15 minutes per demo).
   
   When done:
   ```
   endmovie
   ```

5. **Frame files** will be in:
   ```
   C:\Program Files (x86)\Steam\steamapps\common\Counter-Strike Global Offensive\game\csgo\frames_DEMONAME\
   ```

6. **Copy captured frames** to WSL:
   ```bash
   # From WSL, copy Windows frames to project
   cp -r /mnt/c/Program\ Files\ \(x86\)/Steam/steamapps/common/Counter-Strike\ Global\ Offensive/game/csgo/frames_* data/frames/
   ```

### Step 4: Build Dataset (WSL terminal)
```bash
python3 scripts/build_dataset.py \
    --frames data/frames/ \
    --actions data/parsed/ \
    --output data/processed/
```

### Step 5: Start Training
```bash
cd src
python3 main.py \
    env.path_data_low_res=/home/cooper/mst/spring2026/cs6406/project/diamond-cs2/data/processed/low_res \
    env.path_data_full_res=/home/cooper/mst/spring2026/cs6406/project/diamond-cs2/data/processed/full_res
```

## Tips
- Start with 2-3 demos to verify the pipeline works end-to-end
- Each demo produces ~20,000 frames at 16fps (for a 20-minute half)
- You need ~10 HDF5 episodes (10,000 frames) minimum for a test run
- Use first-person spectator view (spec a player) for best results
- Keep HUD visible - testing HUD stability is part of the project!

## Troubleshooting
- If `startmovie` produces blank frames: make sure the demo is playing
- If frames look wrong: check your CS2 resolution matches expected 4:3
- If actions don't align: verify --fps matches host_framerate
