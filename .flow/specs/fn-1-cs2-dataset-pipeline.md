# CS2 Dataset Pipeline

## Goal & Context
Create a CS2 gameplay dataset in the same format as the original DIAMOND CS:GO dataset (HDF5 files, 150x280 resolution, 16fps, 51-dim actions). This enables direct comparison between CS2 and CS:GO world model performance.

## Architecture & Data Models
- **Source**: HLTV professional match demos (.dem files) filtered to Dust 2
- **Demo parsing**: demoparser2 extracts tick-by-tick player state + button presses
- **Frame capture**: CS2 startmovie command renders frames at 16fps during demo playback
- **Action encoding**: 51-dim multi-hot vector (11 keys + 2 clicks + 23 mouse_x + 15 mouse_y)
- **Output format**: HDF5 files with 1000 frames each, matching DIAMOND convention

## Pipeline Steps
1. Collect ~30-40 Dust 2 demos from HLTV (~20 hours)
2. Parse demos with demoparser2 to extract actions at 16fps
3. Capture frames via CS2 demo playback (startmovie at 16fps)
4. Build HDF5 dataset: pair frames + actions, create full_res (150x280) and low_res (30x56)
5. Validate dataset: inspect frame quality, action distributions, episode count

## Acceptance Criteria
- [ ] 10+ hours of Dust 2 gameplay in HDF5 format
- [ ] Frames at 150x280 resolution, RGB
- [ ] Actions properly encoded in 51-dim format
- [ ] Train/test split created
- [ ] Low-res (30x56) dataset ready for training

## Boundaries
- Only Dust 2 map
- Pro match demos only (not deathmatch - different from original but feasible)
- Frame capture requires CS2 on Windows side
