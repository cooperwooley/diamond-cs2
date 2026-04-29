# DIAMOND-CS2: Diffusion World Model for Counter-Strike 2

Applying the [DIAMOND](https://github.com/eloialonso/diamond) (NeurIPS 2024) diffusion-based world model to Counter-Strike 2 gameplay, testing the hypothesis that volumetric smoke rendering causes measurable temporal consistency degradation compared to the original CS:GO results.

## Project Overview

This project extends the DIAMOND neural game engine from CS:GO to CS2, specifically investigating:
- How well diffusion world models handle CS2's volumetric 3D smoke grenades (vs CS:GO's 2D sprites)
- Temporal consistency of generated frames in dynamic lighting environments
- HUD stability across generated frame sequences

## Setup

```bash
pip install -r requirements.txt
```

## Dataset Pipeline

1. **Download demos**: `python scripts/download_demos.py --output data/demos/`
2. **Parse demos**: `python scripts/parse_demos.py --input data/demos/ --output data/parsed/`
3. **Capture frames**: Use CS2 with `scripts/capture_frames.py` (requires CS2 on Windows)
4. **Build dataset**: `python scripts/build_dataset.py --frames data/frames/ --actions data/parsed/ --output data/processed/`

## Training

```bash
cd src && python main.py env.path_data_low_res=/path/to/low_res env.path_data_full_res=/path/to/full_res
```

## Architecture

Based on DIAMOND (Alonso et al., NeurIPS 2024):
- Low-resolution denoiser: 30x56 → predicts next frame conditioned on 4 previous frames + actions
- Upsampler: 30x56 → 150x280 super-resolution
- EDM-style diffusion with Euler/Heun ODE solvers
- 51-dimensional multi-hot action space (keyboard + mouse)

## Results

Trained for 150 epochs on an L4 GPU. Generated figures live in `results/`:
- `cloud_loss_curve.png` — denoiser/upsampler test loss across checkpoints
- `generated/` — 64-frame rollouts and side-by-side comparisons vs ground truth
- `single_step/` — single-step prediction vs ground truth
- `upsampled/` — low-res vs upsampled vs bicubic baseline

Reproduce with `python scripts/evaluate.py --checkpoint <path-to-checkpoint> --loss-curve`.

## References

- [DIAMOND: Diffusion for World Modeling](https://github.com/eloialonso/diamond) - Alonso et al., NeurIPS 2024
- [Counter-Strike Behavioural Cloning](https://github.com/TeaPearce/Counter-Strike_Behavioural_Cloning) - Original CS:GO dataset
