# Training & Preliminary Results

## Goal & Context
Train the DIAMOND diffusion world model on the CS2 dataset and generate preliminary visual results comparing CS2 vs CS:GO performance. This addresses the March 31 milestone.

## Architecture
- **Model**: DIAMOND architecture (330M param denoiser + upsampler)
- **Training**: Low-res denoiser at 30x56, upsampler to 150x280
- **Hardware**: RTX 3060 12GB (reduced batch size), or free cloud GPU
- **Config**: batch_size=8, grad_acc=16, effective_batch=128

## Steps
1. Verify data loading pipeline with small subset
2. Run debugging epochs (5-10) to confirm loss decreases
3. Run 50-100 epochs for preliminary results
4. Generate sample frame rollouts
5. Compare visual quality to CS:GO DIAMOND results
6. Document findings for milestone update

## Acceptance Criteria
- [ ] Training loop runs without errors
- [ ] Loss curves show convergence
- [ ] Generated frames show recognizable Dust 2 geometry
- [ ] Side-by-side comparison with CS:GO results documented
- [ ] Preliminary observations about smoke/volumetric rendering

## Boundaries
- Preliminary results only (full training in April)
- Focus on qualitative comparison, not quantitative metrics yet
