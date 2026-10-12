# motionbrick-mlx

Real-time motion generation for a character, on Apple Silicon, in **MLX**.

An independent implementation inspired by MotionBricks (see `NOTICE`): modular
latent motion model (VQ-VAE over motion, decoupled root/pose) plus a conditioned
generator, feeding an **offline bake pipeline** that emits motion assets our
engine already plays (VRMA / animation clips). Runtime cost in the product is
then zero.

## Why here, and why MLX
* The reference implementation is PyTorch + CUDA + MuJoCo and is Linux-first.
* Apple Silicon has no CUDA. MLX is the native array/autograd stack on unified memory.
* We do **not** need the reference training corpus: our own 3D space is a
  simulator, so we can generate (state, action, outcome) motion data at will.

## Status
Scaffold. Order of work:
1. motion representation + data export from our space  ← next
2. MLX VQ-VAE for motion latents (train to reconstruct)
3. conditioned generator (style / target / root trajectory) over the latents
4. bake + export to engine motion assets
5. (optional) distill a small decoder for on-device realtime, via Core ML

## Layout
```
docs/motion-representation.md   the feature spec, units and conventions
src/motionbricks_mlx/motion/    representation, retarget, io
src/motionbricks_mlx/models/    vqvae.py, generator.py
src/motionbricks_mlx/data/      dataset + generated-data loaders
src/motionbricks_mlx/export/    bake to engine formats
scripts/check_env.py            is this machine ready (Apple Silicon + MLX)
datasets/  checkpoints/         ignored; local artifacts
```

## License
Apache-2.0 for this project's code. No NVIDIA weights are used or shipped.
