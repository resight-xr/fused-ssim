# CLAUDE.md — resight-xr/fused-ssim

Claude-specific notes. Universal guidance in [`AGENTS.md`](./AGENTS.md) — read first.

## Footguns specific to this fork

1. **Don't edit the SSIM C++/CUDA code.** This is a CI-shim fork. Edits to the upstream kernels create rebase hell. If something is broken in the algorithm, PR upstream; we rebase.
2. **Don't drop the Jetson row.** The `+jp62` builds are the only reason the fork exists.
3. **sm_87 only** on the Jetson row. Orin Nano is our only Jetson SKU today.
4. **Torch version cross-repo dep**: the Jetson row pulls torch from `resight-xr/pytorch-jetson` releases. Bumping torch in pytorch-jetson cascades here — re-run the matrix.
