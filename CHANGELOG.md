# Changelog (Resight fork)

Changes specific to the Resight fork — CI, build pipeline, wheel publication. Upstream changes come from [rahul-goel/fused-ssim](https://github.com/rahul-goel/fused-ssim).

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## 2026-04 — Jetson wheel pipeline stabilisation

### Added

- **Jetson (JetPack 6.2, CUDA 12.6, sm_87 Orin) matrix row** (commit `c7dc06f`, 2026-04-14). Produces `fused_ssim-<v>+jp62-cp310-cp310-linux_aarch64.whl`.
- **amd64 + arm64 baseline CI** (commits `2026-04-05/04-06`). Produces `fused_ssim-<v>+cu128-cp310-cp310-linux_x86_64.whl`.
- Torch source parameterised — Jetson rows pull from [`resight-xr/pytorch-jetson`](https://github.com/resight-xr/pytorch-jetson) releases.

### Fixed

- **Real libcudss, minimal Jetson libs, DT_NEEDED resolution** (`c9e44a1`, `7d31a78`, `95223ae`, `49b65d3`, 2026-04-15).
- **Python 3.10 matrix fix, cu126 variant** (`ac5b75b`, `b169155`, 2026-04-15).

## Before 2026-04

Fork created to add Jetson wheel builds. Upstream at <https://github.com/rahul-goel/fused-ssim>.

[Unreleased]: https://github.com/resight-xr/fused-ssim/compare/main...HEAD
