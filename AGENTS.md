# AGENTS.md — resight-xr/fused-ssim

Universal instructions for coding agents working in this fork.

## What this repo is

A fork of [rahul-goel/fused-ssim](https://github.com/rahul-goel/fused-ssim) maintained by Resight to build **aarch64 + NVIDIA Jetson CUDA wheels**. Upstream source is unchanged; we add a CI workflow (`.github/workflows/build_wheels.yml`) that compiles wheels for both amd64 (`+cu128`) and arm64-Jetson (`+jp62`).

Why: see [resight-lab's why-fork-jetson-wheels explainer](https://github.com/resight-xr/resight-lab/blob/main/docs/explanation/why-fork-jetson-wheels.md).

Sibling repo with the same pattern: [resight-xr/nvdiffrast](https://github.com/resight-xr/nvdiffrast).

## What NOT to do here

- **Do not change upstream source code** (`ssim.cu`, `ssim.h`, `ssim.mm`, `ext.cpp`, `fused_ssim/`). If we diverge from upstream, rebases break.
- **Do not remove the Jetson matrix row** — the reason this fork exists.
- **Do not break the `+jp62` version stamp** — downstream pins by filename.

## What to do

- Fix CI when it's red.
- Rebase on upstream periodically when they ship improvements.
- Re-run the Jetson matrix row when `pytorch-jetson` bumps torch.

## Build artifacts

- `fused_ssim-<v>+cu128-cp310-cp310-linux_x86_64.whl` (amd64, CUDA 12.8)
- `fused_ssim-<v>+jp62-cp310-cp310-linux_aarch64.whl` (arm64 Jetson, JetPack 6.2 / CUDA 12.6 / sm_87 Orin)

Consumed by [resight-lab](https://github.com/resight-xr/resight-lab) via `vendor/wheels/SOURCES.txt`.

## Commit messages

Conventional Commits. CI-only changes use `ci(jetson): ...` or `ci(wheels): ...`.

## When this fork goes away

If a well-tagged upstream or PyPI-published aarch64 Jetson wheel exists, we archive.

## See also

- [`CLAUDE.md`](./CLAUDE.md)
- [`CHANGELOG.md`](./CHANGELOG.md)
- [resight-lab's explanation](https://github.com/resight-xr/resight-lab/blob/main/docs/explanation/why-fork-jetson-wheels.md)
