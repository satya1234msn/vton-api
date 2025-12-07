# VTON API (scaffold)

## Overview
This project is a scaffold for serving an IDM-VTON-based virtual try-on pipeline over HTTP using FastAPI. It expects you to clone IDM-VTON (or point to your own repo) and provide pretrained weights.

I inspected the Virtual-try-on-evaluation repo (which used IDM-VTON + upscaler) to align this scaffold. :contentReference[oaicite:1]{index=1}

## Prerequisites
- A machine with an NVIDIA GPU and NVIDIA Container Toolkit installed (for Docker with `--gpus all`). For dev you can use an RTX-class GPU; for production prefer A100/T4 depending on throughput. :contentReference[oaicite:2]{index=2}
- Clone IDM-VTON repo and place it in `/path/to/IDM-VTON` (or any path); set the env var `IDM_VTON_REPO` when running.
- Install required model weights as per IDM-VTON instructions.

## Quick local run (without Docker)
1. Clone this scaffold:
   ```bash
   git clone <this-scaffold> vton-api && cd vton-api
