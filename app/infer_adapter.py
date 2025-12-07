# app/infer_adapter.py
"""
Small adapter that demonstrates how to call IDM-VTON inference script from the repo.
Adjust the `idm_repo` location and CLI options to match the repo's real script.
"""
import argparse
from pathlib import Path
from subprocess import run, PIPE

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--idm_repo", required=True)
    p.add_argument("--person", required=True)
    p.add_argument("--garment", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--checkpoint", default="")
    args = p.parse_args()

    # Example: call the repo's script
    script = Path(args.idm_repo) / "gradio_demo.py"  # or inference script name in repo
    cmd = ["python", str(script), "--person", args.person, "--garment", args.garment, "--out", args.out]
    if args.checkpoint:
        cmd.extend(["--checkpoint", args.checkpoint])
    run(cmd, check=True)

if __name__ == "__main__":
    main()
