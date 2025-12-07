# app/model.py
import os
from pathlib import Path
import subprocess, shlex, time
from typing import Optional
from shutil import copyfile

class VTONModel:
    def __init__(self, idm_repo: str = None, checkpoint: str = None):
        # Default Kaggle paths
        self.idm_repo = idm_repo or os.environ.get("IDM_VTON_REPO", "/kaggle/working/vton-api/IDM-VTON")
        self.checkpoint = checkpoint or os.environ.get("IDM_VTON_CHECKPOINT", "/kaggle/working/vton-api/models/IDM-VTON/model.ckpt")
        self._loaded = False
        # mark loaded if repo exists; for production import model properly
        if Path(self.idm_repo).exists():
            self._loaded = True

    def is_loaded(self) -> bool:
        return self._loaded

    def run_inference(self, person_path: str, garment_path: str, prompt: Optional[str]=None, upscale: bool=False) -> str:
        """
        Calls the IDM-VTON inference script (adjust to the actual entrypoint).
        This implementation uses a subprocess adapter. Replace with direct import for efficiency.
        """
        OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_name = f"vton_out_{int(time.time())}.png"
        out_path = OUT_DIR / out_name

        # Example adapter: many repos provide an inference script; adjust this command.
        # If the repo provides a function, import it instead of subprocess.
        possible_script = Path(self.idm_repo) / "inference.py"
        if not possible_script.exists():
            # fallback: if there's a demo script or gradio, user must adapt
            # For now, create a dummy copy of person image to simulate output
            copyfile(person_path, out_path)
            return str(out_path)

        cmd = f"python {shlex.quote(str(possible_script))} --person {shlex.quote(person_path)} --garment {shlex.quote(garment_path)} --out {shlex.quote(str(out_path))}"
        if self.checkpoint:
            cmd += f" --checkpoint {shlex.quote(self.checkpoint)}"
        try:
            subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        except subprocess.CalledProcessError as e:
            # If subprocess fails, raise a helpful error
            raise RuntimeError(f"IDM-VTON script error: {e.stderr[:1000]}")
        # optional upscale placeholder: copy file
        if upscale:
            upscaled = OUT_DIR / ("up_" + out_name)
            copyfile(out_path, upscaled)
            return str(upscaled)
        return str(out_path)
