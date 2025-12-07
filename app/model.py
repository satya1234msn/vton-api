# app/model.py
import os
from pathlib import Path
from typing import Optional
from PIL import Image
import subprocess
import shlex
import time

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

class VTONModel:
    def __init__(self, idm_repo_path: str = None, checkpoint_path: str = None):
        """
        idm_repo_path: path to the cloned IDM-VTON repo on disk.
        checkpoint_path: path to pretrained weights if needed.
        """
        # You can set these via environment vars or pass when constructing.
        self.idm_repo = idm_repo_path or os.environ.get("IDM_VTON_REPO", "/models/IDM-VTON")
        self.checkpoint = checkpoint_path or os.environ.get("IDM_VTON_CHECKPOINT", "")
        self._loaded = False
        self._init_model()

    def _init_model(self):
        """
        Minimal init — for many IDM-VTON setups you just need files in place.
        If you have a python API in IDM-VTON, import & load here to share GPU memory.
        """
        # TODO: if the IDM-VTON repo exposes a Python loader function, import it here:
        # sys.path.insert(0, self.idm_repo)
        # from idm_infer import load_model
        # self.model = load_model(self.checkpoint)
        # self._loaded = True
        # For now, mark as loaded so API responds.
        if Path(self.idm_repo).exists():
            self._loaded = True
        else:
            self._loaded = False

    def is_loaded(self) -> bool:
        return self._loaded

    def run_inference(self, person_path: str, garment_path: str, prompt: Optional[str]=None, upscale: bool=False) -> str:
        """
        Orchestrates running IDM-VTON and optional upscaler.
        Returns path to final PNG file.
        """
        # 1) call the IDM-VTON script / function to generate an output image
        # We'll use an adapter script `infer_adapter.py` (bundled) that demonstrates calling the repo's script.
        out_name = f"vton_out_{int(time.time())}_{Path(person_path).stem}.png"
        out_path = OUTPUT_DIR / out_name

        # Example: call a command-line script inside the cloned IDM repo.
        # The actual command depends on the IDM repo; replace accordingly.
        cmd = f"python {self.idm_repo}/inference/run_infer.py --person {shlex.quote(person_path)} --garment {shlex.quote(garment_path)} --out {shlex.quote(str(out_path))}"
        # If checkpoint is required:
        if self.checkpoint:
            cmd += f" --checkpoint {shlex.quote(self.checkpoint)}"

        # For this scaffold we call a small adapter that you will implement (or change to match repo).
        # If you prefer Python-level import, implement model inference in _init_model and call self.model.infer(...)
        try:
            # Use subprocess to call the IDM-VTON script. Ensure environment has required packages.
            proc = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        except subprocess.CalledProcessError as e:
            # For debugging, include stderr
            raise RuntimeError(f"IDM-VTON adapter failed: {e.stderr[:1000]}")

        # Optionally run an upscaler (if user wants).
        if upscale:
            # Hook your upscaler here, for example call a controlnet upscaler script from your repo.
            upscaled = OUTPUT_DIR / f"up_{out_name}"
            # Example placeholder: copy the same file
            try:
                from shutil import copyfile
                copyfile(out_path, upscaled)
                out_path = upscaled
            except Exception:
                pass

        return str(out_path)
