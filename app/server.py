# app/server.py
import uuid
import logging
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
import uvicorn

from .model import VTONModel
from . import utils

# ---------------------------
# Logging configuration
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("vton-api")

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs"
utils.ensure_dir(OUT_DIR)

app = FastAPI(title="VTON Inference API (logging enabled)")

# Load model once on startup
logger.info("Initializing VTON model...")
vton = VTONModel()
logger.info(f"Model loaded: {vton.is_loaded()}")


@app.get("/health")
def health():
    logger.info("Health check requested.")
    return {"ok": True, "model_loaded": vton.is_loaded()}


@app.post("/predict")
async def predict(
    person_image: UploadFile = File(...),
    garment_image: UploadFile = File(...),
    prompt: str = Form(None),
    upscale: bool = Form(False)
):
    req_id = uuid.uuid4().hex[:8]
    logger.info(f"[{req_id}] /predict called")

    # --- create unique filepaths ---
    person_fname = utils.generate_filename("person", person_image.filename)
    garment_fname = utils.generate_filename("garment", garment_image.filename)
    person_path = OUT_DIR / person_fname
    garment_path = OUT_DIR / garment_fname

    logger.info(f"[{req_id}] Saving uploads → {person_fname}, {garment_fname}")

    # --- Save uploads safely ---
    try:
        saved_person = await utils.save_upload_file(person_image, person_path)
        saved_garment = await utils.save_upload_file(garment_image, garment_path)
        logger.info(f"[{req_id}] Uploads saved successfully.")
    except HTTPException as he:
        logger.warning(f"[{req_id}] Upload validation failed: {he.detail}")
        raise he
    except Exception as e:
        logger.error(f"[{req_id}] Unexpected upload saving error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save uploads: {e}")

    # --- Optional resize ---
    try:
        saved_person, resized_p = utils.resize_image_if_needed(saved_person, max_dim=2048)
        saved_garment, resized_g = utils.resize_image_if_needed(saved_garment, max_dim=2048)
        if resized_p or resized_g:
            logger.info(f"[{req_id}] One or both images were resized to safe VRAM limits.")
        else:
            logger.info(f"[{req_id}] No resizing necessary.")
    except Exception as e:
        logger.error(f"[{req_id}] Error during resizing: {e}")
        raise HTTPException(status_code=500, detail=f"Resize failed: {e}")

    # --- Run inference ---
    logger.info(f"[{req_id}] Starting inference. Upscale={upscale}, Prompt={prompt}")

    try:
        out_path = vton.run_inference(
            str(saved_person),
            str(saved_garment),
            prompt=prompt,
            upscale=upscale
        )
        logger.info(f"[{req_id}] Inference completed → Output: {out_path}")
    except Exception as e:
        logger.error(f"[{req_id}] Inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    # --- Verify output ---
    if not Path(out_path).exists():
        logger.critical(f"[{req_id}] Output file missing after inference!")
        raise HTTPException(status_code=500, detail="Output file missing.")

    logger.info(f"[{req_id}] Sending output file {Path(out_path).name}")

    return FileResponse(out_path, media_type="image/png", filename=Path(out_path).name)


if __name__ == "__main__":
    logger.info("Starting VTON API server on port 8080...")
    uvicorn.run("app.server:app", host="0.0.0.0", port=8080, reload=False)
