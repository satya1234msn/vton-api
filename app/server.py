# app/server.py
import uuid
import logging
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
import uvicorn
from .model import VTONModel
from . import utils

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("vton-api")

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs"
utils.ensure_dir(OUT_DIR)

app = FastAPI(title="VTON Inference API (kaggle)")

logger.info("Initializing model...")
vton = VTONModel(idm_repo="/kaggle/working/vton-api/IDM-VTON",
                 checkpoint="/kaggle/working/vton-api/models/IDM-VTON/model.ckpt")
logger.info(f"Model loaded flag: {vton.is_loaded()}")

@app.get("/health")
def health():
    logger.info("Health check")
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

    person_fname = utils.generate_filename("person", person_image.filename)
    garment_fname = utils.generate_filename("garment", garment_image.filename)
    person_path = OUT_DIR / person_fname
    garment_path = OUT_DIR / garment_fname

    try:
        saved_person = await utils.save_upload_file(person_image, person_path)
        saved_garment = await utils.save_upload_file(garment_image, garment_path)
        logger.info(f"[{req_id}] uploads saved")
    except HTTPException as he:
        logger.warning(f"[{req_id}] upload validation failed: {he.detail}")
        raise he
    except Exception as e:
        logger.error(f"[{req_id}] saving error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save uploads: {e}")

    try:
        saved_person, _ = utils.resize_image_if_needed(saved_person, max_dim=2048)
        saved_garment, _ = utils.resize_image_if_needed(saved_garment, max_dim=2048)
    except Exception as e:
        logger.error(f"[{req_id}] resize error: {e}")
        raise HTTPException(status_code=500, detail=f"Resize failed: {e}")

    logger.info(f"[{req_id}] starting inference (upscale={upscale})")
    try:
        out_path = vton.run_inference(str(saved_person), str(saved_garment), prompt=prompt, upscale=upscale)
        logger.info(f"[{req_id}] inference done -> {out_path}")
    except Exception as e:
        logger.error(f"[{req_id}] inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    if not Path(out_path).exists():
        logger.critical(f"[{req_id}] output missing")
        raise HTTPException(status_code=500, detail="Output missing")

    return FileResponse(out_path, media_type="image/png", filename=Path(out_path).name)

if __name__ == "__main__":
    logger.info("Starting server")
    uvicorn.run("app.server:app", host="0.0.0.0", port=8000)
