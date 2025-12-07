# app/utils.py
import os
import imghdr
import uuid
from pathlib import Path
from typing import Tuple
from PIL import Image
from fastapi import UploadFile, HTTPException

ALLOWED_MIME = {"image/png", "image/jpeg", "image/jpg", "image/webp"}
ALLOWED_IMGHDR = {"png", "jpeg", "webp"}

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def generate_filename(prefix: str, original_name: str) -> str:
    ext = Path(original_name).suffix or ""
    unique = uuid.uuid4().hex
    return f"{prefix}_{unique}{ext}"

async def save_upload_file(upload_file: UploadFile, dest_path: Path) -> Path:
    validate_image_upload(upload_file)
    ensure_dir(dest_path.parent)
    try:
        with dest_path.open("wb") as buffer:
            while True:
                chunk = await upload_file.read(1024*1024)
                if not chunk:
                    break
                buffer.write(chunk)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {e}")
    img_type = imghdr.what(dest_path)
    if img_type is None or img_type.lower() not in ALLOWED_IMGHDR:
        try:
            dest_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise HTTPException(status_code=400, detail="Uploaded file is not a supported image type.")
    return dest_path

def validate_image_upload(upload_file: UploadFile, max_size_mb: int = 20) -> None:
    content_type = upload_file.content_type.lower() if upload_file.content_type else ""
    if content_type not in ALLOWED_MIME:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {content_type}")

def open_image_safely(path: Path) -> Image.Image:
    try:
        im = Image.open(path)
        if im.mode != "RGB":
            im = im.convert("RGB")
        return im
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to open image: {e}")

def resize_image_if_needed(path: Path, max_dim: int = 2048) -> Tuple[Path, bool]:
    try:
        im = Image.open(path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to open image for resizing: {e}")
    w, h = im.size
    if max(w, h) <= max_dim:
        if im.mode != "RGB":
            im = im.convert("RGB")
            im.save(path, format="PNG", optimize=True)
        return path, False
    if w >= h:
        new_w = max_dim
        new_h = int(max_dim * (h / w))
    else:
        new_h = max_dim
        new_w = int(max_dim * (w / h))
    im = im.resize((new_w, new_h), Image.LANCZOS)
    im.save(path, format="PNG", optimize=True)
    return path, True
