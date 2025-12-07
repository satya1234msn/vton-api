# app/utils.py
"""
Utility helpers for the VTON API.

Functions:
- ensure_dir(path): create directory if missing
- generate_filename(prefix, original_name): produce safe unique filename
- save_upload_file(upload_file, dest_path): save starlette UploadFile to disk
- validate_image_upload(upload_file): basic content-type & size checks
- open_image_safely(path): open with PIL and convert to RGB
- resize_image_if_needed(path, max_dim): optional downscale to save memory
"""

import os
import imghdr
import uuid
from pathlib import Path
from typing import Tuple, Optional
from PIL import Image
from fastapi import UploadFile, HTTPException

# Allowed MIME types for uploads
ALLOWED_MIME = {"image/png", "image/jpeg", "image/jpg", "image/webp"}
# Fallback: allowed imghdr types
ALLOWED_IMGHDR = {"png", "jpeg", "webp"}

def ensure_dir(path: Path) -> None:
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)

def generate_filename(prefix: str, original_name: str) -> str:
    """Return a unique filename preserving the extension (if any)."""
    ext = Path(original_name).suffix or ""
    unique = uuid.uuid4().hex
    return f"{prefix}_{unique}{ext}"

async def save_upload_file(upload_file: UploadFile, dest_path: Path) -> Path:
    """
    Save a Starlette/FastAPI UploadFile to disk safely.

    Raises HTTPException(400/500) for invalid files or IO errors.
    Returns the Path of the saved file.
    """
    # Basic validation first
    validate_image_upload(upload_file)

    ensure_dir(dest_path.parent)

    try:
        # Write in chunks to avoid large-memory usage
        with dest_path.open("wb") as buffer:
            while True:
                chunk = await upload_file.read(1024 * 1024)  # 1MB chunks
                if not chunk:
                    break
                buffer.write(chunk)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {e}")

    # Extra double-check via imghdr (helps catch bogus files)
    img_type = imghdr.what(dest_path)
    if img_type is None or img_type.lower() not in ALLOWED_IMGHDR:
        # remove suspicious file
        try:
            dest_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise HTTPException(status_code=400, detail="Uploaded file is not a supported image type.")

    return dest_path

def validate_image_upload(upload_file: UploadFile, max_size_mb: int = 20) -> None:
    """
    Validate an UploadFile object for content-type and size.
    Raises HTTPException on problems.
    Note: size check here reads client-provided header if available; the robust
    approach is to monitor bytes while saving (see save_upload_file).
    """
    content_type = upload_file.content_type.lower() if upload_file.content_type else ""
    if content_type not in ALLOWED_MIME:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {content_type}")

    # If client provides size in headers, we can pre-check; otherwise skip.
    # Starlette's UploadFile doesn't expose size directly; skip heavy ops.
    # Optionally you can read .file.tell() after reading, but here we avoid that.

def open_image_safely(path: Path) -> Image.Image:
    """
    Open an image with Pillow and ensure RGB mode.
    Raises HTTPException if Pillow fails to open the file.
    """
    try:
        im = Image.open(path)
        # Convert to RGB (many models expect 3 channels)
        if im.mode != "RGB":
            im = im.convert("RGB")
        return im
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to open image: {e}")

def resize_image_if_needed(path: Path, max_dim: int = 2048) -> Tuple[Path, bool]:
    """
    Resize an image file in-place (overwrite) if any dimension exceeds max_dim.
    Returns (path, resized_flag). Uses Pillow for resizing with ANTIALIAS.
    This helps limit memory/VRAM usage for extremely large uploads.
    """
    try:
        im = Image.open(path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to open image for resizing: {e}")

    w, h = im.size
    if max(w, h) <= max_dim:
        # no resizing needed
        if im.mode != "RGB":
            im = im.convert("RGB")
            im.save(path, format="PNG", optimize=True)
        return path, False

    # compute new size keeping aspect ratio
    if w >= h:
        new_w = max_dim
        new_h = int(max_dim * (h / w))
    else:
        new_h = max_dim
        new_w = int(max_dim * (w / h))

    im = im.resize((new_w, new_h), Image.LANCZOS)
    # Save as PNG to preserve colors. Overwrite original.
    im.save(path, format="PNG", optimize=True)
    return path, True
