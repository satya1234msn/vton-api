# vton-api Kaggle-ready package

This package contains a minimal FastAPI scaffold to run in a Kaggle notebook with a GPU (T4).
It expects you to place the IDM-VTON repository and the pretrained weights into the `IDM-VTON` folder.

## Folder layout (inside Kaggle working dir)
/kaggle/working/vton-api/
  app/
    server.py
    model.py
    utils.py
  run_api.sh
  requirements.txt

## Steps on Kaggle
1. Create new Notebook, enable GPU (T4) and turn Internet ON.
2. Upload this `vton-api-kaggle` folder into Files -> Upload.
3. Install requirements:
   ```
   %cd /kaggle/working/vton-api
   !pip install -r requirements.txt
   ```
4. Place IDM-VTON repo under:
   `/kaggle/working/vton-api/IDM-VTON`
   and model weights under:
   `/kaggle/working/vton-api/models/IDM-VTON/model.ckpt`
   You can upload or `git clone` the repo and `wget` weights from HuggingFace.
5. Run the API + tunnel:
   ```
   %%bash
   chmod +x run_api.sh
   ./run_api.sh
   ```
6. Cloudflared will print a public URL. Use that to call `/health` and `/predict`.

## Notes
- The provided `model.py` uses a subprocess adapter; update it to the real repo's inference entrypoint or import model functions for performance.
- Kaggle sessions are ephemeral; download outputs if needed.
