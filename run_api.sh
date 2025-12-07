#!/bin/bash
# run_api.sh - start FastAPI and cloudflared tunnel on Kaggle
set -e
# move to working dir (Kaggle will put files in /kaggle/working)
cd /kaggle/working/vton-api || cd /kaggle/working

# start uvicorn in background
nohup uvicorn app.server:app --host 0.0.0.0 --port 8000 > uvicorn.log 2>&1 &

sleep 5

# download cloudflared if missing
if [ ! -f "./cloudflared" ]; then
  curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o cloudflared
  chmod +x cloudflared
fi

# start tunnel (will print public url)
./cloudflared tunnel --url http://127.0.0.1:8000 --no-autoupdate
