# Dockerfile
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Python
RUN apt-get update && apt-get install -y python3-pip python3-venv git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project
COPY . /app

# Install Python deps
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

# (Optional) If IDM-VTON repo is not mounted, clone it (or you can mount a volume)
# RUN git clone https://github.com/yisol/IDM-VTON.git /models/IDM-VTON

# Expose the server port
EXPOSE 8080

ENV PYTHONUNBUFFERED=1
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
