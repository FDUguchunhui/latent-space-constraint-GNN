FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel

WORKDIR /home

# Install uv
RUN pip install --no-cache-dir uv

# Copy only dependency files first
COPY pyproject.toml .

# Install dependencies on system level instead of uv venv
RUN uv pip install . --system

# These environment variables are likely to change or be overridden at runtime
ENV PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV CACHE_DIR=''

# Copy source code last since it changes most frequently
# This ensures we only rebuild from this point when code changes
# COPY . .

# Run main script
CMD ["python", "main.py", '-m', 'verbose=true']

# build image
# docker build -f docker/Dockerfile --platform linux/amd64 -t springlight123/gnn-lsc .

# Run command to test locally:   
# docker run -it --rm --platform linux/amd64 -v $(pwd):/home springlight123/gnn-lsc
# run interactive mode
# docker run --rm -it --platform linux/amd64 -v $(pwd):/home springlight123/gnn-lsc /bin/bash

# upload to docker hub
# docker push springlight123/gnn-lsc

# when run on k8s, mapping /home in cluster to /home in container, only use container as infrastruture
# see train.yaml for details
