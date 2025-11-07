FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel

WORKDIR /home

# Install uv
RUN pip install --no-cache-dir uv

# Copy only dependency files first
COPY pyproject.toml .

# Install dependencies on system level instead of uv venv
RUN uv pip install . --system

# Install openssh-server and git
RUN apt update && \
    DEBIAN_FRONTEND=noninteractive apt-get install openssh-server git -y && \
    rm -rf /var/lib/apt/lists/*

# Create symlink from python to python3
RUN ln -sf /usr/bin/python3 /usr/bin/python

# These environment variables are likely to change or be overridden at runtime
ENV PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV CACHE_DIR=''
ENV PUBLIC_KEY='ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAILrWRmb050DOCw8kVXSLGHArSsFmELFW9qQ6A9AAQXO6 chunhui.gu2@gmail.com'

# Copy source code last since it changes most frequently
# This ensures we only rebuild from this point when code changes
# COPY . .

# Run main script with SSH setup
CMD bash -c 'mkdir -p ~/.ssh; \
cd ~/.ssh; \
chmod 700 ~/.ssh; \
echo "$PUBLIC_KEY" >> authorized_keys; \
chmod 700 authorized_keys; \
service ssh start; \
sleep infinity'

# build image
# docker build -f docker/Dockerfile --platform linux/amd64 -t springlight123/lsc-gnn .

# Run command to test locally:   
# docker run -it --rm --platform linux/amd64 -v $(pwd):/home springlight123/lsc-gnn
# run interactive mode
# docker run --rm -it --platform linux/amd64 -v $(pwd):/home springlight123/lsc-gnn /bin/bash

# upload to docker hub
# docker push springlight123/lsc-gnn

# when run on k8s, mapping /home in cluster to /home in container, only use container as infrastruture
# see train.yaml for details
