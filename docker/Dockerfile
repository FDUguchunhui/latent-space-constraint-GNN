FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel

WORKDIR /home

# Install uv
RUN pip install uv

# Set environment variables early since they rarely change
ENV PYTHONNOUSERSITE=1
ENV CUDA_VISIBLE_DEVICES=0

# Copy only dependency files first
COPY pyproject.toml .

# Install dependencies on system level instead of uv venv
RUN uv pip install . --system

# These environment variables are likely to change or be overridden at runtime
# so putting them later in the Dockerfile
ENV HF_TOKEN=''
ENV CACHE_DIR=''
ENV HF_LOCAL_STORAGE=''

# Copy source code last since it changes most frequently
# This ensures we only rebuild from this point when code changes
COPY . .

# Run main script
CMD ["python", "main.py"]

# docker build --platform linux/amd64 -t springlight123/gnn-lsc .

# Run command with token:   
# docker run --rm --platform linux/amd64 -v $(pwd)/cache:/app/cache springlight123/gnn-lsc .

# For development:
# docker run --rm --platform linux/amd64 -v $(pwd):/app -v $(pwd)/cache:/app/cache springlight123/gnn-lsc
