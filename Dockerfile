FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create workspace directory
WORKDIR /workspace

# Copy project files
COPY pyproject.toml .
COPY src/ src/
COPY . .

# Install the project and dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -e . && \
    pip install jupyter jupyterlab ipykernel

# Install git-based dependencies
RUN pip install git+https://github.com/yfisher-lab/SessionTools.git

# Create kernel for Jupyter
RUN python -m ipykernel install --user --name python310 --display-name "Python 3.10"

CMD ["/bin/bash"]
