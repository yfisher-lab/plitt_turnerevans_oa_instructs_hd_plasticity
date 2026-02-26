FROM python:3.11-slim

WORKDIR /app

# Install uv package manager
RUN pip install --no-cache-dir uv

# Copy the uv.lock and pyproject.toml files
COPY uv.lock pyproject.toml ./

# Install dependencies using uv
RUN uv pip install --system -r <(uv pip compile uv.lock) --require-hashes

# Copy the rest of the application
COPY . .

# Install the package in development mode
RUN pip install -e .

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["python"]
