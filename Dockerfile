# Use Python 3.11 slim image for smaller size
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy everything
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install the package
RUN pip install .

# Default command as specified
CMD ["python", "-m", "blueskyranker.pipeline", "--similarity-threshold", "0.5", "--no-test"]