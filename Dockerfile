# Base image
FROM python:3.10-slim

# Install system dependencies required by librosa
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install git -y

# Uncomment if you want to install opencode inside the container
# RUN apt-get install curl
# RUN curl -fsSL https://opencode.ai/install | bash

# Set working directory
WORKDIR /app

# Copy dependency list
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Default command
# CMD ["python", "main.py"]
