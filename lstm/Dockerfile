# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for building some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the source code into the container
COPY ./src /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Set the Python path to include the app directory
ENV PYTHONPATH="/app"

# Set environment variables and configuration paths
ENV PYTHONUNBUFFERED=1

# Ensure the directory exists where the script expects to operate
RUN mkdir -p /app/lstm

# Set the entry point to run the script
CMD ["python", "runtime/Predictor.py", "/app/lstm/prediction_configuration.properties"]

# Expose necessary ports if the application communicates over a network
EXPOSE 80