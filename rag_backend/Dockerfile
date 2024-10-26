# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Update the package list and install system dependencies
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container at /app
COPY example_prompts.py prompts.py ./
COPY logger.py ./
COPY requirements.txt ./
COPY main.py rag_engine.py query_translator.py __init__.py ./

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' appuser
USER appuser

# Make port available to the world outside this container
EXPOSE $PORT

# Add healthcheck
HEALTHCHECK CMD curl --fail http://localhost:$PORT/health || exit 1

# Run main.py when the container launches
CMD uvicorn main:app --host 0.0.0.0 --port $PORT