# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Create a user and group
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Change ownership of the application files
RUN chown -R appuser:appuser /app

# Switch to the new user
USER appuser

# Add healthcheck
HEALTHCHECK CMD python -c "import requests; requests.get('http://localhost:8000')" || exit 1

# Run the application
CMD ["python", "scrape_transcripts.py"]
