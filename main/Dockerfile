FROM python:3.9-slim

# Prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE 1 
# Prevent Python from buffering stdout and stderr
ENV PYTHONUNBUFFERED 1 

# set work dir
WORKDIR /app

# Update the package list and install system dependencies
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt app.py api_key_validator.py ./

RUN pip install --no-cache-dir -r requirements.txt

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' appuser

EXPOSE $PORT

# Create a shell script to run Streamlit
RUN echo '#!/bin/bash' > /app/run.sh && \
    echo 'streamlit run --server.port=$PORT --server.address=0.0.0.0 app.py' >> /app/run.sh && \
    chmod +x /app/run.sh

USER appuser

CMD ["/app/run.sh"]