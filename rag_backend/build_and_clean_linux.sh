#!/bin/bash

IMAGE_NAME="ragengine"

# Build the new image
docker build --platform linux/amd64 -t ${IMAGE_NAME}:linux-latest .

# Remove the old 'latest' image
docker image prune -f --filter "dangling=true" --filter "label=name=${IMAGE_NAME}"

# Remove any dangling images
docker image prune -f