#!/bin/bash

# Give the script exec perms: chmod +x scraper-indexer-dockerbuild.sh

# Copy requirements.txt to scraper-indexer directory
cp requirements.txt scraper-indexer/

# Build the Docker image
docker build -t chat-huberman/scraper-indexer -f scraper-indexer/dockerfile .

# Remove the copied requirements.txt
rm scraper-indexer/requirements.txt