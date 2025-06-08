#!/bin/bash

# Build the Docker image with tag "myapp"
docker build -t myapp .

# Run the container with ports exposed and interactive terminal
docker run -p 8000:8000 -p 7860:7860 --rm myapp