# Use official Pixi image for reproducible environments
FROM ghcr.io/prefix-dev/pixi:latest

# Set working directory inside container
WORKDIR /app

# Copy Pixi environment files first to cache the layer
COPY pixi.toml pixi.lock ./

# Install dependencies based on lockfile
RUN pixi install --locked

# Copy the rest of your application code
COPY . .

# Expose required ports for FastAPI and Gradio
EXPOSE 8000 7860

# Set the default command to run your application
CMD ["pixi", "run", "up"]
