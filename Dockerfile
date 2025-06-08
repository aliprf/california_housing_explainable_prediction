### --- Build Stage --- ###
FROM ghcr.io/prefix-dev/pixi:latest AS build

WORKDIR /app
COPY pixi.toml pixi.lock ./
COPY . .

# Install locked dependencies and create env activation script
RUN pixi install --locked \
    && pixi shell-hook -s bash > /shell-hook.sh \
    && echo 'exec "$@"' >> /shell-hook.sh

### --- Runtime Stage --- ###
FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxrender1 libxext6 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy Pixi environment and activation script from build stage
COPY --from=build /app/.pixi /app/.pixi
COPY --from=build /shell-hook.sh /app/entrypoint.sh

# Copy app source code
COPY . .

RUN chmod +x /app/entrypoint.sh

# Ensure Pixi environment binaries are in PATH
ENV PATH="/app/.pixi/envs/default/bin:$PATH"

# Expose ports for FastAPI and Gradio
EXPOSE 8000 7860

# Entrypoint to activate Pixi environment, then run CMD
ENTRYPOINT ["/app/entrypoint.sh"]

# Run your app entrypoint
CMD ["python", "main.py"]
