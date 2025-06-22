# --- Stage 1: Builder ---
FROM python:3.12-slim AS builder

# Install system dependencies, Node.js, Rust, and Julia
ENV PYTHONUNBUFFERED=1 \
    RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:/usr/local/julia/bin:$PATH \
    JULIA_VERSION=1.10.0
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential git unzip nodejs npm \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable \
    && curl -fsSL "https://julialang-s3.julialang.org/bin/linux/x64/${JULIA_VERSION%.*}/julia-${JULIA_VERSION}-linux-x86_64.tar.gz" | tar -xz -C /usr/local --strip-components=1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# Install dependencies
RUN pip install --upgrade pip && pip install --no-cache-dir -e .[dev,rust,julia]
RUN cd rust/fastcalc && maturin develop
RUN julia --project=. -e 'using Pkg; Pkg.instantiate()'
RUN cd frontend/nextjs && npm install && npm run build

# --- Stage 2: Final Image ---
FROM python:3.12-slim

WORKDIR /app
COPY --from=builder /app /app
COPY scripts/start-services.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/start-services.sh

# Environment
ENV PYTHONUNBUFFERED=1 \
    MLFLOW_TRACKING_URI=file:///app/mlruns \
    NODE_ENV=production

EXPOSE 3000 5000
CMD ["start-services.sh"] 