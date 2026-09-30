# syntax=docker/dockerfile:1.7

FROM --platform=$BUILDPLATFORM golang:1.24-alpine AS proxy-builder
ARG TARGETOS
ARG TARGETARCH
WORKDIR /src
COPY worker/ping-proxy/main.go ./main.go
RUN CGO_ENABLED=0 GOOS=$TARGETOS GOARCH=$TARGETARCH \
    go build -trimpath -ldflags='-s -w' -o /out/health-proxy ./main.go

FROM nvidia/cuda:12.8.1-devel-ubuntu24.04 AS llama-builder
ARG LLAMA_COMMIT=81ff93ea1d48c0482508d30e75b814df99c25bf0
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential ca-certificates cmake git libcurl4-openssl-dev && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /src/llama.cpp
RUN git init . && git remote add origin https://github.com/ggml-org/llama.cpp && \
    git fetch --depth=1 origin "$LLAMA_COMMIT" && git checkout --detach FETCH_HEAD && \
    test "$(git rev-parse HEAD)" = "$LLAMA_COMMIT"
RUN cmake -S . -B build \
      -DBUILD_SHARED_LIBS=OFF \
      -DGGML_CUDA=ON \
      -DGGML_NATIVE=OFF \
      -DCMAKE_CUDA_ARCHITECTURES=90 \
      -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build --config Release -j"$(nproc)" --target llama-server

FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ca-certificates curl libgomp1 && rm -rf /var/lib/apt/lists/*
COPY --from=llama-builder /src/llama.cpp/build/bin/llama-server /app/llama-server
COPY --from=proxy-builder /out/health-proxy /opt/runpod/health-proxy
COPY --chmod=0755 worker/start.sh /opt/runpod/start-penclaw

ENV MODEL_REPOSITORY=Jerome0207/penclaw-glm-5.3-abliterated-q4-k-m \
    MODEL_CACHE=/runpod-volume/huggingface-cache/hub \
    MODEL_ALIAS=penclaw-glm-5.3-abliterated-q4-k-m \
    CONTEXT_SIZE=131072 \
    GPU_COUNT=4 \
    SPLIT_MODE=layer \
    TENSOR_SPLIT=1,1,1,1 \
    PORT=80 \
    HEALTH_CHECK_PATH=/ping

EXPOSE 80
STOPSIGNAL SIGTERM
HEALTHCHECK --interval=10s --timeout=3s --start-period=20m --retries=3 \
    CMD curl -fsS http://127.0.0.1:80/ping >/dev/null || exit 1
ENTRYPOINT ["/opt/runpod/start-penclaw"]
