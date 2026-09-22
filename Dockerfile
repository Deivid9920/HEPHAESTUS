# HEPHAESTUS CPU image: toolchain + python oracle for golden tests.

FROM ubuntu:22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git python3.10 python3.10-venv python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN python3.10 -m venv .venv \
    && .venv/bin/pip install --upgrade pip \
    && .venv/bin/pip install -r requirements.txt

COPY . .
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build -j"$(nproc)"

CMD ["make", "test"]
