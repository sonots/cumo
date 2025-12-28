FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ARG RUBY_VERSION=3.4.7

ENV DEBIAN_FRONTEND=noninteractive
ENV RBENV_ROOT="/root/.rbenv"
ENV PATH="${RBENV_ROOT}/bin:${RBENV_ROOT}/shims:${PATH}"

ENV CUDA_PATH=/usr/local/cuda
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
ENV CPATH=/usr/local/cuda/include:${CPATH}
ENV LIBRARY_PATH=/usr/local/cuda/lib64:${LIBRARY_PATH}

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    wget \
    curl \
    vim \
    ca-certificates \
    libssl-dev \
    libreadline-dev \
    zlib1g-dev \
    libyaml-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/rbenv/ruby-build.git && \
    cd ruby-build/bin && ./ruby-build ${RUBY_VERSION} /usr && \
    git config --global --add safe.directory /workspace

WORKDIR /workspace

CMD ["/bin/bash"]
