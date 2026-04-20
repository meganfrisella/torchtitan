FROM rayproject/ray:2.44.1-py310-cu128

WORKDIR /workspace
SHELL ["/bin/bash", "-lc"]

# Install torchtitan dependencies only. The source is mounted at runtime
# (see docs/ec2_docker_deploy.md), so code changes don't require a rebuild.
COPY pyproject.toml README.md LICENSE /workspace/torchtitan-deps/
COPY assets/version.txt /workspace/torchtitan-deps/assets/version.txt

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-nsight-systems-12-8 \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/NVIDIA/Megatron-LM.git /workspace/Megatron-LM

RUN conda create -y -n torchtitan python=3.10 && \
    conda run -n torchtitan pip install torch --index-url https://download.pytorch.org/whl/cu128 && \
    conda run -n torchtitan pip install /workspace/torchtitan-deps/ && \
    conda run -n torchtitan pip install --upgrade fsspec && \
    conda clean -afy

RUN conda create -y -n megatron python=3.12 && \
    conda run -n megatron pip install torch --index-url https://download.pytorch.org/whl/cu128 && \
    conda run -n megatron pip install \
        numpy \
        packaging \
        pydantic \
        pyyaml \
        einops \
        importlib-metadata \
        nvdlfw-inspect \
        onnx \
        onnxscript \
        tensorboard && \
    conda run -n megatron pip install --no-deps \
        transformer-engine==2.13.0 \
        transformer-engine-torch==2.13.0 \
        transformer-engine-cu12==2.13.0 && \
    conda run -n megatron pip install -e /workspace/Megatron-LM --no-deps && \
    conda clean -afy

RUN conda create -y -n deepspeed python=3.10 && \
    conda run -n deepspeed pip install torch --index-url https://download.pytorch.org/whl/cu128 && \
    conda run -n deepspeed pip install deepspeed && \
    conda clean -afy

RUN mkdir -p /workspace/out

ENV TORCHTITAN_CONDA_ENV=torchtitan
ENV MEGATRON_CONDA_ENV=megatron
ENV DEEPSPEED_CONDA_ENV=deepspeed
ENV MEGATRON_WORKSPACE=/workspace/Megatron-LM
