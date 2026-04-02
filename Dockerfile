FROM rayproject/ray:2.44.1-py310-cu128

WORKDIR /workspace

# Install torchtitan dependencies only. The source is mounted at runtime
# (see docs/ec2_docker_deploy.md), so code changes don't require a rebuild.
COPY pyproject.toml README.md LICENSE /workspace/torchtitan-deps/
COPY assets/version.txt /workspace/torchtitan-deps/assets/version.txt

USER root
RUN pip install torch --index-url https://download.pytorch.org/whl/cu128
RUN pip install /workspace/torchtitan-deps/
RUN pip install --upgrade fsspec

ENV PYTHONPATH=/workspace/torchtitan
