FROM ghcr.io/nvidia-ai-iot/vllm@sha256:b587dd56b4cb076209ad5156a626ac75f5a976d0e8e7d1e6a9fccd56d1bd65e8
USER root
# Only this derived image is changed. The preinstalled vLLM/PyTorch/CUDA stay in place.
RUN if ! command -v g++ >/dev/null; then apt-get update && apt-get install -y --no-install-recommends g++ && rm -rf /var/lib/apt/lists/*; fi
WORKDIR /opt/sparsedvfs
COPY . /opt/sparsedvfs
RUN bash scripts/build_thor.sh --cuda
ENV PYTHONPATH=/opt/sparsedvfs PYTHONDONTWRITEBYTECODE=1 VLLM_USE_V2_MODEL_RUNNER=0
ENV SPARSEDVFS_APPLY=0
ENV SPARSEDVFS_IMAGE_DIGEST=sha256:b587dd56b4cb076209ad5156a626ac75f5a976d0e8e7d1e6a9fccd56d1bd65e8
ENTRYPOINT []
CMD ["python3", "-B", "-m", "sparsedvfs_vllm.inside", "--help"]
