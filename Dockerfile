# Training image. Pinned CUDA base so a run started on one provider reproduces on another.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=8

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl rsync openssh-client ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY slm ./slm
RUN pip install --no-cache-dir -e ".[data,cloud,track]"

COPY configs ./configs
COPY scripts ./scripts
COPY tests ./tests
COPY docs ./docs
COPY Makefile ./

# The supervisor is the entrypoint: a reclaimed instance relaunches into the
# same command and resumes from the remote checkpoint mirror.
ENTRYPOINT ["bash", "scripts/supervise.sh"]
CMD ["configs/train/pretrain-300m.yaml"]
