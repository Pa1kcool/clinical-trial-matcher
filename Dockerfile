# Production image for the Clinical Trial Matcher.
# The embedding + reranker models are downloaded at BUILD time and baked into the
# image, so the running container answers immediately instead of pulling hundreds
# of megabytes of weights on its first request (which would time out a cold start).
FROM python:3.12-slim

# Run as a non-root user. Hugging Face Spaces runs containers as uid 1000; doing
# this ourselves keeps the model cache readable and works the same on any host.
RUN useradd -m -u 1000 user
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    UV_NO_MANAGED_PYTHON=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    HF_HOME=/home/user/.cache/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1 \
    PYTHONUNBUFFERED=1

# uv: fast, lockfile-based installs.
RUN pip install --user --no-cache-dir uv

WORKDIR /home/user/app

# 1) Dependencies only. Cached unless pyproject.toml / uv.lock change.
COPY --chown=user pyproject.toml uv.lock ./
RUN touch README.md && uv sync --frozen --no-install-project --no-dev

# 2) Bake the two models into the image. Cached with the dependency layer above,
#    so editing app code or docs does NOT re-download them.
RUN .venv/bin/python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('NeuML/pubmedbert-base-embeddings'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# 3) Application code, then install the project itself.
COPY --chown=user src ./src
COPY --chown=user README.md ./README.md
RUN uv sync --frozen --no-dev

EXPOSE 7860

# Bind to the port the host provides ($PORT on Render) or 7860 (Spaces default).
CMD ["sh", "-c", ".venv/bin/uvicorn ctmatch.api:app --host 0.0.0.0 --port ${PORT:-7860}"]
