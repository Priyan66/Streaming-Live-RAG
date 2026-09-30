FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false \
    HF_HOME=/opt/hf HF_HUB_DISABLE_PROGRESS_BARS=1

WORKDIR /app
COPY requirements.txt .
# CPU wheels of torch keep the image small; everything else from PyPI, pinned.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0 \
 && pip install -r requirements.txt

# Bake both models into the image so the replay runs with no network access at all.
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); CrossEncoder('cross-encoder/nli-MiniLM2-L6-H768')"
ENV HF_HUB_OFFLINE=1

COPY . .
CMD ["sh", "-c", "python -m pytest -q -p no:cacheprovider && python run_demo.py --all --out /app/runs"]
