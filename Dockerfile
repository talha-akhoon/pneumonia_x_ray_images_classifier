FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo \
    zlib1g \
    && rm -rf /var/lib/apt/lists/*

COPY api/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

ENV PORT=7860
EXPOSE 7860

ENV MODEL_CHECKPOINT=models/checkpoints/mobilenet_v2_lr0.0001_drop0.5_Unfrozen_layers_3_best_05_recall0.994845.pth
ENV DEVICE=cpu

CMD ["sh", "-c", "uvicorn api.app:app --host 0.0.0.0 --port ${PORT}"]
