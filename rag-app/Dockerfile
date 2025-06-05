FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY rag_logic.py .
COPY milvus_client.py .

RUN mkdir -p /app/uploads

ENV LLM_SERVICE_HOST=host.docker.internal
ENV MILVUS_HOST=host.docker.internal

CMD ["python", "app.py"]
