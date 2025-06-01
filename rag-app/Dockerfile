FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY rag_logic.py .

RUN mkdir -p /app/uploads

ENV LLM_SERVICE_HOST=http://host.docker.internal

CMD ["python", "app.py"]
