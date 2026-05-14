FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1 \
    UVICORN_HOST=0.0.0.0 \
    UVICORN_PORT=8000

CMD ["bash", "-lc", "if [ \"$CMD_MODE\" = \"stream_worker\" ]; then \
                        python cameras_processing/stream_worker.py; \
                     elif [ \"$CMD_MODE\" = \"tasks\" ]; then \
                        python core/tasks_processing.py; \
                     else \
                        uvicorn main_server:app --host 0.0.0.0 --port 8000 --workers 3; \
                     fi"]