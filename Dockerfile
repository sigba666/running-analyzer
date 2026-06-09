FROM python:3.10-slim

# 시스템 패키지 (OpenCV, MediaPipe 의존성)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-ㅇ갸 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p uploads static

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
