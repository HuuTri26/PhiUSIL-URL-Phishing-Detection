FROM python:3.10-slim
LABEL authors="DinhToan21057"

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1 \
    libstdc++6 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /PhiUSIIL-URL-Detector

COPY scripts ./scripts
COPY secrets ./secrets
COPY path_map.json .
COPY requirements.txt .

ENV FIREBASE_CREDENTIAL=phiusiil-url-phishing-database-firebase-adminsdk-fbsvc-510e06e3aa.json

ARG BIND=0.0.0.0:8000
ENV GUNICORN_BIND=$BIND

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["sh", "-c", "gunicorn scripts.app:app -b ${GUNICORN_BIND} -w 4 --timeout 120"]