FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt
RUN pip install --no-cache-dir -r requiremets.txt

COPY streamlit_app.py
COPY streamlit_app_tfidf.py
COPY main.py

COPY data/. /data/

CMD ["streamlit","run","streamlit_app_tfidf","--server.port = 10000"]

