FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 APP_ENV=production
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt && useradd --create-home --uid 10001 studio
COPY --chown=studio:studio . .
RUN mkdir -p logs uploads outputs reports && chown -R studio:studio /app
USER studio
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health',timeout=3)"
CMD ["python","-m","streamlit","run","app.py","--server.address=0.0.0.0","--server.port=8501","--server.headless=true"]
