FROM python:3.10-slim
WORKDIR /app
COPY requirements-serve.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements-serve.txt
COPY . .
ENV AEROGUARD_ROOT=/app/workspace AEROGUARD_MODE=hourly
EXPOSE 8000 8501
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]