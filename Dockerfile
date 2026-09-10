# Usamos una versión estable de Python
FROM python:3.12-slim

# Evitar que Python genere archivos .pyc y forzar logs en tiempo real
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

WORKDIR /app

# Instalar dependencias del sistema necesarias para MarkItDown y PDFs
RUN apt-get update && apt-get install -y \
    build-essential \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Copiar e instalar dependencias de Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir "markitdown[pdf,docx]" llama-index-llms-gemini llama-index-embeddings-google

# Copiar el resto del código
COPY . .

# Exponer el puerto de Streamlit
EXPOSE 8501

# Comando para lanzar la aplicación
CMD ["streamlit", "run", "app/chat.py", "--server.port=8501", "--server.address=0.0.0.0"]
