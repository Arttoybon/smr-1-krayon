import os
import google.generativeai as genai
from dotenv import load_dotenv
from pathlib import Path

# Configurar entorno
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

api_key = os.getenv("GOOGLE_API_KEY")
print(f"Usando clave: {api_key[:10]}...")

try:
    genai.configure(api_key=api_key)
    print("Listando modelos disponibles...")
    for m in genai.list_models():
        print(f" - {m.name} (Soporta: {m.supported_generation_methods})")
except Exception as e:
    print(f"Error al listar modelos: {e}")
