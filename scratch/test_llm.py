import os
import google.generativeai as genai
from dotenv import load_dotenv
from pathlib import Path

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

model = genai.GenerativeModel('gemini-2.5-flash')
try:
    response = model.generate_content("Dí 'Hola'")
    print(f"Respuesta: '{response.text}'")
except Exception as e:
    print(f"Error: {e}")
