# Asistente IA para Clases Particulares

Proyecto RAG para responder preguntas usando exclusivamente los apuntes guardados en `apuntes/`.

## Inicio

1. Instala Python 3.11 o superior.
2. Crea un entorno virtual:

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Instala las dependencias:

   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

4. Copia `.env.example` a `.env` y completa las credenciales.

La implementación del indexador, la interfaz Streamlit y el workflow de GitHub Actions se incorporarán en los siguientes pasos.
