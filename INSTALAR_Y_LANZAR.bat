@echo off
setlocal
cd /d "%~dp0"
title Lanzador Unico SMR Krayon

:: Ruta al entorno virtual
set VENV_PYTHON="%~dp0.venv\Scripts\python.exe"
set APP_PATH="%~dp0app\chat.py"

echo [1/3] Verificando entorno...
if not exist ".venv" (
    echo [INFO] Creando entorno virtual seguro...
    python -m venv .venv
)

echo [2/3] Sincronizando librerias...
:: Instalamos primero lo basico y pesado que no suele fallar
%VENV_PYTHON% -m pip install --upgrade pip --quiet
%VENV_PYTHON% -m pip install streamlit pandas python-dotenv chromadb pymupdf pypdf python-docx fpdf2 requests --quiet

:: Instalamos los de IA. Si fallan por dependencias (Pillow), forzamos sin dependencias
echo [INFO] Instalando componentes de IA...
%VENV_PYTHON% -m pip install llama-index-core llama-index-vector-stores-chroma --quiet
%VENV_PYTHON% -m pip install llama-index-llms-gemini llama-index-embeddings-google --no-deps --quiet
%VENV_PYTHON% -m pip install google-generativeai markitdown --quiet

:: Intentamos instalar una version de Pillow compatible con Python 3.14 (si existe)
:: Si falla, el programa podria funcionar igual si no procesa imagenes pesadas localmente
%VENV_PYTHON% -m pip install "Pillow>=11.0.0" --quiet

echo [3/3] Abriendo Well, Actually...
start "" http://localhost:8501
%VENV_PYTHON% -m streamlit run %APP_PATH% --server.port=8501 --server.address=0.0.0.0 --server.headless=true --global.developmentMode=false

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] No se pudo iniciar el asistente.
    pause
)
exit /b
