@echo off
setlocal
cd /d "%~dp0"
title Lanzador SMR Krayon

:: 1. Verificar Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python no esta instalado.
    if "%1" neq "silent" pause
    exit /b
)

:: 2. Crear entorno virtual si no existe
if not exist ".venv" (
    echo [INFO] Creando entorno virtual...
    python -m venv .venv
)

:: 3. Instalar/Verificar dependencias
:: Solo hacemos pip install si no existe el marcador de instalacion completa
if not exist ".venv\installed.tag" (
    echo [INFO] Instalando dependencias (solo la primera vez)...
    call .venv\Scripts\activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    if %errorlevel% == 0 echo ok > .venv\installed.tag
)

:: 4. Lanzar la aplicacion
call .venv\Scripts\activate
start /b python lanzar_asistente.py

:: No pausar si estamos en modo silencioso
if "%1" neq "silent" (
    echo.
    echo [OK] Aplicacion lanzada. Puedes cerrar esta ventana.
    timeout /t 5
)
exit /b
