import subprocess
import sys
import os
import time
import webbrowser
from pathlib import Path

def get_project_root():
    """Obtiene la raíz del proyecto, manejando si es un ejecutable de PyInstaller."""
    if hasattr(sys, '_MEIPASS'):
        return Path(sys.executable).parent.absolute()
    return Path(__file__).parent.absolute()

def main():
    project_root = get_project_root()
    venv_python = project_root / ".venv" / "Scripts" / "python.exe"
    python_exe = str(venv_python) if venv_python.exists() else sys.executable
    app_path = project_root / "app" / "chat.py"

    if not app_path.exists():
        print(f"❌ Error: No se encontró la carpeta 'app' o el archivo 'chat.py' en: {project_root}")
        input("Presiona Enter para salir...")
        return

    # Forzar la apertura del navegador después de un pequeño retraso
    # Esto asegura que se abra incluso si Streamlit falla al hacerlo automáticamente
    def open_browser():
        time.sleep(3)
        webbrowser.open("http://localhost:8501")

    import threading
    threading.Thread(target=open_browser, daemon=True).start()

    cmd = [
        python_exe,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port=8501",
        "--server.headless=true",
        "--browser.gatherUsageStats=false",
        "--global.developmentMode=false"
    ]

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        # En modo no-console (PyInstaller), imprimir o input() puede fallar
        try:
            print(f"\n❌ Error: {e}")
            if sys.stdin and sys.stdin.readable():
                input("Presiona Enter para salir...")
        except:
            pass

if __name__ == "__main__":
    main()
