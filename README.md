# Asistente IA para Clases Particulares

Proyecto RAG para responder preguntas usando exclusivamente los apuntes guardados en `apuntes/`.
El índice vectorial local se almacenará en `chroma_db/`. Al indexar un PDF, cada
página se renderiza como imagen y se analiza con el modelo multimodal configurado
en `PDF_VISION_MODEL`, por lo que también se pueden recuperar diagramas, tablas,
capturas y texto que no haya detectado el extractor PDF.

## 🚀 Instalación y Uso Rápido (Windows)

Para que el uso sea sencillo, tienes tres opciones en la carpeta principal:

1.  **`Lanzador SMR Krayon.exe`** (Recomendado): Haz doble clic para abrir la aplicación directamente. No necesita terminal.
2.  **`INSTALAR_Y_LANZAR.bat`**: Útil si es la primera vez que instalas o si hay errores, ya que configura el entorno virtual automáticamente.
3.  **`lanzar_asistente.py`**: El script fuente del lanzador.

*Nota: Para que el ejecutable funcione, debe estar siempre en la misma carpeta que `app/`, `scripts/`, `apuntes/` y el archivo `.env`.*

---

## Inicio Manual

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

## Indexar apuntes

Coloca archivos `.md` o `.pdf` dentro de `apuntes/` y ejecuta uno de estos:

**Opción 1: Lite (sin costes, recomendado para empezar)**
- Solo extrae texto de PDFs
- Usa embeddings locales (sentence-transformers)
- Muy rápido y completamente gratis

```powershell
python scripts/indexar_lite.py
```

**Opción 2: Con análisis visual (requiere crédito OpenAI)**
- Renderiza y analiza cada página PDF como imagen
- Recupera diagramas, tablas, capturas y texto
- Llamadas al modelo: ~$0.01 por página PDF

```powershell
python scripts/indexar.py
```

**Nota**: Si ejecutas ambas opciones, la segunda sobrescribe la base de datos con más información.

## Chatbot RAG

Una vez indexados los apuntes, abre el asistente en tu navegador:

```powershell
streamlit run app/chat.py
```

Haz preguntas sobre tus apuntes. El asistente buscará respuestas en la base
de conocimiento indexada y las presentará con contexto de las fuentes.

El workflow de GitHub Actions se incorporará en los siguientes pasos.
