"""
Indexador RAG con MarkItDown (Microsoft): Conversión profesional a Markdown.
"""
from __future__ import annotations
import os
import time
import re
import google.generativeai as genai
from pathlib import Path
import chromadb
from markitdown import MarkItDown
from dotenv import load_dotenv
from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.google import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.vector_stores.chroma import ChromaVectorStore

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS_DIR = PROJECT_ROOT / "apuntes"
VECTOR_DB_DIR = PROJECT_ROOT / "chroma_db"
CACHE_DIR = PROJECT_ROOT / "apuntes_limpios"

# Crear carpeta de caché si no existe
CACHE_DIR.mkdir(exist_ok=True)

def load_documents_with_markitdown(progress_callback=None) -> list[Document]:
    """Usa MarkItDown para convertir archivos a Markdown y los guarda físicamente."""
    documents = []
    md = MarkItDown()
    files = sorted(list(DOCUMENTS_DIR.rglob("*.pdf")) + list(DOCUMENTS_DIR.rglob("*.docx")))

    for i, file_path in enumerate(files, 1):
        if file_path.name.startswith("~$") or file_path.name.startswith("."): continue

        cache_file = CACHE_DIR / f"{file_path.stem}.md"

        try:
            # Si ya existe el Markdown limpio, lo leemos directamente para ahorrar tiempo
            if cache_file.exists():
                msg = f"📂 [Cargando {i}/{len(files)}] {file_path.name} (desde caché)"
                if progress_callback: progress_callback(msg)
                with open(cache_file, "r", encoding="utf-8") as f:
                    text = f.read()
            else:
                msg = f"📄 [Convirtiendo {i}/{len(files)}] {file_path.name}"
                if progress_callback: progress_callback(msg)

                result = md.convert(str(file_path))
                text = result.text_content

                # Guardamos el Markdown físico para que el usuario lo vea
                with open(cache_file, "w", encoding="utf-8") as f:
                    f.write(text)

            if text.strip():
                documents.append(Document(
                    text=text,
                    metadata={"source": file_path.relative_to(PROJECT_ROOT).as_posix(), "format": "markdown_microsoft"}
                ))
        except Exception as e:
            print(f"⚠️ Error con {file_path.name}: {e}")

    return documents

def build_index(progress_callback=None) -> VectorStoreIndex:
    load_dotenv(PROJECT_ROOT / ".env")
    google_key = os.getenv("GOOGLE_API_KEY")
    if google_key: genai.configure(api_key=google_key)

    Settings.chunk_size = 1024
    Settings.embed_model = GeminiEmbedding(model_name="models/gemini-embedding-001", api_key=google_key)
    Settings.llm = Gemini(model_name="models/gemini-flash-latest", api_key=google_key)

    # Cargamos usando el nuevo motor de Microsoft
    documents = load_documents_with_markitdown(progress_callback=progress_callback)
    if not documents: raise RuntimeError("No hay documentos válidos en 'apuntes/'.")

    splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=50)

    chroma_client = chromadb.PersistentClient(
        path=str(VECTOR_DB_DIR),
        settings=chromadb.Settings(anonymized_telemetry=False, is_persistent=True)
    )
    collection = chroma_client.get_or_create_collection(name=os.getenv("CHROMA_COLLECTION", "apuntes"))

    existing_ids = set()
    try:
        data = collection.get(include=[])
        if data and 'ids' in data: existing_ids = set(data['ids'])
    except: pass

    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

    # --- INDEXACIÓN CONTROLADA ---
    for doc in documents:
        filename = doc.metadata['source'].split('/')[-1]
        nodes = splitter.get_nodes_from_documents([doc])

        for j, node in enumerate(nodes, 1):
            node.id_ = f"{filename}_md_{j}" # ID con marca MD
            if node.id_ in existing_ids: continue

            msg = f"🚀 [Memorizando {j}/{len(nodes)}] {filename}"
            if progress_callback: progress_callback(msg)

            success = False
            while not success:
                try:
                    index.insert_nodes([node])
                    success = True
                    time.sleep(5)
                except Exception as e:
                    if "429" in str(e):
                        if progress_callback: progress_callback("⏳ Cuota llena. Esperando 30s...")
                        time.sleep(30)
                    else: raise e

    if progress_callback: progress_callback("✅ ¡Todo actualizado con MarkItDown!")
    return index

if __name__ == "__main__":
    build_index()
