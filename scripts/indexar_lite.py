"""
Indexador RAG v8: Cambio a modelo 004 para saltar bloqueo de cuota diaria.
"""
from __future__ import annotations
import os
import time
import re
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
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
CACHE_DIR.mkdir(exist_ok=True)

def load_documents_with_markitdown(progress_callback=None) -> list[Document]:
    documents = []
    md = MarkItDown()
    files = sorted(list(DOCUMENTS_DIR.rglob("*.pdf")) + list(DOCUMENTS_DIR.rglob("*.docx")))
    for i, file_path in enumerate(files, 1):
        if file_path.name.startswith("~$"): continue
        cache_file = CACHE_DIR / f"{file_path.stem}.md"
        try:
            if cache_file.exists():
                with open(cache_file, "r", encoding="utf-8") as f: text = f.read()
            else:
                msg = f"📄 [Leyendo {i}/{len(files)}] {file_path.name}"
                if progress_callback: progress_callback(msg)
                result = md.convert(str(file_path))
                text = result.text_content
                with open(cache_file, "w", encoding="utf-8") as f: f.write(text)
            if text.strip():
                documents.append(Document(text=text, metadata={"source": file_path.relative_to(PROJECT_ROOT).as_posix()}))
        except Exception: continue
    return documents

def build_index(progress_callback=None) -> VectorStoreIndex:
    load_dotenv(PROJECT_ROOT / ".env")
    google_key = os.getenv("GOOGLE_API_KEY")
    if google_key: genai.configure(api_key=google_key)

    # Usamos bloques más pequeños para ver progreso real y evitar bloqueos
    embed_model = GeminiEmbedding(model_name="models/gemini-embedding-001", api_key=google_key, embed_batch_size=20)
    Settings.embed_model = embed_model
    Settings.llm = Gemini(model_name="models/gemini-flash-latest", api_key=google_key)

    documents = load_documents_with_markitdown(progress_callback=progress_callback)
    splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=50)

    chroma_client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR), settings=chromadb.Settings(anonymized_telemetry=False, is_persistent=True))
    collection = chroma_client.get_or_create_collection(name=os.getenv("CHROMA_COLLECTION", "apuntes"))

    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

    all_nodes = splitter.get_nodes_from_documents(documents)

    # Nuevo sistema de IDs para el modelo 004
    nodes_to_add = []
    existing_ids = set()
    try:
        data = collection.get(include=[])
        if data and 'ids' in data: existing_ids = set(data['ids'])
    except: pass

    for i, node in enumerate(all_nodes):
        node.id_ = f"node_v8_{i}"
        if node.id_ not in existing_ids: nodes_to_add.append(node)

    if not nodes_to_add:
        if progress_callback: progress_callback("✅ Todo al día.")
        return index

    batch_size = 20 # Bajamos de 100 a 20 para ver el progreso
    total = len(nodes_to_add)
    for i in range(0, total, batch_size):
        batch = nodes_to_add[i : i + batch_size]
        msg = f"🚀 [Sincronizando {min(i+batch_size, total)}/{total}] Procesando bloque..."
        if progress_callback: progress_callback(msg)
        print(msg, flush=True)

        success = False
        while not success:
            try:
                index.insert_nodes(batch)
                success = True
                time.sleep(5) # Pausa corta de seguridad
            except Exception as e:
                if "429" in str(e):
                    msg_wait = "⏳ Cuota Gemini agotada. Esperando 60 segundos para continuar..."
                    if progress_callback: progress_callback(msg_wait)
                    print(msg_wait)
                    time.sleep(65)
                else:
                    print(f"⚠️ Error: {e}")
                    break

    if progress_callback: progress_callback("✅ ¡Sincronización terminada!")
    return index

if __name__ == "__main__":
    build_index()
