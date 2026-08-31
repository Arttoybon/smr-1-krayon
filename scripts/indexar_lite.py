"""
Indexador RAG lite: extrae solo texto de PDFs y DOCX sin llamadas a OpenAI Vision.
Usa embeddings locales con sentence-transformers (100% gratis, sin API).
Usa Google Gemini para la lógica de LLM (GRATIS).
"""
from __future__ import annotations

import os
import time
# Workaround para errores de Protobuf en versiones nuevas de Python
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import google.generativeai as genai
from pathlib import Path

import chromadb
import pymupdf
from docx import Document as DocxDocument
from dotenv import load_dotenv
from llama_index.core import Document, Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.embeddings.google import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.vector_stores.chroma import ChromaVectorStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS_DIR = PROJECT_ROOT / "apuntes"
VECTOR_DB_DIR = PROJECT_ROOT / "chroma_db"


def load_documents(progress_callback=None) -> list[Document]:
    """Carga solo PDFs y DOCX extrayendo texto."""
    documents = []

    # 1. Cargar PDFs (solo texto)
    pdf_paths = sorted(DOCUMENTS_DIR.rglob("*.pdf"))
    if pdf_paths:
        for i, pdf_path in enumerate(pdf_paths, 1):
            msg = f"Procesando PDF ({i}/{len(pdf_paths)}): {pdf_path.name}"
            if progress_callback:
                progress_callback(msg)
            print(msg)
            with pymupdf.open(pdf_path) as pdf:
                for page_number, page in enumerate(pdf, start=1):
                    text = page.get_text("text").strip()
                    if text:
                        documents.append(
                            Document(
                                text=text,
                                metadata={
                                    "source": pdf_path.relative_to(PROJECT_ROOT).as_posix(),
                                    "page": page_number,
                                    "content_type": "pdf_page_text_only",
                                },
                            )
                        )

    # 3. Cargar DOCX (solo texto)
    docx_paths = sorted(DOCUMENTS_DIR.rglob("*.docx"))
    if docx_paths:
        for i, docx_path in enumerate(docx_paths, 1):
            msg = f"Procesando DOCX ({i}/{len(docx_paths)}): {docx_path.name}"
            if progress_callback:
                progress_callback(msg)
            print(msg)
            doc = DocxDocument(docx_path)
            full_text = [para.text for para in doc.paragraphs]
            documents.append(
                Document(
                    text="\n".join(full_text),
                    metadata={
                        "source": docx_path.relative_to(PROJECT_ROOT).as_posix(),
                        "content_type": "docx_text_only",
                    },
                )
            )

    return documents


def build_index(progress_callback=None) -> VectorStoreIndex:
    """Construye el índice RAG con embeddings de Google (sin PyTorch local)."""
    load_dotenv(PROJECT_ROOT / ".env")

    google_key = os.getenv("GOOGLE_API_KEY")

    # Forzamos la configuración global del SDK para aceptar claves con formato AQ.
    if google_key:
        genai.configure(api_key=google_key)

    # Usar embeddings con mayor cuota disponible
    Settings.embed_model = GeminiEmbedding(
        model_name="models/gemini-embedding-001",
        api_key=google_key
    )

    if google_key:
        Settings.llm = Gemini(model_name="models/gemini-flash-latest", api_key=google_key)
    else:
        if progress_callback:
            progress_callback("⚠️ Sin GOOGLE_API_KEY: las respuestas usarán solo recuperación.")
        print("⚠️  Sin GOOGLE_API_KEY: las respuestas usarán solo recuperación.")

    documents = load_documents(progress_callback=progress_callback)

    if not documents:
        raise RuntimeError("No hay documentos compatibles dentro de apuntes/")

    if progress_callback:
        progress_callback(f"✓ Indexando {len(documents)} fragmentos...")
    print(f"✓ Indexando {len(documents)} fragmentos...")

    chroma_client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
    collection = chroma_client.get_or_create_collection(
        name=os.getenv("CHROMA_COLLECTION", "apuntes")
    )
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Procesar en bloques para evitar error 429 de cuota
    if progress_callback:
        progress_callback(f"✓ Iniciando indexación controlada de {len(documents)} fragmentos...")

    # Creamos el índice con el primer documento para inicializarlo
    index = VectorStoreIndex.from_documents(
        [documents[0]],
        storage_context=storage_context,
    )

    # Añadimos el resto en bloques de 5 con pausas
    batch_size = 5
    for i in range(1, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        if progress_callback:
            progress_callback(f"📦 Procesando bloque {int(i/batch_size) + 1}... ({i}/{len(documents)})")

        # Reintentar en caso de error de cuota
        success = False
        while not success:
            try:
                for doc in batch:
                    index.insert(doc)
                success = True
                time.sleep(2) # Pausa corta entre bloques
            except Exception as e:
                if "429" in str(e):
                    if progress_callback:
                        progress_callback("⏳ Cuota agotada temporalmente. Esperando 15 segundos...")
                    time.sleep(15)
                else:
                    raise e

    if progress_callback:
        progress_callback(f"✓ Índice creado correctamente con {len(index.docstore.docs)} fragmentos.")

    return index


def main() -> None:
    build_index()


if __name__ == "__main__":
    main()
