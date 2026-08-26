from __future__ import annotations

import os
import base64
from pathlib import Path

import chromadb
import pymupdf
from docx import Document as DocxDocument
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
from llama_index.core import Document, Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.multi_modal_llms.gemini import GeminiMultiModal
from llama_index.core.schema import ImageDocument
from llama_index.vector_stores.chroma import ChromaVectorStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS_DIR = PROJECT_ROOT / "apuntes"
VECTOR_DB_DIR = PROJECT_ROOT / "chroma_db"


def describe_image_with_gemini(mm_model: GeminiMultiModal, image_path: Path, source_name: str) -> str:
    """Utiliza Gemini Vision para describir una imagen o diagrama."""
    prompt = (
        "Eres un experto en Sistemas Microinformáticos y Redes (SMR). "
        "Describe detalladamente este diagrama, esquema o imagen técnica. "
        "Identifica componentes, topologías de red, esquemas de hardware y texto visible. "
        "El objetivo es que esta descripción permita a un sistema RAG encontrar la imagen "
        f"cuando un usuario pregunte por conceptos relacionados. Fuente: {source_name}"
    )

    response = mm_model.complete(
        prompt=prompt,
        image_documents=[ImageDocument(image_path=str(image_path))]
    )
    return str(response)


def describe_pdf_page_with_gemini(mm_model: GeminiMultiModal, page: pymupdf.Page, source_name: str, page_number: int) -> str:
    """Renderiza una página de PDF y la describe con Gemini Vision."""
    pixmap = page.get_pixmap(matrix=pymupdf.Matrix(2.0, 2.0), alpha=False)
    img_data = pixmap.tobytes("png")

    # Guardar temporalmente para que Gemini pueda leerlo (o usar bytes si el SDK lo permite)
    temp_path = PROJECT_ROOT / f"temp_page_{page_number}.png"
    with open(temp_path, "wb") as f:
        f.write(img_data)

    prompt = (
        "Describe el contenido visual de esta pagina de apuntes de SMR. "
        "Transcribe el texto visible importante y explica diagramas, tablas y esquemas. "
        f"Documento: {source_name}. Pagina: {page_number}."
    )

    try:
        response = mm_model.complete(
            prompt=prompt,
            image_documents=[ImageDocument(image_path=str(temp_path))]
        )
        description = str(response)
    finally:
        if temp_path.exists():
            temp_path.unlink()

    return description


def load_documents(progress_callback=None) -> list[Document]:
    documents = []
    google_key = os.getenv("GOOGLE_API_KEY")
    if not google_key:
        raise RuntimeError("Falta GOOGLE_API_KEY en el archivo .env")

    mm_model = GeminiMultiModal(model_name="models/gemini-flash-latest", api_key=google_key)

    # 1. Cargar PDFs con Visión (Gemini Flash)
    pdf_paths = sorted(DOCUMENTS_DIR.rglob("*.pdf"))
    if pdf_paths:
        for i, pdf_path in enumerate(pdf_paths, 1):
            msg = f"Procesando PDF con visión ({i}/{len(pdf_paths)}): {pdf_path.name}"
            if progress_callback:
                progress_callback(msg)
            print(msg)
            try:
                with pymupdf.open(pdf_path) as pdf:
                    for page_number, page in enumerate(pdf, start=1):
                        extracted_text = page.get_text("text").strip()
                        visual_description = describe_pdf_page_with_gemini(
                            mm_model,
                            page,
                            pdf_path.name,
                            page_number,
                        )
                        page_content = (
                            f"Texto extraido:\n{extracted_text}\n\n"
                            f"Descripcion visual:\n{visual_description}"
                        ).strip()
                        documents.append(
                            Document(
                                text=page_content,
                                metadata={
                                    "source": pdf_path.relative_to(PROJECT_ROOT).as_posix(),
                                    "page": page_number,
                                    "content_type": "pdf_page_with_vision",
                                },
                            )
                        )
            except Exception as e:
                print(f"Error procesando PDF {pdf_path.name}: {e}")

    # 3. Cargar DOCX
    docx_paths = sorted(DOCUMENTS_DIR.rglob("*.docx"))
    if docx_paths:
        for i, docx_path in enumerate(docx_paths, 1):
            msg = f"Procesando DOCX ({i}/{len(docx_paths)}): {docx_path.name}"
            if progress_callback:
                progress_callback(msg)
            print(msg)
            try:
                doc = DocxDocument(docx_path)
                full_text = [para.text for para in doc.paragraphs]
                documents.append(
                    Document(
                        text="\n".join(full_text),
                        metadata={
                            "source": docx_path.relative_to(PROJECT_ROOT).as_posix(),
                            "content_type": "docx_text",
                        },
                    )
                )
            except Exception as e:
                print(f"Error procesando DOCX {docx_path.name}: {e}")

    # 4. Cargar Imágenes Standalone con Visión
    img_extensions = [".jpg", ".jpeg", ".png"]
    img_paths = [p for p in DOCUMENTS_DIR.rglob("*") if p.suffix.lower() in img_extensions]
    if img_paths:
        for i, img_path in enumerate(img_paths, 1):
            msg = f"Analizando diagrama ({i}/{len(img_paths)}): {img_path.name}"
            if progress_callback:
                progress_callback(msg)
            print(msg)
            try:
                description = describe_image_with_gemini(
                    mm_model,
                    img_path,
                    img_path.name
                )
                documents.append(
                    Document(
                        text=f"Descripcion tecnica del diagrama:\n{description}",
                        metadata={
                            "source": img_path.relative_to(PROJECT_ROOT).as_posix(),
                            "content_type": "standalone_image_vision",
                        },
                    )
                )
            except Exception as e:
                print(f"Error procesando imagen {img_path.name}: {e}")

    return documents


def build_index(progress_callback=None) -> VectorStoreIndex:
    load_dotenv(PROJECT_ROOT / ".env")

    google_key = os.getenv("GOOGLE_API_KEY")
    if not google_key:
        raise RuntimeError("Falta GOOGLE_API_KEY en el archivo .env")

    # Configurar modelos para el índice
    Settings.llm = GoogleGenAI(model="models/gemini-flash-latest", api_key=google_key)
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

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

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
    )

    if progress_callback:
        progress_callback(f"✓ Índice creado correctamente con {len(index.docstore.docs)} fragmentos.")

    return index


def main() -> None:
    build_index()


if __name__ == "__main__":
    main()
