import os
import sys
import time
import logging
import zipfile
import io
import chromadb
import pymupdf
import streamlit as st

# Desactivar avisos de deprecación de Streamlit y otros logs
os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)

from pathlib import Path
from dotenv import load_dotenv

# Silenciar logs ruidosos de bibliotecas externas
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["GOOGLE_LOGLEVEL"] = "3"

from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from PIL import Image

# Añadir el directorio raíz al path para importar los scripts
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from scripts import indexar_lite, indexar

def load_index() -> VectorStoreIndex:
    """Carga el índice desde la base de datos vectorial configurado para Gemini."""
    load_dotenv(PROJECT_ROOT / ".env")

    # 1. Configurar LLM (Google Gemini - GRATIS)
    google_key = os.getenv("GOOGLE_API_KEY")
    if google_key:
        # Usamos el nuevo SDK oficial de Google GenAI con un modelo garantizado (flash-latest)
        Settings.llm = GoogleGenAI(model="models/gemini-flash-latest", api_key=google_key)
    else:
        st.error("❌ No se encontró GOOGLE_API_KEY en el archivo .env")
        st.stop()

    # 2. Configurar Embeddings locales (Siempre GRATIS)
    try:
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    except Exception as e:
        st.warning(f"⚠️ Error cargando embeddings locales: {e}. Usando configuración por defecto.")

    vector_db_dir = PROJECT_ROOT / "chroma_db"
    if not vector_db_dir.exists():
        st.warning("⚠️ Base de datos no encontrada. Por favor, indexa tus apuntes en la barra lateral.")
        return None

    chroma_client = chromadb.PersistentClient(path=str(vector_db_dir))
    collection = chroma_client.get_or_create_collection(
        name=os.getenv("CHROMA_COLLECTION", "apuntes")
    )
    vector_store = ChromaVectorStore(chroma_collection=collection)

    return VectorStoreIndex.from_vector_store(vector_store=vector_store)

def run_indexing(mode="lite"):
    """Ejecuta el proceso de indexación y muestra el progreso."""
    status_text = st.empty()
    progress_bar = st.progress(0)

    def update_progress(msg):
        status_text.text(msg)

    try:
        if mode == "vision":
            indexar.build_index(progress_callback=update_progress)
        else:
            indexar_lite.build_index(progress_callback=update_progress)

        st.success("✅ Indexación completada con éxito.")
        st.session_state.index = load_index()
        st.rerun()
    except Exception as e:
        st.error(f"❌ Error durante la indexación: {str(e)}")

def render_source(source_path, metadata, expanded=False):
    """Renderiza una fuente consultada, incluyendo imágenes y páginas de PDF."""
    full_path = PROJECT_ROOT / source_path

    # Mostrar el nombre del archivo y página si aplica
    source_label = f"📄 {source_path}"
    if "page" in metadata:
        source_label += f" (Pág. {metadata['page']})"

    with st.expander(source_label, expanded=expanded):
        # Si es una imagen standalone
        if full_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            try:
                st.image(str(full_path), width="stretch")
            except Exception:
                st.write("No se pudo cargar la imagen.")

        # Si es un PDF, renderizamos la página específica
        elif full_path.suffix.lower() == ".pdf" and "page" in metadata:
            try:
                # Usamos PyMuPDF para renderizar la página como imagen
                page_num = int(metadata["page"]) - 1  # 0-indexed
                with pymupdf.open(full_path) as doc:
                    page = doc.load_page(page_num)
                    pix = page.get_pixmap(matrix=pymupdf.Matrix(2.0, 2.0))
                    img_bytes = pix.tobytes("png")
                    st.image(img_bytes, caption=f"Previsualización de la página {metadata['page']}", width="stretch")
            except Exception as e:
                st.write(f"No se pudo previsualizar la página del PDF: {e}")

        # Si es un DOCX, intentamos extraer imágenes internas
        elif full_path.suffix.lower() == ".docx":
            try:
                st.info("📦 Extrayendo imágenes del documento Word...")
                with zipfile.ZipFile(full_path) as z:
                    # Las imágenes en DOCX suelen estar en word/media/
                    media_files = [f for f in z.namelist() if f.startswith('word/media/')]
                    if media_files:
                        cols = st.columns(min(len(media_files), 2))
                        for idx, img_path in enumerate(media_files):
                            with z.open(img_path) as f:
                                img_data = f.read()
                                cols[idx % 2].image(img_data, caption=f"Imagen {idx+1} de {source_path}", width="stretch")
                    else:
                        st.write("No se encontraron imágenes dentro de este documento Word.")
            except Exception as e:
                st.write(f"No se pudieron extraer imágenes del Word: {e}")

        st.caption(f"Tipo: {metadata.get('content_type', 'Desconocido')}")

def main():
    st.set_page_config(
        page_title="SMR Krayon | Editor de Conocimiento",
        page_icon="🤖",
        layout="wide",
    )

    # Estilo VS Code personalizado
    st.markdown("""
        <style>
        /* Fondo principal */
        .stApp {
            background-color: #1E1E1E;
        }

        /* Sidebar como Activity Bar + Explorer */
        section[data-testid="stSidebar"] {
            background-color: #252526 !important;
            border-right: 1px solid #333333;
        }

        /* Títulos y texto */
        h1, h2, h3, p, span, label {
            color: #CCCCCC !important;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }

        /* Bloques de código/fuentes */
        .stExpander {
            background-color: #2D2D2D !important;
            border: 1px solid #3E3E3E !important;
            border-radius: 4px;
        }

        /* Mensajes del chat */
        [data-testid="stChatMessage"] {
            background-color: #2D2D2D;
            border-radius: 4px;
            margin-bottom: 10px;
            border-left: 3px solid #007ACC;
        }

        /* Input del chat estilo terminal */
        .stChatInputContainer {
            background-color: #1E1E1E !important;
            padding-bottom: 20px;
        }

        /* Botones estilo VS Code */
        .stButton button {
            background-color: #0E639C !important;
            color: white !important;
            border-radius: 2px !important;
            border: none !important;
            padding: 0.2rem 1rem !important;
        }
        .stButton button:hover {
            background-color: #1177BB !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # Header estilo pestaña de editor
    st.markdown("### `index.smr` • SMR Krayon Assistant")

    # Inicializar índice con indicador de carga
    if "index" not in st.session_state:
        with st.status("🚀 Iniciando motores de IA...", expanded=True) as status:
            st.write("Cargando base de datos vectorial...")
            st.session_state.index = load_index()
            st.write("Configurando modelos de lenguaje...")
            status.update(label="✅ Sistema Listo", state="complete", expanded=False)

    index = st.session_state.index

    # Sidebar
    with st.sidebar:
        st.header("📂 Gestión de Conocimiento")

        # Listar archivos en @apuntes
        apuntes_dir = PROJECT_ROOT / "apuntes"
        if apuntes_dir.exists():
            files = [f.name for f in apuntes_dir.glob("*") if f.is_file() and not f.name.startswith(".")]
            if files:
                st.write(f"**Archivos detectados ({len(files)}):**")
                for f in files:
                    st.text(f"  • {f}")
            else:
                st.info("La carpeta `@apuntes` está vacía.")
        else:
            st.error("No se encontró la carpeta `@apuntes`.")

        st.divider()

        st.subheader("🔄 Actualizar Índice")
        col1, col2 = st.columns(2)
        if col1.button("🚀 Modo Lite", help="Extracción de texto rápida y gratuita"):
            run_indexing(mode="lite")
        if col2.button("👁️ Modo Visión", help="Analiza diagramas e imágenes (requiere API Key)"):
            run_indexing(mode="vision")

        st.divider()

        if st.button("🗑️ Limpiar Historial"):
            st.session_state.messages = []
            st.rerun()

    # Inicializar historial de chat
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Mostrar historial de chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("📚 Ver fuentes y fragmentos encontrados"):
                    for src in message["sources"]:
                        is_visual = any(ext in src["source"].lower() for ext in [".png", ".jpg", ".jpeg", ".pdf", ".docx"])
                        render_source(src["source"], src["metadata"], expanded=is_visual)

    # Input del usuario
    user_query = st.chat_input("Escribe tu duda sobre SMR...")

    if user_query:
        if not index:
            st.error("El índice no está cargado. Por favor, usa la barra lateral para indexar tus apuntes.")
            return

        # Mostrar pregunta del usuario
        with st.chat_message("user"):
            st.markdown(user_query)

        st.session_state.messages.append({"role": "user", "content": user_query})

        # Generar respuesta
        with st.chat_message("assistant"):
            with st.spinner("Consultando base de conocimientos..."):
                try:
                    # Lógica para detectar menciones @archivo
                    filters = None
                    apuntes_dir = PROJECT_ROOT / "apuntes"
                    if apuntes_dir.exists():
                        available_files = [f.name for f in apuntes_dir.glob("*") if f.is_file()]
                        for filename in available_files:
                            if f"@{filename}" in user_query:
                                st.info(f"🔍 Filtrando búsqueda solo en: `{filename}`")
                                filters = MetadataFilters(filters=[
                                    ExactMatchFilter(key="source", value=f"apuntes/{filename}")
                                ])
                                break

                    from llama_index.core import PromptTemplate

                    qa_prompt_str = (
                        "Contexto de los apuntes:\n"
                        "---------------------\n"
                        "{context_str}\n"
                        "---------------------\n"
                        "Dada la información anterior, responde a la pregunta: {query_str}\n\n"
                        "INSTRUCCIÓN PARA EL ASISTENTE SMR KRAYON:\n"
                        "Si el usuario pide ver imágenes o diagramas, confirma que has encontrado el documento "
                        "y dile que puede ver las previsualizaciones visuales justo debajo de tu respuesta. "
                        "El sistema las mostrará automáticamente. NUNCA digas que no tienes imágenes si hay contexto disponible."
                    )
                    qa_prompt = PromptTemplate(qa_prompt_str)

                    query_engine = index.as_query_engine(
                        similarity_top_k=10,
                        response_mode="compact",
                        filters=filters,
                        text_qa_template=qa_prompt
                    )

                    try:
                        response = query_engine.query(user_query)
                        answer = str(response)
                        sources_nodes = getattr(response, "source_nodes", [])
                    except Exception as ai_err:
                        err_str = str(ai_err).lower()
                        if any(x in err_str for x in ["insufficient_quota", "429", "503", "unavailable", "overloaded"]):
                            st.warning("⚠️ Los servidores de Google están saturados o sin cuota. Mostrando fragmentos encontrados directamente.")
                            # Intentar solo recuperación si falla la generación
                            retriever = index.as_retriever(similarity_top_k=5, filters=filters)
                            sources_nodes = retriever.retrieve(user_query)
                            answer = "He encontrado información relevante en tus apuntes, pero la IA de Google no puede redactar un resumen en este momento (Servidor saturado o sin cuota). Revisa las fuentes abajo para ver los diagramas y textos."
                        else:
                            raise ai_err

                    # Extraer fuentes de forma segura
                    sources = []
                    seen = set()
                    for node in sources_nodes:
                        meta = node.metadata
                        src_id = (meta.get("source"), meta.get("page"))
                        if src_id not in seen:
                            sources.append({
                                "source": meta.get("source"),
                                "metadata": meta
                            })
                            seen.add(src_id)

                    st.markdown(answer)

                    if sources:
                        with st.expander("📚 Ver fuentes y fragmentos encontrados"):
                            for src in sources:
                                is_visual = any(ext in src["source"].lower() for ext in [".png", ".jpg", ".jpeg", ".pdf", ".docx"])
                                render_source(src["source"], src["metadata"], expanded=is_visual)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })

                except Exception as e:
                    error_msg = f"Error al procesar la consulta: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })

if __name__ == "__main__":
    main()
