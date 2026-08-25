import os
import sys
from pathlib import Path

import chromadb
import streamlit as st
from dotenv import load_dotenv
from llama_index.core import Settings, StorageContext, VectorStoreIndex
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

def render_source(source_path, metadata):
    """Renderiza una fuente consultada, incluyendo imágenes si existen."""
    full_path = PROJECT_ROOT / source_path

    # Mostrar el nombre del archivo y página si aplica
    source_label = f"📄 {source_path}"
    if "page" in metadata:
        source_label += f" (Pág. {metadata['page']})"

    with st.expander(source_label):
        # Si es una imagen standalone
        if full_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            try:
                st.image(str(full_path), use_container_width=True)
            except Exception:
                st.write("No se pudo cargar la imagen.")

        # Si es un PDF y tenemos visión (metadata indica que es pdf_page_with_vision)
        elif metadata.get("content_type") == "pdf_page_with_vision":
            st.info("💡 Esta página fue analizada con visión artificial.")

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
            if "sources" in message:
                st.markdown("---")
                st.markdown("**Fuentes consultadas:**")
                for src in message["sources"]:
                    render_source(src["source"], src["metadata"])

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
                    query_engine = index.as_query_engine(
                        similarity_top_k=5,
                        response_mode="compact",
                    )

                    try:
                        response = query_engine.query(user_query)
                        answer = str(response)
                        sources_nodes = getattr(response, "source_nodes", [])
                    except Exception as ai_err:
                        if "insufficient_quota" in str(ai_err).lower() or "429" in str(ai_err):
                            st.warning("⚠️ Límite de cuota alcanzado o error de saldo. Mostrando fragmentos encontrados.")
                            retriever = index.as_retriever(similarity_top_k=5)
                            sources_nodes = retriever.retrieve(user_query)
                            answer = "He encontrado información en tus apuntes, pero la IA no puede generar un resumen por falta de cuota/saldo."
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
                        st.markdown("---")
                        st.markdown("**Fuentes y fragmentos encontrados:**")
                        for src in sources:
                            render_source(src["source"], src["metadata"])

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
