import os
import sys
import time
import logging
import zipfile
import io
import json
import chromadb
import pymupdf
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
from docx import Document as DocxDocument

# --- RUTAS DE CONFIGURACIÓN ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = PROJECT_ROOT / "app" / "user_config.json"
USER_AVATAR_PATH = PROJECT_ROOT / "app" / "user_avatar.png"

def save_user_profile(profile):
    """Guarda el perfil del usuario en un archivo JSON local."""
    data = {
        "name": profile["name"],
        "auto_view": profile["auto_view"],
        "avatar_type": "emoji" if isinstance(profile["avatar"], str) else "custom"
    }
    if data["avatar_type"] == "emoji":
        data["avatar_value"] = profile["avatar"]

    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def load_user_profile():
    """Carga el perfil del usuario desde el archivo local."""
    default_profile = {"name": "Estudiante SMR", "avatar": "👤", "auto_view": True}

    if not CONFIG_FILE.exists():
        return default_profile

    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        profile = {
            "name": data.get("name", "Estudiante SMR"),
            "auto_view": data.get("auto_view", True)
        }

        if data.get("avatar_type") == "custom" and USER_AVATAR_PATH.exists():
            profile["avatar"] = USER_AVATAR_PATH.read_bytes()
        else:
            profile["avatar"] = data.get("avatar_value", "👤")

        return profile
    except Exception:
        return default_profile

# --- CONFIGURACIÓN DE SILENCIO ---
os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("streamlit").setLevel(logging.ERROR)

from llama_index.core import Settings, StorageContext, VectorStoreIndex, PromptTemplate
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts import indexar_lite, indexar

def load_index() -> VectorStoreIndex:
    load_dotenv(PROJECT_ROOT / ".env")
    google_key = os.getenv("GOOGLE_API_KEY")
    if google_key:
        Settings.llm = GoogleGenAI(model="models/gemini-flash-latest", api_key=google_key)
    else:
        st.error("❌ Falta API Key")
        st.stop()
    Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db_dir = PROJECT_ROOT / "chroma_db"
    if not vector_db_dir.exists(): return None
    chroma_client = chromadb.PersistentClient(path=str(vector_db_dir))
    collection = chroma_client.get_or_create_collection(name=os.getenv("CHROMA_COLLECTION", "apuntes"))
    vector_store = ChromaVectorStore(chroma_collection=collection)
    return VectorStoreIndex.from_vector_store(vector_store=vector_store)

def run_indexing(mode="lite"):
    status_text = st.empty()
    try:
        if mode == "vision": indexar.build_index(progress_callback=status_text.text)
        else: indexar_lite.build_index(progress_callback=status_text.text)
        st.success("✅ Completado")
        st.session_state.index = load_index()
        st.rerun()
    except Exception as e: st.error(f"Error: {e}")

def render_doc_viewer(source_path, metadata):
    """Renderiza el documento en el panel central."""
    full_path = PROJECT_ROOT / source_path
    st.markdown(f"#### 📄 `{source_path.split('/')[-1]}`")

    if full_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
        st.image(str(full_path), use_container_width=True)
    elif full_path.suffix.lower() == ".pdf":
        try:
            page_num = int(metadata.get("page", 1)) - 1
            with pymupdf.open(full_path) as doc:
                page = doc.load_page(page_num)
                pix = page.get_pixmap(matrix=pymupdf.Matrix(2.0, 2.0))
                st.image(pix.tobytes("png"), use_container_width=True)
        except Exception: st.write("No se pudo previsualizar.")
    elif full_path.suffix.lower() == ".docx":
        try:
            doc = DocxDocument(full_path)
            txt = [p.text for p in doc.paragraphs if p.text.strip()]
            if txt:
                with st.container(height=300): st.markdown("\n\n".join(txt))
            with zipfile.ZipFile(full_path) as z:
                media = [f for f in z.namelist() if f.startswith('word/media/')]
                for img in media: st.image(z.open(img).read(), use_container_width=True)
        except Exception: st.write("Error cargando Word")

def main():
    st.set_page_config(
        page_title="SMR Krayon Pro",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # --- ESTILOS VS CODE PRO (NIVELACIÓN Y DISEÑO) ---
    st.markdown("""
        <style>
        /* Variables Pro */
        :root { --bg-main: #1E1E1E; --bg-side: #252526; --accent: #007ACC; --border: #333333; }

        /* Ocultar footer y menú, pero dejar el header visible y resaltar el botón lateral */
        footer, #MainMenu {visibility: hidden;}
        header[data-testid="stHeader"] {
            background: transparent !important;
            z-index: 1001 !important;
        }

        /* Hacer el botón de abrir/cerrar sidebar MUY visible */
        [data-testid="stSidebarCollapseButton"] {
            background-color: var(--accent) !important;
            color: white !important;
            border-radius: 5px !important;
            padding: 5px !important;
            margin-top: 5px !important;
            box-shadow: 0 0 15px rgba(0,122,204,0.6) !important;
        }

        .stApp { background-color: var(--bg-main); color: #cccccc; font-family: 'Segoe UI', sans-serif; }

        [data-testid="stAppViewContainer"] { height: 100vh; overflow: hidden; }
        .main .block-container {
            padding: 0 !important;
            max-width: 100% !important;
            height: 100vh !important;
            margin-top: 35px !important;
        }

        [data-testid="stHorizontalBlock"] { gap: 0 !important; }

        /* Visor (Panel 1) */
        [data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-child(1) {
            height: calc(100vh - 57px);
            overflow-y: auto;
            padding: 20px 30px !important;
            border-right: 1px solid var(--border);
        }

        /* Chat (Panel 2) */
        [data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-child(2) {
            height: calc(100vh - 57px);
            padding: 0 !important;
            background-color: #1a1a1a;
            display: flex;
            flex-direction: column;
        }

        /* Nivelar Widgets */
        h4, .stMultiSelect { margin-top: 0 !important; padding-top: 0 !important; }

        /* Contenedor fijo para el input y menciones en la parte inferior */
        .bottom-command-bar {
            position: fixed;
            bottom: 30px;
            right: 20px;
            width: 30%;
            z-index: 1001;
            background-color: #1a1a1a;
            border: 1px solid #333;
            border-radius: 4px;
            padding: 10px;
            box-shadow: 0 -5px 15px rgba(0,0,0,0.3);
        }

        /* Ajustar el tamaño de los avatares en el chat - MÁS GRANDES */
        [data-testid="stChatMessageAvatar"] {
            width: 65px !important;
            height: 65px !important;
            border-radius: 12px !important;
        }
        [data-testid="stChatMessageAvatar"] img, [data-testid="stChatMessageAvatar"] div {
            width: 65px !important;
            height: 65px !important;
            object-fit: cover !important;
            font-size: 35px !important; /* Para cuando es un emoji */
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        }

        .stChatInputContainer { width: 100% !important; position: static !important; padding: 0 !important; }

        .editor-header { position: fixed; top: 0; left: 0; right: 0; height: 35px; background: var(--bg-side); z-index: 1000; display: flex; align-items: center; padding-left: 60px; border-bottom: 1px solid var(--border); }
        .status-bar { position: fixed; bottom: 0; left: 0; right: 0; height: 22px; background: var(--accent); z-index: 1000; color: white; font-size: 11px; display: flex; align-items: center; padding: 0 10px; }
        </style>
        <div class="editor-header"><div style="background:#1E1E1E; padding:0 20px; height:100%; display:flex; align-items:center; border-top:1px solid #007ACC; color:white; font-size:12px;">🤖 krayon_workspace</div></div>
        <div class="status-bar"><span>● Connected</span><span style="margin-left:auto;">Gemini 1.5 Flash | v2.12 Pro</span></div>
    """, unsafe_allow_html=True)

    if "selected_doc" not in st.session_state: st.session_state.selected_doc = None
    if "messages" not in st.session_state: st.session_state.messages = []
    if "active_mentions" not in st.session_state: st.session_state.active_mentions = []
    if "active_tab" not in st.session_state: st.session_state.active_tab = "explorer"

    # Perfil del usuario persistente
    if "user_profile" not in st.session_state:
        st.session_state.user_profile = load_user_profile()

    # --- PREPARACIÓN DE DATOS (Común para sidebar y chat) ---
    apuntes_dir = PROJECT_ROOT / "apuntes"
    all_files = []
    if apuntes_dir.exists():
        all_files = sorted([f for f in os.listdir(apuntes_dir) if not f.startswith(".")])

    # --- PREPARACIÓN DE AVATARES ---
    ai_avatar_path = PROJECT_ROOT / "app" / "ai_avatar.png"
    # Si la imagen existe, la cargamos como objeto Image de PIL para asegurar que Streamlit la renderice
    if ai_avatar_path.exists():
        try:
            AI_AVATAR = Image.open(ai_avatar_path)
        except Exception:
            AI_AVATAR = "🤖"
    else:
        AI_AVATAR = "🤖"

    # --- SIDEBAR ---
    with st.sidebar:
        st.markdown("<br>", unsafe_allow_html=True)
        col_i, col_c = st.columns([1, 4])
        with col_i:
            if st.button("📄", key="b1", help="Explorador"): st.session_state.active_tab = "explorer"
            if st.button("⚙️", key="b2", help="Sistema"): st.session_state.active_tab = "system"
            if st.button("👤", key="b3", help="Mi Perfil"): st.session_state.active_tab = "profile"
            st.markdown("<br><br>", unsafe_allow_html=True)
            if st.button("🔴", key="b_exit", help="Cerrar Aplicación"):
                st.toast("Deteniendo servicios... Adiós 👋")
                time.sleep(1)
                os._exit(0)
        with col_c:
            if st.session_state.active_tab == "explorer":
                st.markdown("📂 **EXPLORADOR**")
                if all_files:
                    for f in all_files:
                        if st.button(f" {f}", key=f"f_{f}", use_container_width=True):
                            st.session_state.selected_doc = {"source": f"apuntes/{f}", "metadata": {}}
                            st.session_state.active_mentions = [f"@{f}"]
                else:
                    st.caption("Carpeta vacía")

            elif st.session_state.active_tab == "profile":
                st.markdown("👤 **MI PERFIL**")

                # File uploader para avatar local
                uploaded_avatar = st.file_uploader("Subir foto de perfil", type=["png", "jpg", "jpeg"], help="Sube una imagen desde tu PC para usarla como avatar.")

                with st.form("profile_form"):
                    new_name = st.text_input("Nombre de usuario", value=st.session_state.user_profile["name"])

                    # Si ha subido una imagen, mostramos aviso de que se usará esa
                    avatar_options = ["Emoji predeterminado", "👤", "👨‍💻", "👩‍💻", "🤖", "🎓", "🌟"]
                    selected_emoji = st.selectbox("O elegir Emoji", avatar_options, index=1)

                    if st.form_submit_button("Guardar cambios", use_container_width=True):
                        st.session_state.user_profile["name"] = new_name

                        # Prioridad: 1. Imagen subida, 2. Emoji seleccionado
                        if uploaded_avatar is not None:
                            img_bytes = uploaded_avatar.read()
                            st.session_state.user_profile["avatar"] = img_bytes
                            # Guardar la imagen físicamente
                            with open(USER_AVATAR_PATH, "wb") as f:
                                f.write(img_bytes)
                        elif selected_emoji != "Emoji predeterminado":
                            st.session_state.user_profile["avatar"] = selected_emoji
                            # Borrar avatar anterior si existe
                            if USER_AVATAR_PATH.exists(): USER_AVATAR_PATH.unlink()

                        # Guardar configuración en JSON
                        save_user_profile(st.session_state.user_profile)
                        st.success("¡Perfil guardado permanentemente!")
                        st.rerun()

                # Mostrar previsualización actual
                st.write("**Vista previa actual:**")
                st.chat_message("user", avatar=st.session_state.user_profile["avatar"]).write(f"Hola, soy {st.session_state.user_profile['name']}")

                st.divider()
                st.caption("Ajustes del Asistente")
                current_auto_view = st.session_state.user_profile["auto_view"]
                new_auto_view = st.checkbox(
                    "Auto-visualizar documentos",
                    value=current_auto_view,
                    help="Abre automáticamente el visor al detectar información relevante o menciones."
                )
                if new_auto_view != current_auto_view:
                    st.session_state.user_profile["auto_view"] = new_auto_view
                    save_user_profile(st.session_state.user_profile)

            else:
                st.markdown("⚙️ **SISTEMA**")
                if st.button("🚀 Re-indexar Lite", use_container_width=True): run_indexing("lite")
                if st.button("🗑️ Limpiar Chat", use_container_width=True):
                    st.session_state.messages = []
                    st.session_state.active_mentions = []
                    st.rerun()

    # --- MAIN LAYOUT ---
    v, c = st.columns([2.3, 1])

    with v:
        if st.session_state.selected_doc:
            render_doc_viewer(st.session_state.selected_doc["source"], st.session_state.selected_doc["metadata"])
        else:
            st.info("Selecciona un archivo de la izquierda o usa la barra de comandos para comenzar.")

    with c:
        # 1. Contenedor de chat con scroll ajustable
        # Calculamos una altura que intente llenar el panel derecho
        chat_box = st.container(height=720) # Aumentado para reducir el hueco inferior
        with chat_box:
            for msg in st.session_state.messages:
                avatar = st.session_state.user_profile["avatar"] if msg["role"] == "user" else AI_AVATAR
                with st.chat_message(msg["role"], avatar=avatar):
                    st.markdown(msg["content"])

        # 2. BARRA DE COMANDOS FLOTANTE (Anclada al fondo de la columna)
        # Eliminamos el div de altura fija que creaba el espacio vacío

        # Esta sección se queda fija abajo mediante el CSS de .stChatInputContainer y el padding de .chat-panel
        with st.container():
            st.caption("🔍 Menciona archivos con @")
            display_options = [f"@{f}" for f in all_files]
            sel_mentions = st.multiselect(
                "mentions",
                options=display_options,
                default=st.session_state.active_mentions,
                placeholder="Escribe @ para filtrar...",
                label_visibility="collapsed",
                key="chat_mentions_bottom"
            )

            query = st.chat_input("Escribe tu duda... (Usa el cuadro de arriba para el @)")

            if query:
                st.session_state.messages.append({"role": "user", "content": query})
                with chat_box:
                    with st.chat_message("user", avatar=st.session_state.user_profile["avatar"]):
                        st.markdown(query)
                    with st.chat_message("assistant", avatar=AI_AVATAR):
                        idx = st.session_state.get("index") or load_index()
                        st.session_state.index = idx
                        with st.spinner("..."):
                            try:
                                f_filter = None
                                all_mentions = [m.lstrip("@") for m in sel_mentions]
                                for name in all_files:
                                    if f"@{name}" in query: all_mentions.append(name)

                                if all_mentions:
                                    target = all_mentions[0]
                                    st.caption(f"🎯 Contexto: `{target}`")
                                    f_filter = MetadataFilters(filters=[ExactMatchFilter(key="source", value=f"apuntes/{target}")])

                                eng = idx.as_query_engine(similarity_top_k=5, filters=f_filter)

                                try:
                                    res = eng.query(query)
                                    answer = str(res)
                                    sources_nodes = getattr(res, "source_nodes", [])
                                except Exception as ai_err:
                                    err_msg = str(ai_err).lower()
                                    if any(x in err_msg for x in ["503", "429", "unavailable", "overloaded", "quota"]):
                                        answer = "⚠️ **Servidor de Google saturado.**\n\nNo puedo redactar una respuesta ahora mismo, pero he localizado los documentos relevantes en tus apuntes. Échales un vistazo en el visor de la izquierda."
                                        # Modo Supervivencia: Recuperar fragmentos sin generar texto
                                        retriever = idx.as_retriever(similarity_top_k=3, filters=f_filter)
                                        sources_nodes = retriever.retrieve(query)
                                    else:
                                        raise ai_err

                                st.markdown(answer)
                                st.session_state.messages.append({"role": "assistant", "content": answer})

                                # --- LÓGICA DE APERTURA INTELIGENTE (Incluye Survival Mode) ---
                                if sources_nodes and st.session_state.user_profile.get("auto_view", True):
                                    # Abrir siempre si es Modo Supervivencia o mención específica con @
                                    is_survival = "saturado" in answer

                                    if is_survival or f_filter is not None:
                                        m = sources_nodes[0].metadata
                                        st.session_state.selected_doc = {"source": m["source"], "metadata": m}
                                        st.session_state.active_mentions = []
                                        st.rerun()

                                    # Para búsquedas generales, ignorar saludos
                                    else:
                                        greetings = ["hola", "buenas", "que tal", "quién eres", "ayuda"]
                                        is_greeting = any(g in query.lower() for g in greetings)
                                        if len(query) > 20 and not is_greeting:
                                            m = sources_nodes[0].metadata
                                            st.session_state.selected_doc = {"source": m["source"], "metadata": m}
                                            st.rerun()
                            except Exception as e: st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
