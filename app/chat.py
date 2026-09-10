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
HISTORY_DIR = PROJECT_ROOT / "app" / "historial"
HISTORY_DIR.mkdir(exist_ok=True)

def save_conversation(conv_id, messages):
    """Guarda una conversación en un archivo JSON."""
    file_path = HISTORY_DIR / f"{conv_id}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=4)

def load_conversation(conv_id):
    """Carga una conversación desde un archivo JSON."""
    file_path = HISTORY_DIR / f"{conv_id}.json"
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def list_conversations():
    """Lista los IDs de las conversaciones guardadas."""
    return sorted([f.stem for f in HISTORY_DIR.glob("*.json")], reverse=True)

def delete_conversation(conv_id):
    """Elimina una conversación guardada."""
    file_path = HISTORY_DIR / f"{conv_id}.json"
    if file_path.exists():
        file_path.unlink()

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
# Workaround para errores de Protobuf en versiones nuevas de Python
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("streamlit").setLevel(logging.ERROR)

from llama_index.core import Settings, StorageContext, VectorStoreIndex, PromptTemplate
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
import google.generativeai as genai
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.google import GeminiEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts import indexar_lite, indexar

def load_index() -> VectorStoreIndex:
    load_dotenv(PROJECT_ROOT / ".env")
    google_key = os.getenv("GOOGLE_API_KEY")
    if not google_key:
        st.error("❌ Falta API Key en el archivo .env")
        st.stop()

    # Forzamos la configuración global del SDK para aceptar claves con formato AQ.
    genai.configure(api_key=google_key)

    # Usamos el modelo que hemos verificado que está disponible
    Settings.embed_model = GeminiEmbedding(model_name="models/gemini-embedding-001", api_key=google_key)
    Settings.llm = Gemini(model_name="models/gemini-flash-latest", api_key=google_key)

    vector_db_dir = PROJECT_ROOT / "chroma_db"
    if not vector_db_dir.exists():
        return None

    try:
        # Configuración optimizada para evitar bloqueos de Windows Defender
        chroma_client = chromadb.PersistentClient(
            path=str(vector_db_dir),
            settings=chromadb.Settings(anonymized_telemetry=False, is_persistent=True)
        )
        collection = chroma_client.get_or_create_collection(name=os.getenv("CHROMA_COLLECTION", "apuntes"))

        # Si la colección está vacía, retornamos None para forzar la re-indexación
        if collection.count() == 0:
            return None

        vector_store = ChromaVectorStore(chroma_collection=collection)
        return VectorStoreIndex.from_vector_store(vector_store=vector_store)
    except Exception as e:
        st.error(f"Error en base de datos: {e}")
        return None

def run_indexing(mode="lite"):
    status_text = st.empty()
    try:
        if mode == "vision": indexar.build_index(progress_callback=status_text.text)
        else: indexar_lite.build_index(progress_callback=status_text.text)
        st.balloons()
        st.success("✅ ¡Proceso de indexación completado con éxito!")
        st.session_state.index = load_index()
        st.rerun()
    except Exception as e: st.error(f"Error: {e}")

def render_doc_viewer(source_path, metadata):
    """Renderiza el contenido del documento en el panel central."""
    full_path = PROJECT_ROOT / source_path

    # Cabecera del visor con botón de cerrar
    c1, c2 = st.columns([0.9, 0.1])
    c1.markdown(f"#### 📄 `{source_path.split('/')[-1]}`")
    if c2.button("✕", key="close_doc_btn", help="Cerrar visor"):
        st.session_state.selected_doc = None
        st.rerun()

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
                st.markdown(f"""
                    <div class="docx-paper">
                        {"<br><br>".join(txt)}
                    </div>
                """, unsafe_allow_html=True)

            with zipfile.ZipFile(full_path) as z:
                media = [f for f in z.namelist() if f.startswith('word/media/')]
                if media:
                    st.divider()
                    st.caption("🖼️ Imágenes del documento:")
                    for img in media: st.image(z.open(img).read(), use_container_width=True)
        except Exception: st.write("Error cargando Word")

def main():
    st.set_page_config(
        page_title="SMR Krayon Pro",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # --- PREPARACIÓN DE DATOS ---
    apuntes_dir = PROJECT_ROOT / "apuntes"
    all_files = []
    if apuntes_dir.exists():
        all_files = sorted([f for f in os.listdir(apuntes_dir) if f.endswith(('.pdf', '.docx')) and not f.startswith((".", "~$"))])

    # --- AUTO-INDEXACIÓN INTELIGENTE ---
    if "auto_indexed" not in st.session_state:
        # Solo comprobar si la DB está lista
        def is_db_ready():
            try:
                db_dir = PROJECT_ROOT / "chroma_db"
                if not db_dir.exists(): return False
                client = chromadb.PersistentClient(path=str(db_dir), settings=chromadb.Settings(anonymized_telemetry=False, is_persistent=True))
                collection = client.get_collection(name=os.getenv("CHROMA_COLLECTION", "apuntes"))
                return collection.count() > 0
            except: return False

        if not is_db_ready():
            status_container = st.empty()
            with status_container.container():
                st.info("🚀 Sincronizando base de conocimientos...")
                progress_bar = st.progress(0)
                status_text = st.empty()

                try:
                    def update_status(msg):
                        status_text.write(f"**{msg}**")
                        # Actualizar barra de progreso si es posible
                        if "[" in msg and "/" in msg:
                            try:
                                ratio = msg.split("[")[1].split("]")[0].split(" ")[-1]
                                curr, tot = map(int, ratio.split("/"))
                                progress_bar.progress(curr / tot)
                            except: pass

                    indexar_lite.build_index(progress_callback=update_status)
                    st.session_state.index = load_index()
                    st.session_state.auto_indexed = True
                    status_container.empty()
                    st.toast("✅ Base de conocimientos lista.")
                except Exception as e:
                    st.error(f"Error en la sincronización: {e}")
        else:
            st.session_state.index = load_index()
            st.session_state.auto_indexed = True

    # --- ESTILOS VS CODE PRO (NIVELACIÓN Y DISEÑO) ---
    st.markdown("""
        <style>
        /* Variables Modernas */
        :root {
            --bg-dark: #0d1117;
            --bg-sidebar: #010409;
            --bg-card: #161b22;
            --bg-hover: #21262d;
            --accent: #2f81f7;
            --text-main: #c9d1d9;
            --text-dim: #8b949e;
            --border: #30363d;
        }

        /* Aplicación Global */
        .stApp { background-color: var(--bg-dark); color: var(--text-main); font-family: 'Segoe UI', sans-serif; }

        /* Sidebar Estilo VS Code */
        [data-testid="stSidebar"] {
            background-color: var(--bg-sidebar) !important;
            border-right: 1px solid var(--border) !important;
        }

        /* ARREGLO DE BOTONES SIDEBAR (Adiós bloques blancos y cortes) */
        [data-testid="stSidebar"] .stButton button {
            background-color: transparent !important;
            color: var(--text-dim) !important;
            border: 1px solid transparent !important;
            border-radius: 6px !important;
            width: 100% !important;
            text-align: left !important;
            padding: 10px 8px !important; /* Reducido padding lateral */
            font-size: 16px !important; /* Un poco más grandes para que no se corten los emojis */
            transition: all 0.2s ease !important;
            margin-bottom: 4px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important; /* Centrar contenido del botón */
        }

        /* Ajuste específico para los botones con texto (en la columna de la derecha) */
        [data-testid="stSidebar"] [data-testid="column"]:nth-child(2) .stButton button {
            justify-content: flex-start !important;
            font-size: 14px !important;
        }
        [data-testid="stSidebar"] .stButton button:hover {
            background-color: var(--bg-hover) !important;
            color: var(--text-main) !important;
            border-color: var(--border) !important;
        }

        /* Botón de Salida */
        button[key="b_exit"] {
            color: #f85149 !important;
        }

        /* === ELIMINACIÓN DE COLORES ROJOS Y GRISES === */

        /* Asegurar que las menciones se vean bien con el nuevo tema */
        div[data-baseweb="tag"] {
            background-color: var(--accent) !important;
            color: white !important;
        }

        /* Chat Input - BLANCO PURO PARA EL TEXTO Y ELIMINAR ROJO */
        [data-testid="stChatInput"] {
            border: 1px solid var(--border) !important;
            background-color: var(--bg-card) !important;
        }
        [data-testid="stChatInput"] textarea {
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
            background-color: transparent !important;
        }
        [data-testid="stChatInput"]:focus-within {
            border-color: var(--accent) !important;
            box-shadow: 0 0 0 1px var(--accent) !important;
        }

        /* Botón de enviar (Icono) - AZUL */
        [data-testid="stChatInputButton"] svg {
            fill: var(--accent) !important;
        }

        /* AVATARES MÁS GRANDES */
        [data-testid="stChatMessageAvatar"] {
            width: 80px !important;
            height: 80px !important;
            border-radius: 12px !important;
            background: var(--bg-card) !important;
            border: 1px solid var(--border) !important;
        }
        [data-testid="stChatMessageAvatar"] img, [data-testid="stChatMessageAvatar"] div {
            width: 80px !important;
            height: 80px !important;
            object-fit: cover !important;
        }
        [data-testid="stChatMessage"] {
            padding-top: 2rem !important;
            padding-bottom: 2rem !important;
        }

        /* Contenedores y Layout */
        [data-testid="stAppViewContainer"] { height: 100vh; overflow: hidden; }
        .main .block-container {
            padding: 0 !important;
            max-width: 100% !important;
            margin-top: 35px !important;
        }

        /* Paneles */
        [data-testid="column"]:nth-child(1) {
            height: calc(100vh - 57px);
            overflow-y: auto;
            border-right: 1px solid var(--border);
            padding: 30px !important;
        }
        [data-testid="column"]:nth-child(2) {
            height: calc(100vh - 57px);
            background-color: #0d1117;
            padding: 0 !important;
        }

        /* Visor de Documentos (Papel Limpio) */
        .docx-paper {
            background-color: #ffffff;
            color: #1f2328;
            padding: 60px !important;
            border-radius: 4px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            font-family: 'Segoe UI', sans-serif;
            line-height: 1.6;
            max-width: 800px;
            margin: 0 auto;
        }

        /* Header y Status Bar */
        footer, #MainMenu {visibility: hidden;}
        header[data-testid="stHeader"] { background: transparent !important; }

        .editor-header {
            position: fixed; top: 0; left: 0; right: 0; height: 35px;
            background: var(--bg-sidebar); z-index: 1000; display: flex;
            align-items: center; padding-left: 60px; border-bottom: 1px solid var(--border);
            color: var(--text-dim); font-size: 12px;
        }
        .status-bar {
            position: fixed; bottom: 0; left: 0; right: 0; height: 22px;
            background: var(--bg-sidebar); z-index: 1000; color: var(--text-dim);
            font-size: 11px; display: flex; align-items: center; padding: 0 10px;
            border-top: 1px solid var(--border);
        }
        </style>
        <div class="editor-header">
            <span style="color:var(--accent)">●</span> krayon_workspace / <b>SMR_Krayon_Pro</b>
        </div>
        <div class="status-bar"><span>● Connected</span><span style="margin-left:auto;">Gemini 1.5 Flash | v3.1</span></div>
    """, unsafe_allow_html=True)

    if "selected_doc" not in st.session_state: st.session_state.selected_doc = None

    # Inicializar historial con mensaje motivacional si está vacío
    if "messages" not in st.session_state or not st.session_state.messages:
        st.session_state.messages = [{
            "role": "assistant",
            "content": "✨ ¡Hola! Soy tu Well, Actually. Recuerda: *'La mejor forma de predecir el futuro es inventándolo'*. Estoy aquí para ayudarte a dominar tus apuntes. ¿Por dónde empezamos hoy? 🚀"
        }]

    if "active_mentions" not in st.session_state: st.session_state.active_mentions = []
    if "active_tab" not in st.session_state: st.session_state.active_tab = "explorer"
    if "chat_key" not in st.session_state: st.session_state.chat_key = 0
    if "current_conv_id" not in st.session_state:
        st.session_state.current_conv_id = time.strftime("%Y%m%d_%H%M%S")

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
        col_i, col_c = st.columns([1.2, 3.8])
        with col_i:
            if st.button("📁", key="b1", help="Explorador"): st.session_state.active_tab = "explorer"
            if st.button("💬", key="b_chat", help="Historial"): st.session_state.active_tab = "history"
            if st.button("⚙️", key="b2", help="Sistema"): st.session_state.active_tab = "system"
            if st.button("👤", key="b3", help="Perfil"): st.session_state.active_tab = "profile"
            st.markdown("<br><br>", unsafe_allow_html=True)
            if st.button("❌", key="b_exit", help="Salir"):
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

            elif st.session_state.active_tab == "history":
                st.markdown("💬 **HISTORIAL**")
                if st.button("➕ Nueva Conversación", use_container_width=True):
                    st.session_state.current_conv_id = time.strftime("%Y%m%d_%H%M%S")
                    st.session_state.messages = [{
                        "role": "assistant",
                        "content": "✨ ¡Hola! Nueva conversación iniciada. ¿En qué puedo ayudarte?"
                    }]
                    st.session_state.chat_key += 1
                    st.rerun()

                st.divider()
                convs = list_conversations()
                if convs:
                    for c_id in convs:
                        col_text, col_del = st.columns([0.8, 0.2])
                        # Formatear fecha para mostrar
                        display_name = c_id.replace("_", " ")
                        if col_text.button(f" {display_name}", key=f"conv_{c_id}", use_container_width=True):
                            st.session_state.current_conv_id = c_id
                            st.session_state.messages = load_conversation(c_id)
                            st.session_state.chat_key += 1
                            st.rerun()
                        if col_del.button("🗑️", key=f"del_{c_id}"):
                            delete_conversation(c_id)
                            if st.session_state.current_conv_id == c_id:
                                st.session_state.current_conv_id = time.strftime("%Y%m%d_%H%M%S")
                                st.session_state.messages = []
                            st.rerun()
                else:
                    st.caption("No hay chats guardados.")

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
                    st.session_state.messages = [{
                        "role": "assistant",
                        "content": "✨ ¡Chat reiniciado! Nueva oportunidad para aprender algo increíble. ¿En qué trabajamos ahora? 💻"
                    }]
                    st.session_state.active_mentions = []
                    st.session_state.chat_key += 1
                    st.rerun()

                st.divider()
                st.markdown("🌐 **COMPARTIR**")
                try:
                    import socket
                    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    s.connect(("8.8.8.8", 80))
                    local_ip = s.getsockname()[0]
                    s.close()
                    st.write(f"🏠 **IP Local (Mismo Wi-Fi):**")
                    st.code(f"http://{local_ip}:8501")
                except:
                    st.caption("No se pudo obtener la IP local.")

                st.write(f"🌍 **A internet (Otras casas):**")
                st.info("Usa localtunnel para saltar el Firewall:")
                st.code("npx localtunnel --port 8501")

    # --- MAIN LAYOUT ---
    v, c = st.columns([1, 1])

    with v:
        if st.session_state.selected_doc:
            render_doc_viewer(st.session_state.selected_doc["source"], st.session_state.selected_doc["metadata"])
        else:
            st.info("Selecciona un archivo de la izquierda o usa la barra de comandos para comenzar.")

    with c:
        # 1. Contenedor de chat con scroll maximizado
        # Usamos un key dinámico para forzar la limpieza del DOM y evitar el error removeChild
        chat_box = st.container(height=750, key=f"chat_box_{st.session_state.chat_key}")
        with chat_box:
            for msg in st.session_state.messages:
                avatar = st.session_state.user_profile["avatar"] if msg["role"] == "user" else AI_AVATAR
                with st.chat_message(msg["role"], avatar=avatar):
                    st.markdown(msg["content"])

        # 2. BARRA DE COMANDOS (Fija abajo)
        with st.container():
            st.caption("🔍 Menciona archivos con @")
            display_options = [f"@{f}" for f in all_files]
            sel_mentions = st.multiselect(
                "mentions",
                options=display_options,
                default=st.session_state.active_mentions,
                placeholder="Escribe @ para filtrar...",
                label_visibility="collapsed",
                key="chat_mentions_v4"
            )

            query = st.chat_input("Escribe tu duda... (Usa el cuadro de arriba para el @)")

            if query:
                st.session_state.messages.append({"role": "user", "content": query})
                save_conversation(st.session_state.current_conv_id, st.session_state.messages)

                # --- RESPUESTA RÁPIDA A SALUDOS (Ahorro de cuota) ---
                saludos = ["hola", "buenas", "buenos dias", "buenas tardes", "hola!", "hola?", "ey", "hi", "hello"]
                user_name = st.session_state.user_profile.get("name", "Estudiante")

                query_clean = query.lower().strip().strip("!").strip("?")

                if query_clean in saludos:
                    res_hola = f"¡Hola, {user_name}! 👋 Soy tu Well, Actually. Estoy listo para ayudarte con tus apuntes. ¿Qué quieres repasar hoy?"
                    st.session_state.messages.append({"role": "assistant", "content": res_hola})
                    save_conversation(st.session_state.current_conv_id, st.session_state.messages)
                    st.rerun()

                if "mi nombre" in query_clean or "como me llamo" in query_clean or "quien soy" in query_clean:
                    res_name = f"Te llamas **{user_name}**. ¡Un placer saludarte de nuevo! 😊 ¿Necesitas ayuda con algún tema de SMR?"
                    st.session_state.messages.append({"role": "assistant", "content": res_name})
                    save_conversation(st.session_state.current_conv_id, st.session_state.messages)
                    st.rerun()

                # --- DETECCIÓN DE RESPUESTA A PREGUNTA DE INTERNET ---
                if "internet_query" in st.session_state and st.session_state.internet_query:
                    if any(word in query.lower() for word in ["si", "sí", "claro", "vale", "adelante", "busca"]):
                        target_query = st.session_state.internet_query
                        with chat_box:
                            with st.chat_message("user", avatar=st.session_state.user_profile["avatar"]): st.markdown(query)
                            with st.chat_message("assistant", avatar=AI_AVATAR):
                                with st.spinner("Buscando en internet..."):
                                    try:
                                        gen_prompt = f"El usuario quiere saber: '{target_query}'. Responde usando tu conocimiento general como experto SMR."
                                        response = Settings.llm.complete(gen_prompt)
                                        st.markdown(str(response))
                                        st.session_state.messages.append({"role": "assistant", "content": str(response)})
                                        save_conversation(st.session_state.current_conv_id, st.session_state.messages)
                                        st.session_state.internet_query = None
                                        st.rerun()
                                    except Exception as e: st.error(f"Error en internet: {e}")
                        return
                    else:
                        st.session_state.internet_query = None

                # --- FLUJO NORMAL ---
                with chat_box:
                    with st.chat_message("user", avatar=st.session_state.user_profile["avatar"]):
                        st.markdown(query)
                    with st.chat_message("assistant", avatar=AI_AVATAR):
                        idx = st.session_state.get("index") or load_index()
                        st.session_state.index = idx

                        if idx is None:
                            st.warning("⚠️ No hay documentos indexados en la base de datos de la IA.")
                            st.info("Para activarla: ve al icono de **Sistema** (⚙️) en la barra lateral y pulsa **🚀 Re-indexar Lite**.")
                            st.stop()

                        with st.spinner("..."):
                            try:
                                f_filter = None
                                # Limpiar menciones y buscar archivos reales
                                all_mentions = [m.lstrip("@") for m in sel_mentions]
                                for name in all_files:
                                    if f"@{name}" in query:
                                        if name not in all_mentions: all_mentions.append(name)

                                if all_mentions:
                                    target = all_mentions[0]
                                    # Intentar coincidencia exacta o por nombre base (sin extensión)
                                    base_target = target.rsplit(".", 1)[0]

                                    # Ver si el archivo está indexado comprobando la base de datos
                                    is_indexed = False
                                    try:
                                        # Comprobamos si hay algún nodo con ese origen
                                        test_filter = MetadataFilters(filters=[ExactMatchFilter(key="source", value=f"apuntes/{target}")])
                                        test_retriever = idx.as_retriever(similarity_top_k=1, filters=test_filter)
                                        if test_retriever.retrieve("test"):
                                            is_indexed = True
                                            f_filter = test_filter
                                    except: pass

                                    if not is_indexed:
                                        # Si no encontramos el .docx, probamos con el .pdf equivalente
                                        other_ext = ".pdf" if target.endswith(".docx") else ".docx"
                                        alt_target = base_target + other_ext
                                        try:
                                            alt_filter = MetadataFilters(filters=[ExactMatchFilter(key="source", value=f"apuntes/{alt_target}")])
                                            alt_retriever = idx.as_retriever(similarity_top_k=1, filters=alt_filter)
                                            if alt_retriever.retrieve("test"):
                                                is_indexed = True
                                                f_filter = alt_filter
                                                st.caption(f"ℹ️ Usando versión {other_ext} de los apuntes.")
                                        except: pass

                                    if not is_indexed:
                                        st.warning(f"⚠️ El archivo '{target}' aún no ha sido procesado por la IA.")
                                        st.info("Ve a **Sistema (⚙️)** -> **Re-indexar Lite** para activarlo.")
                                        st.stop()

                                # Generar contexto del historial (memoria)
                                history_str = ""
                                if len(st.session_state.messages) > 1:
                                    # Tomamos los últimos 6 mensajes para no saturar la cuota
                                    last_msgs = st.session_state.messages[-7:-1]
                                    history_str = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in last_msgs])

                                qa_prompt = PromptTemplate(
                                    "Eres el Well, Actually, experto en Sistemas Microinformáticos y Redes. "
                                    "Tu misión es ayudar al alumno usando los APUNTES proporcionados y recordando la CONVERSACIÓN ACTUAL.\n\n"
                                    "HISTORIAL DE LA CONVERSACIÓN:\n"
                                    f"{history_str}\n\n"
                                    "REGLAS:\n"
                                    "1. Usa el CONTEXTO de los apuntes para temas técnicos.\n"
                                    "2. Usa el HISTORIAL para responder preguntas sobre la charla (ej: '¿qué te pregunté antes?').\n"
                                    "3. Si el usuario pregunta algo personal que ya dijo (nombre, etc), responde usando el historial.\n"
                                    "4. Si realmente no hay información en ningún sitio, di: 'NO_DATA'.\n\n"
                                    "CONTEXTO DE LOS APUNTES:\n{context_str}\n\n"
                                    "PREGUNTA ACTUAL DEL ALUMNO: {query_str}"
                                )

                                # Aumentamos la calidad ahorrando tokens (top_k=6 con texto limpio)
                                eng = idx.as_query_engine(similarity_top_k=6, filters=f_filter, text_qa_template=qa_prompt)

                                try:
                                    # Añadimos un reintento automático para errores de cuota temporales
                                    max_retries = 1
                                    for attempt in range(max_retries + 1):
                                        try:
                                            res = eng.query(query)
                                            break
                                        except Exception as e:
                                            if attempt < max_retries and any(x in str(e).upper() for x in ["429", "503", "LIMIT"]):
                                                time.sleep(2) # Espera corta y reintento
                                                continue
                                            raise e

                                    answer = str(res)
                                    sources_nodes = getattr(res, "source_nodes", [])
                                except Exception as ai_err:
                                    err_msg = str(ai_err).upper()
                                    if any(x in err_msg for x in ["503", "429", "RESOURCE_EXHAUSTED", "LIMIT"]):
                                        answer = "⚠️ **La IA está tomando un respiro (Cuota agotada).**\n\nComo acabamos de procesar muchos documentos, Google nos pide esperar unos segundos. Inténtalo de nuevo en un momento o consulta los documentos directamente en el visor."
                                        retriever = idx.as_retriever(similarity_top_k=3, filters=f_filter)
                                        sources_nodes = retriever.retrieve(query)
                                    else:
                                        raise ai_err

                                if "NO_DATA" in answer or not answer or answer == "Empty Response":
                                    st.session_state.internet_query = query
                                    st.warning("🧐 No he encontrado información específica en los apuntes seleccionados.")
                                    msg = "¿Quieres que busque en internet por ti para darte una respuesta general?"
                                    st.markdown(msg)
                                    st.session_state.messages.append({"role": "assistant", "content": msg})
                                else:
                                    st.markdown(answer)
                                    st.session_state.messages.append({"role": "assistant", "content": answer})
                                    save_conversation(st.session_state.current_conv_id, st.session_state.messages)

                                    if sources_nodes and st.session_state.user_profile.get("auto_view", True):
                                        # Inteligencia: No abrir archivos si es una pregunta sobre la charla o personal
                                        chat_keywords = [
                                            "te pregunté", "te preguntaste", "dije antes", "mi nombre",
                                            "quien soy", "hola", "gracias", "adios", "chao", "que tal",
                                            "recordar", "memoria", "conversacion", "chat"
                                        ]
                                        is_chat_query = any(kw in query.lower() for kw in chat_keywords)

                                        # Solo auto-visualizar si es una pregunta técnica real (larga o con filtro)
                                        if not is_chat_query and (f_filter is not None or len(query) > 25):
                                            m = sources_nodes[0].metadata
                                            st.session_state.selected_doc = {"source": m["source"], "metadata": m}
                                            st.rerun()

                            except Exception as e: st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
