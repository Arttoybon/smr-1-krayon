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
# Workaround para errores de Protobuf en versiones nuevas de Python
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
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

    # Usamos modelos Flash-Latest que tienen mayor cuota gratuita
    Settings.llm = Gemini(model_name="models/gemini-flash-latest", api_key=google_key)
    Settings.embed_model = GeminiEmbedding(model_name="models/gemini-embedding-001", api_key=google_key)

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
        st.success("✅ Completado")
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
                # Estilo de hoja de papel para los DOCX
                st.markdown(f"""
                    <div style="background-color: white; color: #1e1e1e; padding: 25px; border-radius: 2px;
                         box-shadow: 0 5px 15px rgba(0,0,0,0.3); font-family: 'Segoe UI', sans-serif;
                         line-height: 1.6; margin-bottom: 20px; font-size: 14px;">
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

    # --- AUTO-INDEXACIÓN INTELIGENTE (A prueba de F5) ---
    if "auto_indexed" not in st.session_state:
        apuntes_dir = PROJECT_ROOT / "apuntes"
        files_on_disk = [f for f in os.listdir(apuntes_dir) if f.endswith(('.pdf', '.docx')) and not f.startswith("~$")]

        # Función para verificar integridad de la DB
        def count_indexed_files():
            try:
                db_dir = PROJECT_ROOT / "chroma_db"
                if not db_dir.exists(): return 0
                client = chromadb.PersistentClient(path=str(db_dir), settings=chromadb.Settings(anonymized_telemetry=False, is_persistent=True))
                collection = client.get_collection(name=os.getenv("CHROMA_COLLECTION", "apuntes"))
                res = collection.get(include=['metadatas'])
                if not res or not res['metadatas']: return 0
                return len(set(m['source'].split('/')[-1] for m in res['metadatas'] if m))
            except: return 0

        indexed_count = count_indexed_files()

        # Si faltan archivos o la DB está vacía, forzar sincronización
        if indexed_count < len(files_on_disk):
            status_container = st.empty()
            with status_container.container():
                st.info(f"🚀 Sincronizando base de conocimientos ({indexed_count}/{len(files_on_disk)} archivos listos)...")
                progress_bar = st.progress(indexed_count / len(files_on_disk) if len(files_on_disk) > 0 else 0)
                status_text = st.empty()

                try:
                    def update_status(msg):
                        try:
                            if "[" in msg and "/" in msg:
                                parts = msg.split("[")[1].split("]")[0].split(" ")
                                ratio = parts[-1] if "/" in parts[-1] else parts[1]
                                curr, tot = map(int, ratio.split("/"))
                                if "Convirtiendo" in msg: p = (curr / tot) * 0.3
                                else: p = 0.3 + (curr / tot) * 0.7
                                progress_bar.progress(min(p, 1.0))
                        except: pass
                        status_text.markdown(f"**{msg}**")

                    indexar_lite.build_index(progress_callback=update_status)
                    st.session_state.index = load_index()
                    st.session_state.auto_indexed = True
                    status_container.empty()
                    st.toast("✅ Base de conocimientos actualizada.")
                except Exception as e:
                    st.error(f"Error en la sincronización: {e}")
        else:
            st.session_state.index = load_index()
            st.session_state.auto_indexed = True

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

    # Inicializar historial con mensaje motivacional si está vacío
    if "messages" not in st.session_state or not st.session_state.messages:
        st.session_state.messages = [{
            "role": "assistant",
            "content": "✨ ¡Hola! Soy tu Asistente SMR Krayon. Recuerda: *'La mejor forma de predecir el futuro es inventándolo'*. Estoy aquí para ayudarte a dominar tus apuntes. ¿Por dónde empezamos hoy? 🚀"
        }]

    if "active_mentions" not in st.session_state: st.session_state.active_mentions = []
    if "active_tab" not in st.session_state: st.session_state.active_tab = "explorer"
    if "chat_key" not in st.session_state: st.session_state.chat_key = 0

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

                                qa_prompt = PromptTemplate(
                                    "Eres el Asistente SMR Krayon, experto en Sistemas Microinformáticos y Redes. "
                                    "Tu misión es ayudar al alumno usando los APUNTES proporcionados.\n\n"
                                    "REGLAS:\n"
                                    "1. Usa el CONTEXTO de abajo para responder.\n"
                                    "2. Si el usuario pide un 'resumen', analiza todo el contexto y destaca los puntos clave.\n"
                                    "3. Si realmente no hay nada de información sobre el tema, di: 'NO_DATA'.\n\n"
                                    "CONTEXTO DE LOS APUNTES:\n{context_str}\n\n"
                                    "PREGUNTA DEL ALUMNO: {query_str}"
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
                                        answer = "⚠️ **Servidor saturado o cuota agotada.**\n\nHe localizado los documentos relevantes. Échales un vistazo en el visor."
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

                                    if sources_nodes and st.session_state.user_profile.get("auto_view", True):
                                        if f_filter is not None or (len(query) > 20 and "hola" not in query.lower()):
                                            m = sources_nodes[0].metadata
                                            st.session_state.selected_doc = {"source": m["source"], "metadata": m}
                                            st.rerun()

                            except Exception as e: st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
