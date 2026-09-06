import os
import chromadb
from dotenv import load_dotenv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

db_dir = PROJECT_ROOT / "chroma_db"
if not db_dir.exists():
    print("❌ La carpeta chroma_db no existe.")
else:
    try:
        client = chromadb.PersistentClient(path=str(db_dir))
        collection_name = os.getenv("CHROMA_COLLECTION", "apuntes")
        collection = client.get_collection(name=collection_name)
        count = collection.count()
        print(f"✅ La colección '{collection_name}' tiene {count} fragmentos indexados.")

        # Listar fuentes únicas
        metadatas = collection.get(include=['metadatas'])['metadatas']
        sources = set(m.get('source') for m in metadatas if m)
        print(f"📄 Archivos indexados ({len(sources)}):")
        for s in sources:
            print(f" - {s}")

    except Exception as e:
        print(f"❌ Error al acceder a la base de datos: {e}")
