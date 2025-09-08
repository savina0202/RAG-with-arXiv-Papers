from pathlib import Path

BASE_DIR = Path(".").resolve()
DATA_DIR = BASE_DIR / "data"
PAPER_DIR = DATA_DIR / "papers"
PAPERLIST_FILENAME = DATA_DIR / "paperlist.json"
FAISS_INDEX_DIR = DATA_DIR / "faiss_index"
# FAISS_INDEX_FILE = DATA_DIR / "faiss_index" / "faiss.index"
# CHUNKS_FILE = DATA_DIR / "faiss_index" / "chunks.pkl"
# METADATA_FILE = DATA_DIR / "faiss_index" / "metadata.pkl"

PROJECT_DIR = Path(__file__).resolve().parent.parent
PAPERLIST_FILENAME = PROJECT_DIR / "core" / "data" / "paperlist.json"
FAISS_INDEX_FILE = PROJECT_DIR / "core" / "data" / "faiss_index" / "faiss.index"
CHUNKS_FILE = PROJECT_DIR / "core" / "data" / "faiss_index" / "chunks.pkl"
METADATA_FILE = PROJECT_DIR / "core" / "data" / "faiss_index" / "metadata.pkl"
