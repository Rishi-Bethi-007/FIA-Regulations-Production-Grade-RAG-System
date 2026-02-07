# scripts/build_index.py
from index.build_index import build_index_from_pdfs
from config import PDF_DIR

if __name__ == "__main__":
    build_index_from_pdfs(PDF_DIR)
