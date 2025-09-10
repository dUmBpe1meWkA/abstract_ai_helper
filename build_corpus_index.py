# -*- coding: utf-8 -*-
import json, re
from pathlib import Path
import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from app.config import CORPUS_DIR, INDEX_DIR, EMB_MODEL

CHUNK_SIZE, CHUNK_OVERLAP = 1200, 150

def pdf_to_text(path: Path) -> str:
    try:
        reader = PdfReader(str(path))
        return "\n".join([p.extract_text() or "" for p in reader.pages])
    except Exception as e:
        print(f"[RAG] read fail {path}: {e}")
        return ""

def split_text(t: str):
    t = re.sub(r"\s+", " ", t.strip())
    i, out = 0, []
    while i < len(t):
        j = min(len(t), i + CHUNK_SIZE)
        out.append(t[i:j])
        i = j - CHUNK_OVERLAP
        if i < 0: i = 0
    return [c for c in out if c.strip()]

def build_index():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    texts, meta = [], []
    for p in Path(CORPUS_DIR).glob("**/*.pdf"):
        raw = pdf_to_text(p)
        for chunk in split_text(raw):
            texts.append(chunk)
            meta.append({"source": str(p)})
    if not texts:
        print("[RAG] no texts in corpus_pdfs/")
        return None

    model = SentenceTransformer(EMB_MODEL)
    embs = model.encode([f"passage: {t}" for t in texts], normalize_embeddings=True, batch_size=32, show_progress_bar=True)
    embs = embs.astype("float32")

    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)

    faiss.write_index(index, str(INDEX_DIR / "faiss.index"))
    (INDEX_DIR / "texts.json").write_text(json.dumps(texts, ensure_ascii=False), encoding="utf-8")
    (INDEX_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[RAG] built index, chunks={len(texts)}")
    return INDEX_DIR

if __name__ == "__main__":
    build_index()
