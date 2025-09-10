# -*- coding: utf-8 -*-
import json
from pathlib import Path
from typing import List
import faiss, numpy as np
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz
from app.config import OUT_DIR, INDEX_DIR, EMB_MODEL, LLM_PROVIDER, OPENAI_MODEL, OPENAI_API_KEY, LLAMA_CPP_MODEL
from app.prompts import SEGMENT_PROMPT, AGGREGATE_PROMPT, TOPICS_PROMPT, ENRICH_PROMPT

def mmss(t: float) -> str:
    mm, ss = int(t // 60), int(round(t % 60))
    return f"{mm:02d}:{ss:02d}"

def load_chunks():
    with (OUT_DIR / "transcript.json").open("r", encoding="utf-8") as f:
        return json.load(f)["chunks"]

# = LLM providers =
class OpenAIProv:
    def __init__(self):
        from openai import OpenAI
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is required for openai provider")
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = OPENAI_MODEL
    def chat(self, prompt: str) -> str:
        r = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role":"user","content":prompt}],
            temperature=0.2
        )
        return r.choices[0].message.content.strip()

class LlamaCppProv:
    def __init__(self):
        from llama_cpp import Llama
        self.llm = Llama(model_path=LLAMA_CPP_MODEL, n_ctx=8192, n_gpu_layers=35, verbose=False)
    def chat(self, prompt: str) -> str:
        out = self.llm(prompt, temperature=0.2, max_tokens=2048, stop=["</s>"])
        return out["choices"][0]["text"].strip()

def get_llm():
    return LlamaCppProv() if LLM_PROVIDER == "llamacpp" else OpenAIProv()

# = RAG =
def load_rag():
    ip = INDEX_DIR / "faiss.index"
    if not ip.exists():
        return None
    index = faiss.read_index(str(ip))
    texts = json.loads((INDEX_DIR / "texts.json").read_text(encoding="utf-8"))
    meta  = json.loads((INDEX_DIR / "meta.json").read_text(encoding="utf-8"))
    emb_model = SentenceTransformer(EMB_MODEL)
    return index, texts, meta, emb_model

def search_snippets(queries: List[str], k=5):
    rag = load_rag()
    if not rag or not queries:
        return []
    index, texts, meta, emb_model = rag
    out = []
    for q in queries[:5]:
        q_emb = emb_model.encode([f"query: {q}"], normalize_embeddings=True).astype("float32")
        D, I = index.search(q_emb, k)
        for score, idx in zip(D[0], I[0]):
            snippet = texts[idx][:800]
            src = Path(meta[idx]["source"]).name
            out.append(f"[{src}] {snippet}")
    # dedupe
    uniq = []
    for s in out:
        if all(fuzz.token_set_ratio(s, u) < 90 for u in uniq):
            uniq.append(s)
    return uniq[:10]

def run():
    llm = get_llm()
    chunks = load_chunks()

    # 1) мини-конспекты
    minis = []
    for ch in chunks:
        p = SEGMENT_PROMPT.format(text=ch["text"], t0=mmss(ch["start"]), t1=mmss(ch["end"]))
        minis.append(llm.chat(p))
    joined = "\n\n".join(minis)

    # 2) агрегирование
    master = llm.chat(AGGREGATE_PROMPT.format(mini_md=joined))

    # 3) темы → RAG → доп. материалы
    topics_raw = llm.chat(TOPICS_PROMPT.format(md=master))
    try:
        topics = json.loads(topics_raw)
        topics = [t.strip() for t in topics if isinstance(t, str) and t.strip()]
    except Exception:
        topics = []
    snippets = search_snippets(topics, k=5)
    if snippets:
        enrich = llm.chat(ENRICH_PROMPT.format(topics=", ".join(topics[:5]), snippets="\n\n".join(snippets)))
        full_md = master.strip() + "\n\n" + enrich.strip()
    else:
        full_md = master

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "lecture.md").write_text(full_md, encoding="utf-8")
    return OUT_DIR / "lecture.md"

if __name__ == "__main__":
    print(f"[SUM] wrote {run()}")
