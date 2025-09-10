# Notetaker: Лекция → Конспект (ASR + LLM + RAG)

## Быстрый старт
```bash
python -m venv .venv && . .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# положи audio/lecture.wav и PDF в corpus_pdfs/
# (опционально) export OPENAI_API_KEY=sk-...
# (опционально) export HF_TOKEN=hf_...
python -m app.build_corpus_index     # один раз, если есть PDF
python -m app.pipeline --all         # полный прогон
