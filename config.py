import os
from pathlib import Path

# Путья
ROOT = Path(__file__).resolve().parents[1]
AUDIO_PATH = ROOT / "audio" / "lecture.wav"
OUT_DIR = ROOT / "out"
CORPUS_DIR = ROOT / "corpus_pdfs"
INDEX_DIR = OUT_DIR / "rag_index"
LATEX_TEMPLATE = ROOT / "latex" / "lecture_template.tex"

# ASR
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3")
USE_GPU = os.getenv("USE_GPU", "1") == "1"

# LLM провайдер: "openai" | "llamacpp"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

LLAMA_CPP_MODEL = os.getenv("LLAMA_CPP_MODEL", "models/llama-3.1-8b-instruct-q5.gguf")

# Диаризация (опционально)
HF_TOKEN = os.getenv("HF_TOKEN")  # HuggingFace token для pyannote

# RAG
EMB_MODEL = os.getenv("EMB_MODEL", "intfloat/multilingual-e5-base")

# Чанкование
ASR_CHARS_PER_CHUNK = 1600
ASR_CHARS_OVERLAP = 0

# Сколько сниппетов из RAG
RAG_K = 5
