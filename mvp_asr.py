# -*- coding: utf-8 -*-
import json, os
from faster_whisper import WhisperModel
from app.config import AUDIO_PATH, OUT_DIR, WHISPER_MODEL, USE_GPU, ASR_CHARS_PER_CHUNK
OUT_JSON = OUT_DIR / "transcript.json"

def transcribe():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = WhisperModel(
        WHISPER_MODEL,
        device="cuda" if USE_GPU else "cpu",
        compute_type="float16" if USE_GPU else "int8"
    )
    segments, _ = model.transcribe(
        str(AUDIO_PATH), vad_filter=True, word_timestamps=False, language="ru"
    )
    segs = [{"start": float(s.start), "end": float(s.end), "text": s.text.strip()} for s in segments]
    # простое чанкование
    chunks, buf, t0, t1 = [], "", None, None
    for s in segs:
        if t0 is None: t0 = s["start"]
        if len(buf) + len(s["text"]) > ASR_CHARS_PER_CHUNK:
            chunks.append({"start": t0, "end": t1 if t1 else s["end"], "text": buf.strip()})
            buf, t0 = "", s["start"]
        buf += (" " if buf else "") + s["text"]
        t1 = s["end"]
    if buf: chunks.append({"start": t0, "end": t1, "text": buf.strip()})

    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump({"segments": segs, "chunks": chunks}, f, ensure_ascii=False, indent=2)
    return OUT_JSON

if __name__ == "__main__":
    print(f"[ASR] writing to {transcribe()}")
