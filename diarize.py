# -*- coding: utf-8 -*-
import json
from app.config import AUDIO_PATH, OUT_DIR, HF_TOKEN
DIARIZED_JSON = OUT_DIR / "diarized.json"

def diarize():
    if not HF_TOKEN:
        print("[DIAR] HF_TOKEN не задан — пропускаю диаризацию.")
        return None
    from pyannote.audio import Pipeline
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=HF_TOKEN)
    diarization = pipeline(str(AUDIO_PATH))
    out = [{"start": turn.start, "end": turn.end, "speaker": speaker}
           for turn, _, speaker in diarization.itertracks(yield_label=True)]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with DIARIZED_JSON.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return DIARIZED_JSON

if __name__ == "__main__":
    print(f"[DIAR] wrote {diarize()}")
