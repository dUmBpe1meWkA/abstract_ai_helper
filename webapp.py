# -*- coding: utf-8 -*-
import shutil, subprocess
from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from app.config import OUT_DIR
from app.pipeline import main as pipeline_main

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <html><body style="font-family:system-ui;max-width:720px;margin:2rem auto;color:#ddd;background:#111">
      <h2>Загрузка лекции</h2>
      <form action="/upload" enctype="multipart/form-data" method="post">
        <input type="file" name="file" accept="audio/*">
        <button type="submit">Обработать</button>
      </form>
      <p>После загрузки автоматически запустится: ASR → Summarize/RAG → Export.</p>
    </body></html>
    """

@app.post("/upload")
async def upload(file: UploadFile):
    from app.config import AUDIO_PATH
    AUDIO_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(AUDIO_PATH, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # прогоняем end-to-end: ASR → SUM → EXPORT
    subprocess.run(["python", "-m", "app.pipeline", "--asr"])
    subprocess.run(["python", "-m", "app.pipeline", "--sum"])
    subprocess.run(["python", "-m", "app.pipeline", "--export"])

    return JSONResponse({
        "message": "Готово",
        "download_md": "/download/lecture.md",
        "download_docx": "/download/lecture.docx",
        "download_html": "/download/lecture.html"
    })

@app.get("/download/{fname}")
def download(fname: str):
    path = OUT_DIR / fname
    return FileResponse(str(path))
