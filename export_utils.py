# -*- coding: utf-8 -*-
import subprocess
from app.config import OUT_DIR, LATEX_TEMPLATE

def export_all():
    md = OUT_DIR / "lecture.md"
    assert md.exists(), "out/lecture.md not found"

    # DOCX
    subprocess.run(["pandoc", str(md), "-o", str(OUT_DIR / "lecture.docx")])
    # HTML
    subprocess.run(["pandoc", str(md), "-o", str(OUT_DIR / "lecture.html"), "--toc"])

    # LaTeX (через шаблон быстро; для продакшена лучше pandoc → tex/pdf)
    if LATEX_TEMPLATE.exists():
        txt = LATEX_TEMPLATE.read_text(encoding="utf-8")
        safe = md.read_text(encoding="utf-8").replace("\\", "\\textbackslash{}")
        (OUT_DIR / "lecture.tex").write_text(txt.replace("%__CONTENT__", safe), encoding="utf-8")

    print("[EXP] exported: MD, DOCX, HTML (+TeX)")
