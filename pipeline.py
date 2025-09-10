# -*- coding: utf-8 -*-
import argparse
from app.mvp_asr import transcribe
from app.diarize import diarize
from app.build_corpus_index import build_index
from app.summarize_and_enrich import run as summarize_run
from app.export_utils import export_all

def main():
    p = argparse.ArgumentParser(description="Lecture Notetaker pipeline")
    p.add_argument("--asr", action="store_true", help="run ASR")
    p.add_argument("--diar", action="store_true", help="run diarization")
    p.add_argument("--index", action="store_true", help="build RAG index")
    p.add_argument("--sum", action="store_true", help="summarize + enrich")
    p.add_argument("--export", action="store_true", help="export MD/DOCX/HTML/TeX")
    p.add_argument("--all", action="store_true", help="run full pipeline")
    args = p.parse_args()

    if args.all or args.asr:
        transcribe()
    if args.all or args.diar:
        diarize()  # безопасно пропустится без HF_TOKEN
    if args.all or args.index:
        build_index()
    if args.all or args.sum:
        summarize_run()
    if args.all or args.export:
        export_all()

if __name__ == "__main__":
    main()
