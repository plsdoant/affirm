"""affirm CLI: `python -m affirm {download,transcribe,score}`."""

from __future__ import annotations

import argparse

from . import data


def main() -> None:
    ap = argparse.ArgumentParser(prog="affirm")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download", help="fetch corpora from the Hugging Face Hub into data/raw")
    d.add_argument("names", nargs="*", choices=list(data.CORPORA), help="default: all")

    t = sub.add_parser("transcribe", help="run a Whisper model over the eval splits")
    t.add_argument("model", help="alias (whisper-atc, whisper-large-v3, ...) or HF repo id")
    t.add_argument("--sets", nargs="+", default=list(data.CORPORA), choices=list(data.CORPORA))
    t.add_argument("--max-utts", type=int, default=None, help="deterministic per-set subset size")
    t.add_argument("--batch-size", type=int, default=8)

    s = sub.add_parser("score", help="score every hypothesis file under results/hyps")
    s.add_argument("--bootstrap", type=int, default=1000)

    args = ap.parse_args()
    if args.cmd == "download":
        data.download(args.names or None)
    elif args.cmd == "transcribe":
        from .transcribe import transcribe

        transcribe(args.model, args.sets, args.max_utts, args.batch_size)
    elif args.cmd == "score":
        from .score import main as score_main

        score_main(args.bootstrap)


if __name__ == "__main__":
    main()
