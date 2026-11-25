"""Score hypothesis files: WER under the ATC normalizer (with bootstrap CI), WER under a naive
normalizer for comparison, and the error rate on spoken digits (the values that matter for
verification: levels, headings, frequencies, callsign numbers).

Writes results/summary.json and results/baselines.md.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import jiwer
import numpy as np

from . import data
from .normalize import DIGIT_WORDS, basic_normalize, normalize
from .transcribe import HYPS_DIR, MODELS, RESULTS_DIR

DIGITS = set(DIGIT_WORDS)


@dataclass
class Score:
    model: str
    corpus: str
    utts: int
    ref_words: int
    wer: float
    wer_ci: tuple[float, float]
    sub: float
    dele: float
    ins: float
    wer_basic: float
    digit_er: float
    digit_words: int


def _per_utt_errors(out: jiwer.WordOutput) -> np.ndarray:
    errs = []
    for chunks in out.alignments:
        e = 0
        for c in chunks:
            if c.type == "substitute" or c.type == "delete":
                e += c.ref_end_idx - c.ref_start_idx
            elif c.type == "insert":
                e += c.hyp_end_idx - c.hyp_start_idx
        errs.append(e)
    return np.array(errs)


def _digit_errors(out: jiwer.WordOutput) -> tuple[int, int]:
    """Reference digit words that were substituted or deleted / all reference digit words."""
    wrong = total = 0
    for ref, chunks in zip(out.references, out.alignments):
        total += sum(w in DIGITS for w in ref)
        for c in chunks:
            if c.type in ("substitute", "delete"):
                wrong += sum(w in DIGITS for w in ref[c.ref_start_idx : c.ref_end_idx])
    return wrong, total


def _wer(refs: list[str], hyps: list[str]) -> jiwer.WordOutput:
    # jiwer rejects empty references; an empty hypothesis is fine
    return jiwer.process_words(refs, [h if h else "" for h in hyps])


def score_file(model: str, corpus: str, rows: list[dict], n_boot: int, seed: int = 0) -> Score:
    rows = [r for r in rows if normalize(r["ref"])]
    refs = [normalize(r["ref"]) for r in rows]
    hyps = [normalize(r["hyp"]) for r in rows]
    out = _wer(refs, hyps)
    n_ref = np.array([len(r.split()) for r in refs])
    errs = _per_utt_errors(out)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(rows), size=(n_boot, len(rows)))
    boot = errs[idx].sum(1) / n_ref[idx].sum(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    total = n_ref.sum()

    basic_rows = [r for r in rows if basic_normalize(r["ref"])]
    basic = jiwer.wer([basic_normalize(r["ref"]) for r in basic_rows], [basic_normalize(r["hyp"]) for r in basic_rows])
    dw, dt = _digit_errors(out)
    return Score(
        model=model,
        corpus=corpus,
        utts=len(rows),
        ref_words=int(total),
        wer=out.wer,
        wer_ci=(float(lo), float(hi)),
        sub=out.substitutions / total,
        dele=out.deletions / total,
        ins=out.insertions / total,
        wer_basic=basic,
        digit_er=dw / dt if dt else float("nan"),
        digit_words=dt,
    )


def load_rows(path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def main(n_boot: int = 1000) -> None:
    order = list(MODELS)
    model_dirs = sorted(
        (p for p in HYPS_DIR.iterdir() if p.is_dir()),
        key=lambda p: (order.index(p.name) if p.name in order else len(order), p.name),
    )
    scores: list[Score] = []
    for mdir in model_dirs:
        for corpus in data.CORPORA:
            f = mdir / f"{corpus}.jsonl"
            if f.exists():
                scores.append(score_file(mdir.name, corpus, load_rows(f), n_boot))

    (RESULTS_DIR / "summary.json").write_text(json.dumps([asdict(s) for s in scores], indent=2))
    md = render(scores)
    (RESULTS_DIR / "baselines.md").write_text(md)
    print(md)


def _pct(x: float) -> str:
    return f"{100 * x:.1f}"


def render(scores: list[Score]) -> str:
    corpora = [c for c in data.CORPORA if any(s.corpus == c for s in scores)]
    models = list(dict.fromkeys(s.model for s in scores))
    by = {(s.model, s.corpus): s for s in scores}

    lines = ["# Baselines", ""]
    lines += [
        "WER in %, after `affirm.normalize` on both sides. Lower is better. "
        "Greedy decoding, `language=en`, no prompt. 95% CI from 1,000 utterance-level bootstrap resamples.",
        "",
    ]
    for c in corpora:
        s0 = next(s for s in scores if s.corpus == c)
        split = data.eval_split(c)
        lines.append(f"- **{c}** — `{data.CORPORA[c].repo}` `{split}`: {s0.utts} utts, {s0.ref_words} words")
    lines += ["", "## WER (ATC-normalized)", ""]
    lines.append("| model | " + " | ".join(corpora) + " |")
    lines.append("|---|" + "---|" * len(corpora))
    for m in models:
        cells = []
        for c in corpora:
            s = by.get((m, c))
            cells.append(f"**{_pct(s.wer)}** ({_pct(s.wer_ci[0])}–{_pct(s.wer_ci[1])})" if s else "—")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")

    lines += ["", "## Digit error rate", "", "Share of spoken digits in the reference that the model got wrong or dropped.", ""]
    lines.append("| model | " + " | ".join(corpora) + " |")
    lines.append("|---|" + "---|" * len(corpora))
    for m in models:
        lines.append(f"| {m} | " + " | ".join(_pct(by[(m, c)].digit_er) if (m, c) in by else "—" for c in corpora) + " |")

    lines += ["", "## Why the normalizer matters", "", "WER with only lowercasing and punctuation removal, vs. ATC-normalized.", ""]
    lines.append("| model | " + " | ".join(f"{c} naive → ATC" for c in corpora) + " |")
    lines.append("|---|" + "---|" * len(corpora))
    for m in models:
        cells = [f"{_pct(by[(m, c)].wer_basic)} → {_pct(by[(m, c)].wer)}" if (m, c) in by else "—" for c in corpora]
        lines.append(f"| {m} | " + " | ".join(cells) + " |")

    lines += ["", "## Error breakdown (ATC-normalized, % of reference words)", ""]
    lines.append("| model | corpus | sub | del | ins |")
    lines.append("|---|---|---|---|---|")
    for s in scores:
        lines.append(f"| {s.model} | {s.corpus} | {_pct(s.sub)} | {_pct(s.dele)} | {_pct(s.ins)} |")
    lines.append("")
    return "\n".join(lines)
