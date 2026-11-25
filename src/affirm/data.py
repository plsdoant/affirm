"""Dataset registry, download, and loading.

Every corpus is pulled from the Hugging Face Hub as parquet, pinned to a commit,
and decoded with soundfile (no `datasets` dependency, no remote loader scripts).

Split policy — the baseline we have to beat is `jlvdoorn/whisper-large-v3-atco2-asr-atcosim`,
trained on the *train* splits of `jlvdoorn/atco2-asr` and `jlvdoorn/atcosim`. So:

* ATCO2 and ATCOSIM are evaluated on jlvdoorn's *validation* splits (held out from the baseline).
  Do NOT use `Jzuluaga/atcosim_corpus` or `Jzuluaga/atco2_corpus_1h` for eval: their test
  splits are a different random cut and overlap the baseline's training data.
* UWB-ATCC was never seen by the baseline, so the `Jzuluaga/uwb_atcc` test split is clean.

Our own fine-tuning must train only on the `train` splits below and never touch the eval splits.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import soxr

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class Corpus:
    name: str
    repo: str
    revision: str
    license: str
    description: str
    # split name in the HF repo -> our role for it
    splits: dict[str, str]


CORPORA: dict[str, Corpus] = {
    "atco2": Corpus(
        name="atco2",
        repo="jlvdoorn/atco2-asr",
        revision="3b96f57915",
        license="ATCO2 1h test-set licence (free, see atco2.org)",
        description="ATCO2 free 1h sample: real VHF radio, Prague + Brno, controller and pilot, multi-speaker segments",
        splits={"train": "train", "validation": "eval"},
    ),
    "atcosim": Corpus(
        name="atcosim",
        repo="jlvdoorn/atcosim",
        revision="b5839d9404",
        license="ATCOSIM licence (free for research)",
        description="ATCOSIM: ~10.7h simulated controller speech, clean close-talk mic, non-native English",
        splits={"train": "train", "validation": "eval"},
    ),
    "uwb_atcc": Corpus(
        name="uwb_atcc",
        repo="Jzuluaga/uwb_atcc",
        revision="1e8270beb8",
        license="CC BY-NC-SA 4.0",
        description="UWB-ATCC: ~20h real Czech ATC radio, controller and pilot, noisy",
        splits={"train": "train", "test": "eval"},
    ),
}


def corpus_dir(c: Corpus) -> Path:
    return RAW_DIR / c.name


def download(names: list[str] | None = None) -> None:
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi()
    for name in names or list(CORPORA):
        c = CORPORA[name]
        sha = api.dataset_info(c.repo, revision=c.revision).sha
        out = corpus_dir(c)
        out.mkdir(parents=True, exist_ok=True)
        print(f"[{name}] {c.repo}@{sha[:10]} -> {out}")
        snapshot_download(
            repo_id=c.repo,
            repo_type="dataset",
            revision=sha,
            local_dir=out,
            allow_patterns=["data/*.parquet", "README.md"],  # never the repo's loader scripts
        )
        (out / "SOURCE.json").write_text(
            json.dumps({"repo": c.repo, "revision": sha, "license": c.license}, indent=2)
        )


def split_files(name: str, split: str) -> list[Path]:
    c = CORPORA[name]
    files = sorted((corpus_dir(c) / "data").glob(f"{split}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"{name}/{split} not downloaded; run `python -m affirm download {name}`")
    return files


def eval_split(name: str) -> str:
    return next(s for s, role in CORPORA[name].splits.items() if role == "eval")


@dataclass
class Utterance:
    id: str
    text: str
    audio: np.ndarray  # float32 mono @ 16 kHz
    duration: float


def _decode(audio_cell: dict) -> np.ndarray:
    wav, sr = sf.read(io.BytesIO(audio_cell["bytes"]), dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if sr != SAMPLE_RATE:
        wav = soxr.resample(wav, sr, SAMPLE_RATE)
    return wav.astype(np.float32)


def iter_utterances(
    name: str, split: str, with_audio: bool = True, ids: set[str] | None = None
) -> Iterator[Utterance]:
    """Yield utterances in a stable order. IDs are `<corpus>/<split>/<row>` when the source has none.
    `ids` restricts to those utterances (audio is only decoded for them)."""
    row = 0
    for f in split_files(name, split):
        pf = pq.ParquetFile(f)
        cols = [c for c in pf.schema_arrow.names if c in ("id", "text", "audio")]
        if not with_audio:
            cols = [c for c in cols if c != "audio"]
        for batch in pf.iter_batches(batch_size=64, columns=cols):
            for rec in batch.to_pylist():
                uid = rec.get("id") or f"{name}/{split}/{row:05d}"
                row += 1
                if ids is not None and uid not in ids:
                    continue
                audio = _decode(rec["audio"]) if with_audio else np.zeros(0, np.float32)
                yield Utterance(uid, rec["text"], audio, len(audio) / SAMPLE_RATE)
