"""Run a Whisper checkpoint over the eval splits and write hypotheses to JSONL.

Every model is decoded the same way (greedy, English, transcribe task, no prompt), so differences
come from the weights, not the decoding setup. Output is resumable: rerunning skips finished ids.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import torch

from . import data

RESULTS_DIR = data.REPO_ROOT / "results"
HYPS_DIR = RESULTS_DIR / "hyps"

# Baselines. The fine-tuned "bar to beat" first.
MODELS = {
    "whisper-atc": "jlvdoorn/whisper-large-v3-atco2-asr-atcosim",
    "whisper-large-v3": "openai/whisper-large-v3",
    "whisper-large-v3-turbo": "openai/whisper-large-v3-turbo",
}


def model_slug(model: str) -> str:
    for k, v in MODELS.items():
        if model in (k, v):
            return k
    return model.replace("/", "__")


def subset_ids(name: str, max_utts: int | None) -> set[str] | None:
    """Deterministic, model-independent subset: the `max_utts` ids with the smallest sha1."""
    if not max_utts:
        return None
    ids = [u.id for u in data.iter_utterances(name, data.eval_split(name), with_audio=False)]
    if max_utts >= len(ids):
        return None
    return set(sorted(ids, key=lambda i: hashlib.sha1(i.encode()).hexdigest())[:max_utts])


def pick_device() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda", torch.float16
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def transcribe(model: str, sets: list[str], max_utts: int | None = None, batch_size: int = 8) -> None:
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    repo = MODELS.get(model, model)
    slug = model_slug(model)
    device, dtype = pick_device()
    print(f"loading {repo} on {device} ({dtype})")
    processor = WhisperProcessor.from_pretrained(repo)
    net = WhisperForConditionalGeneration.from_pretrained(repo, dtype=dtype).to(device).eval()
    # Older fine-tunes ship forced_decoder_ids in their generation config; language/task below
    # replaces them, so drop the stale field to keep decoding identical across models.
    net.generation_config.forced_decoder_ids = None
    net.config.forced_decoder_ids = None

    out_dir = HYPS_DIR / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "MODEL.json").write_text(
        json.dumps({"repo": repo, "decoding": "greedy, language=en, task=transcribe"}, indent=2)
    )

    for name in sets:
        split = data.eval_split(name)
        keep = subset_ids(name, max_utts)
        path = out_dir / f"{name}.jsonl"
        done = set()
        if path.exists():
            raw = path.read_text()
            if raw and not raw.endswith("\n"):  # killed mid-write: drop the partial last line
                raw = raw[: raw.rfind("\n") + 1]
                path.write_text(raw)
            done = {json.loads(l)["id"] for l in raw.splitlines() if l.strip()}
        all_ids = keep or {u.id for u in data.iter_utterances(name, split, with_audio=False)}
        todo = data.iter_utterances(name, split, ids=all_ids - done)
        total = len(all_ids)

        t0, n, audio_s = time.time(), len(done), 0.0
        with path.open("a") as fh:
            batch: list[data.Utterance] = []

            def flush():
                nonlocal n, audio_s
                feats = processor.feature_extractor(
                    [u.audio for u in batch], sampling_rate=data.SAMPLE_RATE, return_tensors="pt"
                ).input_features.to(device, dtype)
                with torch.inference_mode():
                    ids = net.generate(feats, language="en", task="transcribe", num_beams=1, max_length=448)
                texts = processor.batch_decode(ids, skip_special_tokens=True)
                for u, hyp in zip(batch, texts):
                    fh.write(json.dumps({"id": u.id, "ref": u.text, "hyp": hyp.strip(), "duration": round(u.duration, 3)}) + "\n")
                fh.flush()
                n += len(batch)
                audio_s += sum(u.duration for u in batch)
                el = time.time() - t0
                print(f"  [{slug}/{name}] {n}/{total}  RTF {el / max(audio_s, 1e-9):.3f}", flush=True)
                batch.clear()

            for u in todo:
                batch.append(u)
                if len(batch) == batch_size:
                    flush()
            if batch:
                flush()


def compression_ratio(text: str) -> float:
    """gzip-style ratio OpenAI's Whisper uses to detect repetition loops (> 2.4 = looping)."""
    import zlib

    b = text.encode()
    return len(b) / max(1, len(zlib.compress(b)))


FALLBACK_TEMPS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def repair(model: str, sets: list[str], threshold: float = 2.4, batch_size: int = 4) -> None:
    """Greedy + Whisper's temperature fallback, applied only where greedy output loops.

    Copies the greedy hypotheses to `<slug>+fallback/` and re-decodes the utterances whose greedy
    output has compression ratio > `threshold` (or is empty) with temperature fallback
    (compression-ratio and avg-logprob checks, as in `openai-whisper`'s transcribe). Seeded, so
    reruns give the same output.
    """
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    repo = MODELS.get(model, model)
    slug = model_slug(model)
    src_dir, out_dir = HYPS_DIR / slug, HYPS_DIR / f"{slug}+fallback"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "MODEL.json").write_text(
        json.dumps(
            {
                "repo": repo,
                "decoding": f"greedy; where compression ratio > {threshold} or empty: temperature fallback "
                f"{list(FALLBACK_TEMPS)}, compression_ratio_threshold=1.35 (tokens), logprob_threshold=-1.0, seed 0",
            },
            indent=2,
        )
    )
    device, dtype = pick_device()
    processor = net = None

    for name in sets:
        src = src_dir / f"{name}.jsonl"
        if not src.exists():
            continue
        rows = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
        bad = {r["id"] for r in rows if not r["hyp"].strip() or compression_ratio(r["hyp"]) > threshold}
        fixed: dict[str, str] = {}
        if bad:
            if net is None:
                print(f"loading {repo} on {device} ({dtype})")
                processor = WhisperProcessor.from_pretrained(repo)
                net = WhisperForConditionalGeneration.from_pretrained(repo, dtype=dtype).to(device).eval()
                net.generation_config.forced_decoder_ids = None
                net.config.forced_decoder_ids = None
            utts = list(data.iter_utterances(name, data.eval_split(name), ids=bad))
            for k in range(0, len(utts), batch_size):
                batch = utts[k : k + batch_size]
                feats = processor.feature_extractor(
                    [u.audio for u in batch], sampling_rate=data.SAMPLE_RATE, return_tensors="pt"
                ).input_features.to(device, dtype)
                torch.manual_seed(k)
                with torch.inference_mode():
                    ids = net.generate(
                        feats, language="en", task="transcribe", max_length=448, temperature=FALLBACK_TEMPS,
                        compression_ratio_threshold=1.35, logprob_threshold=-1.0, do_sample=False,
                    )
                for u, hyp in zip(batch, processor.batch_decode(ids, skip_special_tokens=True)):
                    fixed[u.id] = hyp.strip()
        with (out_dir / f"{name}.jsonl").open("w") as fh:
            for r in rows:
                r = dict(r, hyp=fixed.get(r["id"], r["hyp"]), repaired=r["id"] in fixed)
                fh.write(json.dumps(r) + "\n")
        print(f"  [{slug}+fallback/{name}] re-decoded {len(fixed)}/{len(rows)}", flush=True)
