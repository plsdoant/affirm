# Baselines

WER in %, after `affirm.normalize` on both sides. Lower is better. Greedy decoding, `language=en`, no prompt. 95% CI from 1,000 utterance-level bootstrap resamples.

- **atco2** — `jlvdoorn/atco2-asr` `validation`: 113 utts, 2283 words
- **atcosim** — `jlvdoorn/atcosim` `validation`: 1913 utts, 21447 words
- **uwb_atcc** — `Jzuluaga/uwb_atcc` `test`: 2822 utts, 28033 words

## WER (ATC-normalized)

| model | atco2 | atcosim | uwb_atcc |
|---|---|---|---|
| whisper-atc | **47.7** (38.0–58.3) | **15.1** (13.7–16.5) | **55.9** (53.5–58.4) |
| whisper-atc+fallback | **42.2** (35.1–49.8) | **13.5** (12.4–14.6) | **49.2** (47.8–50.6) |
| whisper-large-v3 | **31.2** (27.1–36.1) | **17.7** (16.9–18.5) | **68.7** (63.5–74.6) |
| whisper-large-v3+fallback | **31.2** (27.1–36.1) | **17.7** (16.9–18.5) | **59.0** (57.5–60.6) |
| whisper-large-v3-turbo | **41.5** (29.3–65.4) | **19.4** (18.7–20.1) | **97.4** (84.7–111.4) |
| whisper-large-v3-turbo+fallback | **33.0** (29.0–37.3) | **19.4** (18.7–20.1) | **63.1** (61.6–64.7) |

## Digit error rate

Share of spoken digits in the reference that the model got wrong or dropped.

| model | atco2 | atcosim | uwb_atcc |
|---|---|---|---|
| whisper-atc | 38.6 | 8.8 | 29.3 |
| whisper-atc+fallback | 34.8 | 8.1 | 28.9 |
| whisper-large-v3 | 7.8 | 1.7 | 27.5 |
| whisper-large-v3+fallback | 7.8 | 1.7 | 27.5 |
| whisper-large-v3-turbo | 9.7 | 1.9 | 28.9 |
| whisper-large-v3-turbo+fallback | 9.5 | 1.9 | 28.9 |

## Why the normalizer matters

WER with only lowercasing and punctuation removal, vs. ATC-normalized.

| model | atco2 naive → ATC | atcosim naive → ATC | uwb_atcc naive → ATC |
|---|---|---|---|
| whisper-atc | 49.1 → 47.7 | 15.3 → 15.1 | 56.9 → 55.9 |
| whisper-atc+fallback | 43.4 → 42.2 | 13.7 → 13.5 | 49.9 → 49.2 |
| whisper-large-v3 | 57.3 → 31.2 | 66.9 → 17.7 | 92.6 → 68.7 |
| whisper-large-v3+fallback | 57.3 → 31.2 | 66.9 → 17.7 | 83.2 → 59.0 |
| whisper-large-v3-turbo | 67.7 → 41.5 | 66.2 → 19.4 | 118.4 → 97.4 |
| whisper-large-v3-turbo+fallback | 59.0 → 33.0 | 66.2 → 19.4 | 84.9 → 63.1 |

## Error breakdown (ATC-normalized, % of reference words)

| model | corpus | sub | del | ins |
|---|---|---|---|---|
| whisper-atc | atco2 | 15.5 | 23.5 | 8.7 |
| whisper-atc | atcosim | 5.5 | 5.7 | 3.9 |
| whisper-atc | uwb_atcc | 18.1 | 23.4 | 14.4 |
| whisper-atc+fallback | atco2 | 10.0 | 26.7 | 5.5 |
| whisper-atc+fallback | atcosim | 5.1 | 5.4 | 3.0 |
| whisper-atc+fallback | uwb_atcc | 17.4 | 23.7 | 8.1 |
| whisper-large-v3 | atco2 | 17.1 | 5.3 | 8.8 |
| whisper-large-v3 | atcosim | 11.7 | 2.2 | 3.8 |
| whisper-large-v3 | uwb_atcc | 31.2 | 12.9 | 24.6 |
| whisper-large-v3+fallback | atco2 | 17.1 | 5.3 | 8.8 |
| whisper-large-v3+fallback | atcosim | 11.7 | 2.2 | 3.8 |
| whisper-large-v3+fallback | uwb_atcc | 31.0 | 13.0 | 14.9 |
| whisper-large-v3-turbo | atco2 | 20.0 | 5.9 | 15.5 |
| whisper-large-v3-turbo | atcosim | 12.9 | 2.9 | 3.6 |
| whisper-large-v3-turbo | uwb_atcc | 36.6 | 10.5 | 50.2 |
| whisper-large-v3-turbo+fallback | atco2 | 19.9 | 5.8 | 7.3 |
| whisper-large-v3-turbo+fallback | atcosim | 12.9 | 2.9 | 3.6 |
| whisper-large-v3-turbo+fallback | uwb_atcc | 36.4 | 10.7 | 16.1 |
