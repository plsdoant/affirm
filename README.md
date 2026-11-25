# affirm

ATC speech → verified commands. Transcribe air traffic control radio, extract the
instructions, and check what controllers said against what planes actually did.

## Pipeline

**Voice radio**

1. **Radio audio** — RTL-SDR dongle on the airband (118–137 MHz)
2. **Segment** — split transmissions by squelch, tag speaker role (controller / pilot)
3. **Transcribe** — Whisper fine-tuned on ATC speech
4. **Context** — nearby callsigns from ADS-B fed to the model as a prompt
5. **Extract** — callsign, command, value → JSON

**Aircraft data (ADS-B)**

- RTL-SDR dongle on 1090 MHz
- Decode + store with `readsb` (live) or OpenSky (history)
- Aircraft tracks: callsign, altitude, autopilot targets

**Verify**

6. **Verify** — match the callsign to a track: did the plane do what it was told?
   Confirmed transcripts feed back into training.

**Training**

- Free corpora: ATCOSIM, UWB-ATCC, ATCO2 sample
- Fine-tune Whisper with LoRA + radio-noise augmentation

**Outputs**

- Live map demo with transcripts shown next to each plane
- Readback check (stretch goal): flag wrong pilot readbacks

## Build order

1. **Baselines + eval** — text normalizer, scoring
2. **Fine-tune Whisper** (step 3) — LoRA + augmentation
3. **Receivers + own data** (steps 1–2) — record a few weeks, hand-label a test set
4. **Context + extraction** (steps 4–5) — callsign prompts, command JSON
5. **Verify + self-train** (step 6) — flight-data check, demo, write-up
6. **Readback check** (stretch)

## Baselines + eval

Results: [`results/baselines.md`](results/baselines.md).

```sh
source .venv/bin/activate
uv pip install -e . pytest

python -m affirm download                       # ~3 GB of parquet into data/raw/ (gitignored)
python -m affirm transcribe whisper-atc         # or whisper-large-v3, whisper-large-v3-turbo, any HF repo id
python -m affirm score                          # -> results/baselines.md, results/summary.json
pytest
```

`transcribe` resumes where it left off; `--max-utts N` scores a fixed, model-independent subset
for quick checks.

### Eval sets

| corpus | source split | utts | audio | what it is |
|---|---|---|---|---|
| `atco2` | `jlvdoorn/atco2-asr` validation | 113 | 0.23 h | real VHF radio, Prague/Brno, controller + pilot |
| `atcosim` | `jlvdoorn/atcosim` validation | 1,913 | 2.09 h | simulated controller speech, clean mic |
| `uwb_atcc` | `Jzuluaga/uwb_atcc` test | 2,822 | 2.43 h | real Czech ATC radio, noisy, both sides |

The bar to beat is [`jlvdoorn/whisper-large-v3-atco2-asr-atcosim`](https://huggingface.co/jlvdoorn/whisper-large-v3-atco2-asr-atcosim)
(TU Delft's Whisper-ATC, large-v3 fine-tuned on the ATCO2 and ATCOSIM *train* splits above). The
ATCO2 and ATCOSIM evals use jlvdoorn's own validation splits so that model is scored on audio it
never trained on; the `Jzuluaga/*` ATCO2/ATCOSIM test splits are a different random cut and overlap
its training data. It never saw UWB-ATCC.

**Our fine-tuning must train only on the `train` splits** (`affirm.data.CORPORA`), or the
comparison is void.

### Normalizer

`affirm.normalize.normalize()` maps references and hypotheses to one spoken form before WER:

| written | normalized |
|---|---|
| `FL350`, `flight level 350` | `flight level three five zero` |
| `4,500 ft`, `four thousand five hundred feet` | `four five zero zero feet` |
| `118.1`, `one one eight point one` | `one one eight decimal one` |
| `niner`, `tree`, `fife` | `nine`, `three`, `five` |
| `Runway 27L` | `runway two seven left` |
| `Lufthansa 8MM`, `OK-HHH`, `triple hotel` | `lufthansa eight mike mike`, `oscar kilo hotel hotel hotel` |
| `Alpha`, `Juliet`, `X-ray`, `K` | `alfa`, `juliett`, `xray`, `kilo` |
| `KLM`, `k l m` | `kilo lima mike` |
| `Speed Bird`, `Sky Travel`, `stand by` | `speedbird`, `skytravel`, `standby` |
| `ah`, `ehm`, `heading(-ding)`, `re-` | dropped / `heading` / dropped |

Numbers are always spelled digit by digit, so a grouped reading and a digit-wise reading of the
same value match. The test cases are in `tests/test_normalize.py`; add one whenever a new mismatch
shows up in model output. Caveats: UWB-ATCC segments sometimes cut words in half (`fiv`, `imme`),
and ATCO2 keeps non-English greetings (`dobrý den`); both count as errors for every model alike.
