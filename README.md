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

## Build order (~14 weeks part-time)

| Weeks | Milestone |
|-------|-----------|
| 1–2   | Baselines + eval: text normalizer, scoring |
| 3–5   | Fine-tune Whisper (step 3): LoRA + augmentation |
| 4–8   | Receivers + own data (steps 1–2): record a few weeks, hand-label a test set |
| 8–11  | Context + extraction (steps 4–5): callsign prompts, command JSON |
| 11–14 | Verify + self-train (step 6): flight-data check, demo, write-up |
| 14+   | Readback check (stretch) |
