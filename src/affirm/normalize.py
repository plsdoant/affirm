"""ATC transcript normalizer.

Reference transcripts and model outputs write the same speech in different ways:

    "FL350"            "flight level three five zero"
    "niner"            "nine"
    "4,500 ft"         "four thousand five hundred feet"
    "118.1"            "one one eight decimal one" / "one eighteen point one"
    "Runway 27L"       "runway two seven left"
    "OK-HHH"           "oscar kilo triple hotel"
    "Alpha" / "Juliet" "alfa" / "juliett"
    "Sky Travel"       "Skytravel"

Without collapsing these, WER mostly measures formatting. `normalize()` maps any of the above
onto one canonical *spoken* form:

* lowercase ASCII words, no punctuation;
* every number is spelled digit by digit ("350" -> "three five zero", "four thousand five
  hundred" -> "four five zero zero"), so digit-wise and grouped readings of the same value
  match;
* letters are ICAO alphabet words ("K" -> "kilo", "alpha" -> "alfa");
* hesitations and annotation markup are removed.

It is applied identically to references and hypotheses. The rules are deliberately
conservative: when a rewrite could change meaning (e.g. "a", "i", "oh") it is left alone.
"""

from __future__ import annotations

import re
import unicodedata

# --------------------------------------------------------------------------- vocab

DIGIT_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
UNITS = {w: i for i, w in enumerate(DIGIT_WORDS)}
TEENS = {
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
SCALES = {"hundred": 100, "thousand": 1000}

ICAO = {
    "a": "alfa", "b": "bravo", "c": "charlie", "d": "delta", "e": "echo", "f": "foxtrot",
    "g": "golf", "h": "hotel", "i": "india", "j": "juliett", "k": "kilo", "l": "lima",
    "m": "mike", "n": "november", "o": "oscar", "p": "papa", "q": "quebec", "r": "romeo",
    "s": "sierra", "t": "tango", "u": "uniform", "v": "victor", "w": "whiskey", "x": "xray",
    "y": "yankee", "z": "zulu",
}
ICAO_WORDS = set(ICAO.values())

# Single-word spelling variants -> canonical. Applied after lowercasing.
WORD_MAP = {
    # ICAO alphabet spellings
    "alpha": "alfa", "juliet": "juliett", "whisky": "whiskey", "x-ray": "xray",
    # ICAO number pronunciations
    "niner": "nine", "tree": "three", "fower": "four", "fife": "five", "tousand": "thousand",
    "hundert": "hundred",
    # common abbreviations / spellings
    "ok": "okay", "okey": "okay", "o.k.": "okay",
    "centre": "center", "fl": "flight level", "rwy": "runway", "hdg": "heading",
    "ft": "feet", "kt": "knots", "kts": "knots", "knot": "knots",
    "nm": "miles", "hpa": "hectopascals", "hectopascal": "hectopascals",
    "bye-bye": "bye bye", "byebye": "bye bye",
    "thank's": "thanks", "thankyou": "thank you",
    "°": "degrees",
}

# Hesitations, dropped from both sides.
FILLERS = {"ah", "aah", "eh", "ehm", "er", "erm", "uh", "uhm", "um", "umm", "hm", "hmm", "mm", "mhm", "em"}

# Multi-word phrases -> canonical (callsign telephony written split or joined, spelled acronyms).
# Matched on whole tokens after lowercasing. Keep this list to cases seen in the corpora / outputs.
PHRASE_MAP = {
    # airline telephony, written both split and joined in the corpora
    "speed bird": "speedbird",
    "sky travel": "skytravel",
    "swiss air": "swissair",
    "top swiss": "topswiss",
    "euro wings": "eurowings",
    "german wings": "germanwings",
    "euro trans": "eurotrans",
    "trans avia": "transavia",
    "ryan air": "ryanair",
    "lux air": "luxair",
    "finn air": "finnair",
    "japan air": "japanair",
    "egypt air": "egyptair",
    "korean air": "koreanair",
    "bee air": "beeair",
    "time air": "timeair",
    "sun turk": "sunturk",
    "wizz air": "wizzair",
    "wiz air": "wizzair",
    "smart lynx": "smartlynx",
    "sky shuttle": "skyshuttle",
    "easy jet": "easyjet",
    "air force": "airforce",
    # ATC compounds written both ways
    "stand by": "standby",
    "air space": "airspace",
    "push back": "pushback",
    "back track": "backtrack",
    "touch down": "touchdown",
    "re cleared": "recleared",
    "good bye": "goodbye",
    "x ray": "xray",
    # contractions: apostrophes are stripped, some corpora split them instead
    "we re": "were", "we ll": "well", "we ve": "weve", "you re": "youre", "they re": "theyre",
    "i m": "im", "i ll": "ill", "i ve": "ive", "it s": "its", "that s": "thats",
    "don t": "dont", "can t": "cant", "didn t": "didnt", "let s": "lets",
    # acronyms spelled out letter by letter
    "i l s": "ils",
    "q n h": "qnh",
    "v o r": "vor",
    "d m e": "dme",
    "i f r": "ifr",
    "v f r": "vfr",
    "a t i s": "atis",
}

# All-caps tokens that are spoken as words (or as a well-known letter group) and must NOT be
# spelled out letter by letter.
KEEP_UPPER = {
    "ILS", "QNH", "QFE", "IFR", "VFR", "VOR", "DME", "NDB", "ATIS", "RNAV", "RNP", "TCAS",
    "CAVOK", "ATC", "ACC", "APP", "TWR", "GND", "UTC", "GPS", "ETA", "ETD", "SID", "STAR",
    "TAP", "LOT", "OK", "NATO", "UK", "US", "USA",
    "TUI", "OKAY", "ROGER", "WILCO", "AFFIRM", "NEGATIVE", "MAYDAY", "PAN", "SQUAWK",
}

UNIT_SUFFIX = {"ft": "feet", "kt": "knots", "kts": "knots", "nm": "miles", "hpa": "hectopascals", "z": "zulu"}

# Airline / company names spoken as letters. Corpora write them both ways ("klm", "k l m"), so
# they are always spelled out.
SPELLED = {"klm", "csa", "sas", "ups", "dhl", "tnt", "tui", "ltu", "ana", "ba"}

RUNWAY_SIDE = {"l": "left", "r": "right", "c": "center"}

# --------------------------------------------------------------------------- text cleanup

_MARKUP = [
    re.compile(r"\[[^\]]*\]"),          # [noise], [HNOISE], [unk], [#...]
    re.compile(r"<[^>]*>"),             # <OT>, <FL>, xml-ish tags
    re.compile(r"\([^)]*\)"),           # (-ding) truncation notes, (spoken form) annotations
    re.compile(r"\{[^}]*\}"),
    re.compile(r"(?<![\w-])[\w']+-(?=\s|$)"),  # truncated fragments: "re-", "pre-"
    re.compile(r"(?<=\s)-[\w']+(?![\w-])"),     # "-ding"
]


def _strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


def _mostly_upper(s: str) -> bool:
    letters = [c for c in s if c.isalpha()]
    return bool(letters) and sum(c.isupper() for c in letters) / len(letters) > 0.5


_HYPHEN_PREFIXES = {"RE", "CO", "NO", "UP", "SO", "GO", "BY", "TO", "IN", "ON", "DE", "E", "X"}


def _registration(m: re.Match) -> str:
    if m.group(1) in _HYPHEN_PREFIXES:
        return m.group(0)
    return f"{_spell(m.group(1))} {_spell(m.group(2))}"


def _pre_regex(s: str) -> str:
    """Rewrites that need the original characters (case, '.', ',', '-', '/')."""
    s = re.sub(r"\bx[\s-]?ray\b", "xray", s, flags=re.I)
    # flight level: FL350, FL 350, F/L350, fl350
    s = re.sub(r"\bF\s?/?\s?L\s*(?=\d)", "flight level ", s, flags=re.I)
    # registrations like OK-HHH, D-EABC, N-123AB: spell both halves letter by letter
    s = re.sub(r"\b([A-Z]{1,2})-([A-Z0-9]{3,5})\b", _registration, s)
    # thousands separators: 4,500 / 12,000
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)
    # decimals: 118.1, 118,1, 132.775
    s = re.sub(r"(?<=\d)[.,](?=\d)", " decimal ", s)
    # degrees sign, times 12:30
    s = s.replace("°", " degrees ")
    s = re.sub(r"(?<=\d):(?=\d)", " ", s)
    return s


def _spell(run: str) -> str:
    out = []
    for ch in run:
        if ch.isdigit():
            out.append(ch)
        elif ch.lower() in ICAO:
            out.append(ICAO[ch.lower()])
    return " ".join(out)


_TOKEN = re.compile(r"[A-Za-z0-9']+")
_RUNS = re.compile(r"\d+|[A-Za-z]+")


def _tokenize(s: str) -> list[str]:
    """Split into tokens; split letter/digit runs; spell out all-caps letter groups."""
    mostly_upper = _mostly_upper(s)

    out: list[str] = []
    for tok in _TOKEN.findall(s):
        tok = tok.strip("'")
        if not tok:
            continue
        has_digit = any(c.isdigit() for c in tok)
        has_alpha = any(c.isalpha() for c in tok)
        if has_digit and has_alpha:
            # mixed token: 27L, EWG7AB, 3000ft, 1230Z, N123AB
            prev_word = out[-1].lower() if out else ""
            runs = _RUNS.findall(tok)
            for k, run in enumerate(runs):
                low = run.lower()
                if run.isdigit():
                    out.append(run)
                elif low in RUNWAY_SIDE and k > 0 and runs[k - 1].isdigit() and prev_word in ("runway", "rwy"):
                    out.append(RUNWAY_SIDE[low])
                elif low in UNIT_SUFFIX and k > 0 and runs[k - 1].isdigit():
                    out.append(UNIT_SUFFIX[low])
                elif run.upper() in KEEP_UPPER or len(run) > 3:
                    out.append(low)
                else:
                    out.extend(_spell(run).split())
        elif (
            has_alpha
            and not mostly_upper
            and tok.isupper()
            and tok not in KEEP_UPPER
            and tok not in ("A", "I")
            and len(tok) <= 5
        ):
            # all-caps letter group in mixed-case text: HHH, EAB, K -> spelled letters
            out.extend(_spell(tok).split())
        else:
            out.append(tok)
    return out


# --------------------------------------------------------------------------- numbers


def _is_num(t: str) -> bool:
    return t.isdigit() or t in UNITS or t in TEENS or t in TENS or t in SCALES


def _small(toks: list[str], i: int) -> tuple[str | None, int]:
    """Parse a value < 100 (or a digit string) at i. Returns (digit string, next index)."""
    t = toks[i]
    if t.isdigit():
        return t, i + 1
    if t in UNITS:
        return str(UNITS[t]), i + 1
    if t in TEENS:
        return str(TEENS[t]), i + 1
    if t in TENS:
        if i + 1 < len(toks) and toks[i + 1] in UNITS and toks[i + 1] != "zero":
            return str(TENS[t] + UNITS[toks[i + 1]]), i + 2
        return str(TENS[t]), i + 1
    return None, i


def _group(toks: list[str], i: int) -> tuple[str, int]:
    """Parse one number group at i. Digit-by-digit readings come back one digit at a time;
    compound readings ("four thousand five hundred", "nineteen hundred") are summed."""
    n = len(toks)
    if toks[i] in SCALES:
        v, j = "1", i
    else:
        v, j = _small(toks, i)
        assert v is not None
    if j >= n or toks[j] not in SCALES:
        return v, j

    total = 0
    if toks[j] == "thousand":
        total = int(v) * 1000
        j += 1
        # optional "<small> hundred"
        if j < n and _small(toks, j)[0] is not None:
            h, k = _small(toks, j)
            if k < n and toks[k] == "hundred":
                total += int(h) * 100
                j = k + 1
    else:  # hundred
        total = int(v) * 100
        j += 1
    # optional remainder: "and fifty", "twenty", "fifteen"; a bare unit only after "and"
    if j < n:
        has_and = toks[j] == "and" and j + 1 < n and _small(toks, j + 1)[0] is not None
        k = j + 1 if has_and else j
        if k < n and (toks[k] in TENS or toks[k] in TEENS or (has_and and toks[k] in UNITS)):
            r, k2 = _small(toks, k)
            total += int(r)
            j = k2
    return str(total), j


def _numbers_to_digits(toks: list[str]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(toks):
        if _is_num(toks[i]):
            digits, i = _group(toks, i)
            out.extend(DIGIT_WORDS[int(d)] for d in digits)
        else:
            out.append(toks[i])
            i += 1
    return out


# --------------------------------------------------------------------------- main


def _apply_phrases(toks: list[str]) -> list[str]:
    keys = sorted(((k.split(), v) for k, v in PHRASE_MAP.items()), key=lambda kv: -len(kv[0]))
    out: list[str] = []
    i = 0
    while i < len(toks):
        for k, v in keys:
            if toks[i : i + len(k)] == k:
                out.extend(v.split())
                i += len(k)
                break
        else:
            out.append(toks[i])
            i += 1
    return out


def _expand_repeats(toks: list[str]) -> list[str]:
    """'double seven' -> 'seven seven', 'triple hotel' -> 'hotel hotel hotel'."""
    out: list[str] = []
    i = 0
    rep = {"double": 2, "triple": 3}
    while i < len(toks):
        t = toks[i]
        if t in rep and i + 1 < len(toks) and (toks[i + 1] in UNITS or toks[i + 1] in ICAO_WORDS):
            out.extend([toks[i + 1]] * rep[t])
            i += 2
        else:
            out.append(t)
            i += 1
    return out


def normalize(text: str) -> str:
    """Canonical ATC spoken form. Apply to both references and hypotheses."""
    s = unicodedata.normalize("NFKC", text)
    for pat in _MARKUP:
        s = pat.sub(" ", s)
    s = _strip_accents(s)
    s = _pre_regex(s)
    s = s.replace("-", " ").replace("/", " ").replace("_", " ")

    toks = [t.lower().replace("'", "") for t in _tokenize(s)]
    toks = [w for t in toks for w in WORD_MAP.get(t, t).split()]
    toks = [w for t in toks for w in (_spell(t).split() if t in SPELLED else [t])]
    toks = [t for t in toks if t not in FILLERS]
    toks = _apply_phrases(toks)
    # single letters are spelled letters ("k oscar zulu"); "a" and "i" stay words
    toks = [ICAO[t] if len(t) == 1 and t.isalpha() and t not in ("a", "i") else t for t in toks]
    toks = _numbers_to_digits(toks)
    toks = _expand_repeats(toks)
    # "point" between digits is the decimal separator
    toks = [
        "decimal" if t == "point" and 0 < k < len(toks) - 1 and toks[k - 1] in UNITS and toks[k + 1] in UNITS else t
        for k, t in enumerate(toks)
    ]
    return " ".join(toks)


def basic_normalize(text: str) -> str:
    """Lowercase + strip punctuation only. Used to show how much the ATC normalizer matters."""
    s = unicodedata.normalize("NFKC", text).lower()
    s = re.sub(r"[^\w\s']", " ", s)
    return " ".join(s.split())
