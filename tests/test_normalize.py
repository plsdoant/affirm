import pytest

from affirm.normalize import basic_normalize, normalize

# (reference-style, model-style) pairs that say the same thing and must normalize identically
EQUIVALENT = [
    ("flight level three five zero", "FL350"),
    ("flight level three five zero", "FL 350"),
    ("flight level three five zero", "flight level 350"),
    ("flight level one hundred", "FL100"),
    ("flight level one hundred", "flight level one zero zero"),
    ("climb to flight level two four zero", "Climb to F/L240."),
    ("heading zero niner zero", "heading 090"),
    ("heading zero nine zero", "Heading zero niner zero."),
    ("descend four thousand five hundred feet", "Descend 4,500 ft."),
    ("descend four thousand five hundred feet", "descend four five zero zero feet"),
    ("passing nineteen hundred", "passing 1900"),
    ("climb five thousand feet", "climb 5000 feet"),
    ("squawk seven thousand", "squawk 7000"),
    ("squawk seven zero zero zero", "Squawk 7000"),
    ("contact one one eight decimal one", "contact 118.1"),
    ("contact one one eight decimal one", "contact one one eight point one"),
    ("contact one three two decimal seven seven five", "Contact 132.775"),
    ("one twenty seven decimal three", "127.3"),
    ("qnh one zero one three", "QNH 1013"),
    ("runway two seven left", "runway 27L"),
    ("runway zero six right", "Runway 06R"),
    ("cleared ils approach runway two four", "Cleared ILS approach runway 24"),
    ("lufthansa eight mike mike", "Lufthansa 8MM"),
    ("klm one two three four", "KLM1234"),
    ("k l m three hundred kilo", "KLM 300 kilo"),
    ("csa five zero eight", "CSA 508"),
    ("german wings seven papa", "Germanwings 7P"),
    ("oscar kilo triple hotel", "Oscar Kilo HHH"),
    ("oscar kilo hotel hotel hotel", "OK-HHH"),
    ("double seven", "seven seven"),
    ("alfa bravo", "Alpha Bravo"),
    ("juliett", "Juliet"),
    ("xray", "X-ray"),
    ("speedbird five six three", "Speed Bird 563"),
    ("skytravel two eight two seven", "Sky Travel 2827"),
    ("k oscar zulu", "Kilo Oscar Zulu"),
    ("speed two hundred and fifty knots", "speed 250 kts"),
    ("two five zero knots", "250 knots"),
    ("one one thousand", "11,000"),
    ("time one two three zero zulu", "time 1230Z"),
    ("good bye", "goodbye"),
    ("stand by", "Standby."),
    ("okay", "OK"),
    ("roger we re ready", "Roger, we're ready."),
    ("cleared ah kines st prex", "cleared kines st prex"),
    ("heading(-ding) two one zero", "heading two one zero"),
    ("Sierra(Sie-) one", "Sierra one"),
    ("re- cleared", "cleared"),
    ("dobré ráno", "dobre rano"),
    ("schone_tag", "schone tag"),
]


@pytest.mark.parametrize("ref,hyp", EQUIVALENT)
def test_equivalent(ref, hyp):
    assert normalize(ref) == normalize(hyp)


EXACT = [
    ("FL350", "flight level three five zero"),
    ("Descend 4,500 ft.", "descend four five zero zero feet"),
    ("contact 118.1", "contact one one eight decimal one"),
    ("runway 27L", "runway two seven left"),
    ("Lufthansa 8MM", "lufthansa eight mike mike"),
    ("Eurowings Seven Alfa Bravo turn right heading two one zero", "eurowings seven alfa bravo turn right heading two one zero"),
    ("a holding point", "a holding point"),  # "a" and "point" are words here, not a letter / decimal
    ("I have it", "i have it"),
    ("one moment", "one moment"),
]


@pytest.mark.parametrize("text,expected", EXACT)
def test_exact(text, expected):
    assert normalize(text) == expected


NOT_EQUIVALENT = [
    ("flight level three five zero", "flight level three six zero"),
    ("runway two seven left", "runway two seven right"),
    ("climb", "descend"),
    ("heading one two zero", "heading two one zero"),
]


@pytest.mark.parametrize("a,b", NOT_EQUIVALENT)
def test_not_equivalent(a, b):
    assert normalize(a) != normalize(b)


def test_idempotent():
    for ref, hyp in EQUIVALENT:
        for s in (ref, hyp):
            once = normalize(s)
            assert normalize(once) == once, s


def test_basic_normalize():
    assert basic_normalize("Climb to FL350, Lufthansa 8MM.") == "climb to fl350 lufthansa 8mm"
