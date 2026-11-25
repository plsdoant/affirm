import pytest

from affirm.score import score_file


def test_score_counts():
    rows = [
        {"ref": "lufthansa three five five six climb flight level two nine zero", "hyp": "Lufthansa 3556, climb FL290."},
        {"ref": "speedbird one two contact one one eight decimal one", "hyp": "Speedbird 12 contact 118.7"},
        {"ref": "roger", "hyp": ""},
    ]
    s = score_file("m", "c", rows, n_boot=50)
    assert s.ref_words == 11 + 9 + 1
    # one digit substitution (1 -> 7) + one deletion ("roger")
    assert s.wer == pytest.approx(2 / 21)
    assert s.digit_er == pytest.approx(1 / 13)  # 7 + 6 reference digits
    assert s.wer_basic > s.wer
    assert s.wer_ci[0] <= s.wer <= s.wer_ci[1]
