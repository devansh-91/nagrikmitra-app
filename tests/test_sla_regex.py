"""
SLA-lock regex tests — the guard against false positives described in the spec.

Cover:
- "48 hours" matches sla_regex(48), does NOT match sla_regex(4) or sla_regex(8).
- "P1 ticket, response due" (contains digit "1") does NOT match sla_regex(1).
- "168 hours" does NOT falsely match sla_regex(1) or sla_regex(6) via substring.
- Hindi "48 ghante" matches sla_regex(48).
"""
from nagrikmitra.reply import sla_regex


def test_48_hours_matches_48():
    assert sla_regex(48).search("48 hours")


def test_48_hours_does_not_match_4():
    assert not sla_regex(4).search("48 hours")


def test_48_hours_does_not_match_8():
    assert not sla_regex(8).search("48 hours")


def test_p1_ticket_does_not_match_1():
    assert not sla_regex(1).search("P1 ticket, response due")


def test_168_hours_does_not_falsely_match_1():
    assert not sla_regex(1).search("168 hours")


def test_168_hours_does_not_falsely_match_6():
    assert not sla_regex(6).search("168 hours")


def test_168_hours_matches_168():
    assert sla_regex(168).search("168 hours")


def test_hindi_ghante_matches():
    assert sla_regex(48).search("48 ghante")


def test_hrs_abbreviation_matches():
    assert sla_regex(336).search("336 hrs")
