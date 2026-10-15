"""Stage 6 / Task 1.1 tests: channel extraction and power-set coalitions.

Every channel list and path below is a TEST FIXTURE, invented only to isolate one rule. Fixtures
never feed any project result. The last test runs on the real data/processed/journeys.csv and is
skipped if Stage 3 has not been run.
"""
from math import comb
from pathlib import Path

import pandas as pd
import pytest

from coalition_generator import (
    CoalitionLimitError,
    CoalitionValidationError,
    build_and_save,
    deserialize_coalition,
    extract_channels,
    extract_channels_from_journeys,
    generate_coalitions,
    load_coalitions,
    power_set_size,
    save_coalitions,
    serialize_coalition,
    validate_coalitions,
)
from mta.journey_builder import DEFAULT_OUTPUT_PATH


# ---------------------------------------------------------------- the 10 required tests
def test_01_three_channels_give_eight_coalitions():
    assert len(generate_coalitions(["A", "B", "C"])) == 8


def test_02_empty_coalition_exists():
    assert frozenset() in generate_coalitions(["A", "B", "C"])


def test_03_grand_coalition_exists():
    assert frozenset({"A", "B", "C"}) in generate_coalitions(["A", "B", "C"])


def test_04_no_duplicate_coalitions():
    cs = generate_coalitions(["A", "B", "C", "D"])
    assert len(cs) == len(set(cs)) == 16


def test_05_sizes_range_from_zero_to_n_with_binomial_counts():
    cs = generate_coalitions(["A", "B", "C", "D"])
    sizes = [len(c) for c in cs]
    assert min(sizes) == 0 and max(sizes) == 4
    assert [sizes.count(k) for k in range(5)] == [comb(4, k) for k in range(5)]


def test_06_structural_states_are_excluded():
    assert extract_channels(["Start", "A", "B", "Conversion", "Null"]) == ["A", "B"]
    assert extract_channels(["Start > A > B > Conversion", "Start > C > Null"]) == ["A", "B", "C"]


def test_07_duplicate_occurrences_become_one_player():
    assert extract_channels(["Google", "Email", "Google", "Email"]) == ["Email", "Google"]


def test_08_order_independence():
    assert generate_coalitions(["A", "B", "C"]) == generate_coalitions(["C", "A", "B"])


def test_09_input_is_not_mutated():
    channels = ["C", "A", "B", "A"]
    snapshot = list(channels)
    generate_coalitions(channels)
    extract_channels(channels)
    assert channels == snapshot


def test_10_grand_coalition_contains_every_extracted_channel():
    channels = extract_channels(["Start > Social > Email > Conversion", "Start > Search > Null"])
    grand = generate_coalitions(channels)[-1]
    assert grand == frozenset(channels) == frozenset({"Social", "Email", "Search"})


# ---------------------------------------------------------------- edge cases
def test_no_channels_gives_only_the_empty_coalition():
    assert generate_coalitions([]) == [frozenset()]
    assert power_set_size(0) == 1


def test_one_channel_gives_two_coalitions():
    assert generate_coalitions(["A"]) == [frozenset(), frozenset({"A"})]


def test_start_conversion_only_journey_has_no_players():
    assert extract_channels(["Start > Conversion"]) == []


def test_null_ending_journey_keeps_its_channel():
    assert extract_channels(["Start > A > Null"]) == ["A"]


def test_channel_in_many_journeys_is_one_player():
    paths = ["Start > A > Conversion"] * 50 + ["Start > A > B > Null"] * 50
    assert extract_channels(paths) == ["A", "B"]


def test_channel_names_with_spaces_are_kept_intact():
    assert extract_channels(["Start > Search Ads > Social Media > Null"]) == ["Search Ads", "Social Media"]


def test_empty_state_name_is_rejected():
    with pytest.raises(ValueError):
        extract_channels([["Start", " ", "Null"]])


def test_non_string_state_is_rejected():
    with pytest.raises(TypeError):
        extract_channels([["Start", 5, "Null"]])


# ---------------------------------------------------------------- ordering / representation
def test_canonical_order_is_size_then_alphabetical():
    cs = generate_coalitions(["B", "A", "C"])
    assert [serialize_coalition(c) for c in cs] == ["", "A", "B", "C", "A|B", "A|C", "B|C", "A|B|C"]


def test_same_members_are_the_same_coalition():
    assert frozenset({"Google", "Email"}) == frozenset({"Email", "Google"})
    assert serialize_coalition(frozenset({"Google", "Email"})) == "Email|Google"


def test_serialize_roundtrip_including_empty():
    for c in generate_coalitions(["Search Ads", "Email", "Social Media"]):
        assert deserialize_coalition(serialize_coalition(c)) == c
    assert serialize_coalition(frozenset()) == ""
    assert deserialize_coalition("") == frozenset()


def test_channel_containing_separator_is_rejected():
    with pytest.raises(ValueError):
        generate_coalitions(["A|B", "C"])
    with pytest.raises(ValueError):
        serialize_coalition(frozenset({"A|B"}))


def test_structural_name_as_player_is_rejected():
    with pytest.raises(CoalitionValidationError):
        generate_coalitions(["A", "Start"])
    with pytest.raises(CoalitionValidationError):
        generate_coalitions(["null"])


# ---------------------------------------------------------------- safety limit
def test_limit_blocks_impractical_power_sets_before_generating():
    with pytest.raises(CoalitionLimitError, match="impractically large power set"):
        generate_coalitions([f"c{i}" for i in range(25)])  # default limit is 20
    with pytest.raises(CoalitionLimitError):
        generate_coalitions(["A", "B", "C"], max_channels=2)


def test_limit_is_configurable_and_does_not_truncate():
    assert len(generate_coalitions(["A", "B", "C"], max_channels=3)) == 8
    assert len(generate_coalitions([f"c{i}" for i in range(12)])) == 4096


# ---------------------------------------------------------------- validation catches errors
def good(channels=("A", "B", "C")):
    return list(channels), generate_coalitions(channels)


def test_validation_passes_on_correct_output():
    ch, cs = good()
    validate_coalitions(cs, ch)


def test_validation_detects_missing_coalition():
    ch, cs = good()
    with pytest.raises(CoalitionValidationError, match="expected 2\\^3"):
        validate_coalitions(cs[:-1], ch)


def test_validation_detects_duplicate_replacing_a_coalition():
    ch, cs = good()
    cs[3] = cs[2]
    with pytest.raises(CoalitionValidationError, match="duplicate"):
        validate_coalitions(cs, ch)


def test_validation_detects_foreign_member():
    ch, cs = good()
    cs[1] = frozenset({"Z"})
    with pytest.raises(CoalitionValidationError, match="non-players"):
        validate_coalitions(cs, ch)


def test_validation_detects_wrong_order():
    ch, cs = good()
    cs[1], cs[2] = cs[2], cs[1]
    with pytest.raises(CoalitionValidationError, match="canonical"):
        validate_coalitions(cs, ch)


def test_validation_detects_missing_empty_and_grand():
    ch, cs = good()
    no_empty = [frozenset({"A"})] + cs[1:]
    with pytest.raises(CoalitionValidationError):
        validate_coalitions(no_empty, ch)
    no_grand = cs[:-1] + [frozenset({"A", "B"})]
    with pytest.raises(CoalitionValidationError):
        validate_coalitions(no_grand, ch)


def test_validation_rejects_duplicate_or_structural_channels():
    _, cs = good()
    with pytest.raises(CoalitionValidationError):
        validate_coalitions(cs, ["A", "B", "C", "C"])
    with pytest.raises(CoalitionValidationError):
        validate_coalitions(cs, ["A", "B", "Null"])


# ---------------------------------------------------------------- DataFrame / CSV
def journeys_frame(paths):
    return pd.DataFrame({"journey_id": [f"u{i}#1" for i in range(len(paths))], "path": paths,
                         "segment": "new", "converted": [int(p.endswith("Conversion")) for p in paths]})


def test_extract_from_dataframe_uses_person_a_path_validation():
    df = journeys_frame(["Start > Social > Email > Conversion", "Start > Search > Null"])
    assert extract_channels_from_journeys(df) == ["Email", "Search", "Social"]
    with pytest.raises(ValueError):
        extract_channels_from_journeys(journeys_frame(["Social > Email > Conversion"]))  # no Start
    with pytest.raises(ValueError):
        extract_channels_from_journeys(pd.DataFrame({"x": [1]}))


def test_save_and_load_roundtrip_has_2_pow_n_rows(tmp_path):
    channels = ["Search Ads", "Email", "Social Media"]
    cs = generate_coalitions(channels)
    out = save_coalitions(cs, tmp_path / "sub" / "coalitions.csv")
    raw = pd.read_csv(out, keep_default_na=False)
    assert list(raw.columns) == ["coalition_id", "coalition_size", "coalition"]
    assert len(raw) == 2 ** 3
    assert raw.loc[0, "coalition"] == "" and raw.loc[0, "coalition_size"] == 0
    assert raw.loc[len(raw) - 1, "coalition"] == "Email|Search Ads|Social Media"
    assert raw["coalition_id"].tolist() == list(range(8))
    assert load_coalitions(out) == cs


def test_output_is_byte_identical_across_runs_and_input_orders(tmp_path):
    a = save_coalitions(generate_coalitions(["B", "A", "C"]), tmp_path / "a.csv")
    b = save_coalitions(generate_coalitions(["C", "B", "A"]), tmp_path / "b.csv")
    assert a.read_bytes() == b.read_bytes()


def test_build_and_save_end_to_end_on_fixture_journeys(tmp_path):
    jp = tmp_path / "journeys.csv"
    journeys_frame(["Start > Social > Email > Conversion", "Start > Search > Null",
                    "Start > Email > Conversion"]).to_csv(jp, index=False)
    r = build_and_save(jp, tmp_path / "coalitions.csv")
    assert r["channels"] == ["Email", "Search", "Social"]
    assert r["n_coalitions"] == r["expected_coalitions"] == 8 == len(pd.read_csv(r["output_path"], keep_default_na=False))


def test_build_and_save_with_zero_channels(tmp_path):
    jp = tmp_path / "journeys.csv"
    journeys_frame(["Start > Conversion"]).to_csv(jp, index=False)
    r = build_and_save(jp, tmp_path / "coalitions.csv")
    assert r["n_channels"] == 0 and r["n_coalitions"] == 1
    assert load_coalitions(r["output_path"]) == [frozenset()]


def test_build_and_save_missing_journeys_file_fails_clearly(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_and_save(tmp_path / "nope.csv", tmp_path / "out.csv")


# ---------------------------------------------------------------- real data
def test_real_journeys_give_2_pow_n_coalitions(tmp_path):
    if not Path(DEFAULT_OUTPUT_PATH).exists():
        pytest.skip("data/processed/journeys.csv not found; run scripts/run_stage3.py first")
    r = build_and_save(DEFAULT_OUTPUT_PATH, tmp_path / "coalitions.csv")
    assert r["n_coalitions"] == r["expected_coalitions"] == 2 ** r["n_channels"]
    assert not set(r["channels"]) & {"Start", "Conversion", "Null"}
