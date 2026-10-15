"""Stage 12 tests.

The weight tables below are small TEST FIXTURES with hand-chosen numbers, never project results.
The last tests use the project's real hybrid_attribution.csv when it exists.

Hand-solved examples
  weights 0.40/0.35/0.25, budget 10,00,000      -> 4,00,000 / 3,50,000 / 2,50,000 (no rounding needed)
  weights 0.5/0.3/0.2,   budget 7               -> quotas 3.5/2.1/1.4, floors 3/2/1 (=6), the 1 leftover
                                                   rupee goes to the largest remainder (0.5): 4/2/1
  three equal weights,   budget 100             -> quotas 33.33 each, floors 33 (=99), the leftover rupee
                                                   goes to the alphabetically first channel: 34/33/33
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from budget_allocator import (
    ALLOCATION_COLUMNS,
    ALLOCATION_NOTICE,
    DEFAULT_HYBRID_PATH,
    MAX_BUDGET,
    allocate_budget,
    allocation_summary,
    check_allocation,
    export_allocation,
    format_inr,
    load_hybrid_attribution,
    parse_budget_text,
    validate_budget,
    validate_weights,
)


def wt(mapping):
    return pd.DataFrame({"channel": list(mapping), "hybrid_weight": list(mapping.values())})


def amounts(table):
    return table.set_index("channel")["recommended_budget"].to_dict()


# ---------------------------------------------------------------- arithmetic
def test_allocation_matches_hand_calculation_without_rounding():
    t = allocate_budget(wt({"A": 0.40, "B": 0.35, "C": 0.25}), 1_000_000)
    assert amounts(t) == {"A": 400_000, "B": 350_000, "C": 250_000}
    assert t["hybrid_percentage"].tolist() == pytest.approx([40.0, 35.0, 25.0])


def test_leftover_rupee_goes_to_the_largest_remainder():
    t = allocate_budget(wt({"A": 0.5, "B": 0.3, "C": 0.2}), 7)
    assert amounts(t) == {"A": 4, "B": 2, "C": 1}


def test_ties_are_broken_alphabetically():
    t = allocate_budget(wt({"C": 1 / 3, "B": 1 / 3, "A": 1 / 3}), 100)
    assert amounts(t) == {"A": 34, "B": 33, "C": 33}


def test_two_leftover_rupees_go_to_the_two_largest_remainders():
    # quotas for budget 10 with weights .45/.25/.30 -> 4.5/2.5/3.0: floors 4/2/3 = 9, one leftover
    # (remainders .5, .5, 0 tie between A and B -> A). Budget 11: 4.95/2.75/3.3 -> floors 4/2/3 = 9, two leftover
    t = allocate_budget(wt({"A": 0.45, "B": 0.25, "C": 0.30}), 11)
    assert amounts(t) == {"A": 5, "B": 3, "C": 3}


def test_zero_budget_gives_zero_allocations():
    t = allocate_budget(wt({"A": 0.6, "B": 0.4}), 0)
    assert t["recommended_budget"].tolist() == [0, 0]
    assert (t["unrounded_budget"] == 0).all()


def test_zero_weight_channel_gets_nothing_even_when_rupees_are_left_over():
    t = allocate_budget(wt({"A": 0.5, "B": 0.5, "Z": 0.0}), 3)
    assert amounts(t) == {"A": 2, "B": 1, "Z": 0}


def test_one_channel_with_all_the_weight_gets_the_whole_budget():
    assert amounts(allocate_budget(wt({"A": 1.0, "B": 0.0}), 12345)) == {"A": 12345, "B": 0}


def test_single_channel():
    assert amounts(allocate_budget(wt({"A": 1.0}), 99)) == {"A": 99}


# ---------------------------------------------------------------- reconciliation (property-style)
def test_allocations_always_reconcile_and_stay_within_one_rupee_of_the_exact_share():
    rng = np.random.default_rng(12345)
    for _ in range(300):
        n = int(rng.integers(1, 9))
        w = rng.dirichlet(np.ones(n))
        w[rng.random(n) < 0.15] = 0.0  # sprinkle exact zeros
        if w.sum() == 0:
            w[0] = 1.0
        w = w / w.sum()
        budget = int(rng.integers(0, 50_000_000))
        t = allocate_budget(wt({f"ch{i}": float(x) for i, x in enumerate(w)}), budget)
        assert int(t["recommended_budget"].sum()) == budget
        assert (t["recommended_budget"] >= 0).all()
        assert (t["rounding_adjustment"].abs() < 1).all()
        assert t["rounding_adjustment"].sum() == pytest.approx(0.0, abs=1e-6)  # rupees given == rupees taken


def test_weights_summing_to_one_only_within_tolerance_still_reconcile_exactly():
    t = allocate_budget(wt({"A": 0.5000004, "B": 0.5}), 1_000_001)  # sums to 1.0000004
    assert int(t["recommended_budget"].sum()) == 1_000_001


def test_output_is_independent_of_input_row_order():
    base = wt({"A": 0.2, "B": 0.5, "C": 0.3})
    shuffled = base.sample(frac=1, random_state=3).reset_index(drop=True)
    pd.testing.assert_frame_equal(allocate_budget(base, 1_000_003), allocate_budget(shuffled, 1_000_003))


def test_results_are_deterministic():
    w = wt({"A": 0.2, "B": 0.5, "C": 0.3})
    pd.testing.assert_frame_equal(allocate_budget(w, 777), allocate_budget(w, 777))


def test_output_schema_types_and_ordering():
    t = allocate_budget(wt({"A": 0.2, "B": 0.5, "C": 0.3}), 1000)
    assert list(t.columns) == ALLOCATION_COLUMNS
    assert t["recommended_budget"].dtype == np.int64
    assert t["channel"].tolist() == ["B", "C", "A"]  # largest allocation first


def test_input_is_not_modified():
    w = wt({"A": 0.2, "B": 0.8})
    before = w.copy()
    allocate_budget(w, 100)
    pd.testing.assert_frame_equal(w, before)


def test_extra_input_columns_are_ignored():
    w = wt({"A": 0.2, "B": 0.8}).assign(hybrid_percentage=[999, 999], note="x")
    t = allocate_budget(w, 100)
    assert t["hybrid_percentage"].tolist() == pytest.approx([80.0, 20.0])  # recomputed, never trusted from input


# ---------------------------------------------------------------- budget validation
@pytest.mark.parametrize("bad", [-1, -0.0001, -1_000_000])
def test_negative_budget_raises(bad):
    with pytest.raises(ValueError):
        validate_budget(bad)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_budget_raises(bad):
    with pytest.raises(ValueError, match="finite"):
        validate_budget(bad)


@pytest.mark.parametrize("bad", [1000.5, 0.01, 99.999])
def test_fractional_rupees_are_rejected_not_rounded(bad):
    with pytest.raises(ValueError, match="whole number of rupees"):
        validate_budget(bad)


@pytest.mark.parametrize("bad", ["1000", None, True, False, [100], 1 + 2j])
def test_non_numeric_budget_raises_type_error(bad):
    with pytest.raises(TypeError):
        validate_budget(bad)


def test_budget_above_the_supported_maximum_raises():
    with pytest.raises(ValueError, match="maximum"):
        validate_budget(MAX_BUDGET + 1)
    assert validate_budget(MAX_BUDGET) == MAX_BUDGET


@pytest.mark.parametrize("ok, expected", [(0, 0), (1_000_000, 1_000_000), (1e6, 1_000_000),
                                          (np.int64(500), 500), (np.float64(2500.0), 2500)])
def test_valid_budgets_are_accepted(ok, expected):
    assert validate_budget(ok) == expected


def test_bad_budget_stops_allocation():
    with pytest.raises(ValueError):
        allocate_budget(wt({"A": 1.0}), -5)


@pytest.mark.parametrize("text, expected", [("10,00,000", 1_000_000), ("₹ 10,00,000", 1_000_000),
                                            ("1000000", 1_000_000), (" 250 ", 250), ("1e6", 1_000_000.0)])
def test_budget_text_parsing(text, expected):
    assert parse_budget_text(text) == expected


@pytest.mark.parametrize("bad", ["", "   ", "abc", "₹", "1,0,0x"])
def test_unreadable_budget_text_raises(bad):
    with pytest.raises(ValueError):
        parse_budget_text(bad)


def test_parsed_fractional_text_is_still_rejected_by_validation():
    with pytest.raises(ValueError, match="whole number"):
        validate_budget(parse_budget_text("1000.50"))


# ---------------------------------------------------------------- weight validation
def test_negative_weight_raises():
    with pytest.raises(ValueError, match="negative"):
        validate_weights(wt({"A": 1.2, "B": -0.2}))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), "abc", None])
def test_non_finite_or_missing_weight_raises(bad):
    w = wt({"A": 0.5, "B": 0.5})
    w["hybrid_weight"] = w["hybrid_weight"].astype(object)
    w.loc[1, "hybrid_weight"] = bad
    with pytest.raises(ValueError, match="non-finite"):
        validate_weights(w)


@pytest.mark.parametrize("mapping", [{"A": 0.5, "B": 0.4}, {"A": 0.6, "B": 0.6}, {"A": 0.0, "B": 0.0}])
def test_weights_that_do_not_sum_to_one_raise(mapping):
    with pytest.raises(ValueError, match="expected 1"):
        validate_weights(wt(mapping))


def test_duplicate_blank_and_reserved_channels_raise():
    dup = pd.concat([wt({"A": 0.5}), wt({"A": 0.5})], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        validate_weights(dup)
    with pytest.raises(ValueError, match="blank"):
        validate_weights(wt({"A": 0.5, " ": 0.5}))
    for state in ("Start", "Conversion", "Null", "null"):
        with pytest.raises(ValueError, match="model state"):
            validate_weights(wt({"A": 0.5, state: 0.5}))


def test_empty_and_malformed_tables_raise():
    with pytest.raises(ValueError, match="no channels"):
        validate_weights(pd.DataFrame({"channel": [], "hybrid_weight": []}))
    with pytest.raises(ValueError, match="missing required column"):
        validate_weights(pd.DataFrame({"channel": ["A"], "weight": [1.0]}))


# ---------------------------------------------------------------- reconciliation checks
def test_check_allocation_catches_a_table_that_does_not_reconcile():
    t = allocate_budget(wt({"A": 0.5, "B": 0.5}), 100)
    broken = t.copy()
    broken.loc[0, "recommended_budget"] += 1
    with pytest.raises(ValueError, match="not the budget"):
        check_allocation(broken, 100)
    negative = t.copy()
    negative.loc[0, "recommended_budget"] = -1
    with pytest.raises(ValueError, match="negative"):
        check_allocation(negative, 100)


def test_allocation_summary_reports_reconciliation():
    t = allocate_budget(wt({"A": 0.4, "B": 0.6}), 1_000_000)
    s = allocation_summary(t, 1_000_000)
    assert amounts(t) == {"A": 400_000, "B": 600_000}
    assert (s["total_budget"], s["allocated_total"], s["difference"], s["reconciles"]) == (
        1_000_000, 1_000_000, 0, True)
    # float 0.4 is not exactly 2/5 in binary, so the adjustment is float noise, not exactly 0
    assert s["max_rounding_adjustment"] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------- rupee formatting
@pytest.mark.parametrize("amount, expected", [
    (0, "₹0"), (7, "₹7"), (999, "₹999"), (1000, "₹1,000"), (10_000, "₹10,000"), (100_000, "₹1,00,000"),
    (1_000_000, "₹10,00,000"), (12_345_678, "₹1,23,45,678"), (1_000_000_000, "₹1,00,00,00,000"),
    (-1_500_000, "-₹15,00,000"),
])
def test_indian_rupee_grouping(amount, expected):
    assert format_inr(amount) == expected


def test_rupee_formatting_with_decimals():
    assert format_inr(1234.5, decimals=2) == "₹1,234.50"
    assert format_inr(1_000_000, decimals=2) == "₹10,00,000.00"


# ---------------------------------------------------------------- interpretation notice
def test_notice_states_the_required_limitations():
    text = " ".join(ALLOCATION_NOTICE).lower()
    for phrase in ("hypothetical", "contribution-based", "no channel-level spend or revenue",
                   "roi", "roas", "not mean a channel is more cost-effective", "optimal", "not causal"):
        assert phrase in text, f"notice is missing: {phrase}"


# ---------------------------------------------------------------- IO and integration
def test_export_round_trip(tmp_path):
    t = allocate_budget(wt({"A": 0.4, "B": 0.6}), 1_000_000)
    out = export_allocation(t, tmp_path / "nested" / "alloc.csv")
    back = pd.read_csv(out)
    assert list(back.columns) == ALLOCATION_COLUMNS
    assert int(back["recommended_budget"].sum()) == 1_000_000


def test_hybrid_engine_output_feeds_the_allocator():
    from hybrid_engine import hybrid_attribution
    m = pd.DataFrame({"segment": "all", "channel": ["A", "B"], "baseline_conversion_probability": 0.8,
                      "removal_effect": [0.30, 0.10], "status": "ok"})
    s = pd.DataFrame({"channel": ["A", "B"], "shapley_value": [0.2, 0.6]})
    hybrid = hybrid_attribution(m, s, alpha=0.5)  # hand-solved earlier: weights 0.5 / 0.5
    t = allocate_budget(hybrid, 1_000_000)
    assert amounts(t) == {"A": 500_000, "B": 500_000}


def test_missing_hybrid_file_has_a_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="run_stage11"):
        load_hybrid_attribution(tmp_path / "nope.csv")


def test_hybrid_file_without_required_columns_raises(tmp_path):
    p = tmp_path / "h.csv"
    p.write_text("channel,score\nA,1\n")
    with pytest.raises(ValueError, match="missing required column"):
        load_hybrid_attribution(p)


REAL_HYBRID = Path(DEFAULT_HYBRID_PATH)


@pytest.mark.skipif(not REAL_HYBRID.exists(), reason="real hybrid_attribution.csv not found at the default path")
@pytest.mark.parametrize("budget", [0, 1, 999_999, 1_000_000, 25_00_000, 987_654_321])
def test_real_hybrid_output_allocates_and_reconciles(budget):
    hybrid = load_hybrid_attribution(REAL_HYBRID)
    t = allocate_budget(hybrid, budget)
    assert set(t["channel"]) == set(hybrid["channel"]) and len(t) == len(hybrid)
    assert int(t["recommended_budget"].sum()) == budget
    exact = hybrid.set_index("channel")["hybrid_weight"] * budget / hybrid["hybrid_weight"].sum()
    got = t.set_index("channel")["recommended_budget"]
    assert (got - exact.reindex(got.index)).abs().max() < 1.0
