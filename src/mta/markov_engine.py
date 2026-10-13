"""Stage 4: first-order Markov transition model over customer journeys.

Stage 4: transition counts, transition probabilities, absorbing terminal states, and the
probability of EVENTUALLY reaching Conversion from Start.
Stage 5: channel removal effects (see "Removal policy" below). Stage 4 functions are unchanged.

Removal policy (Stage 5): REDIRECT TO NULL
  To remove channel k, every transition INTO k is redirected to Null: P'(i, Null) = P(i, Null) +
  P(i, k) and P'(i, k) = 0, for every state i (including k's own self-loop). Rows still sum to 1.
  Reading: a journey that would have touched k is lost, so it can no longer convert. k stays in the
  matrix as an unreachable dummy state that is absorbed into Null, so every model keeps the same
  row/column labels. Nothing is renormalised and no probability is invented.
  Why not just delete k's column? That leaves rows summing to less than 1, and "fixing" it by
  renormalising silently changes the assumption to "users who would have touched k switch to the
  other channels", which understates every channel's removal effect.
  Removal effect: RE_k = 1 - CR_without_k / CR_base, where CR is the probability of EVENTUALLY
  reaching Conversion from Start. Under this policy CR_without_k <= CR_base, so 0 <= RE_k <= 1.
  Special cases (all explicit, all finite): CR_base = 0 gives RE = 0 for every channel with status
  'zero_baseline'; a channel absent from the journeys gives RE = 0 with status 'not_in_journeys'.

Model in one paragraph
  States: Start, one state per channel, and two terminal states, Conversion and Null.
  The model is first-order: the next state depends only on the current state, not on how the
  journey got there. Each probability P(i -> j) is the observed share n(i, j) / n(i) of departures
  from state i that went to j. Conversion and Null are absorbing: once there, the chain stays
  there forever, P(Conversion -> Conversion) = P(Null -> Null) = 1.

Assumptions and limitations (read before trusting any number)
  1. Memoryless: path history before the current touch, and time gaps between touches, are ignored.
  2. Probabilities are raw observed frequencies with no smoothing. Rows with few observations are
     noisy; the model flags states with fewer than SPARSE_ROW_THRESHOLD departures.
  3. Results inherit the Stage 3 journey definition. Under "a journey ends at the first
     conversion", Null journeys only arise from a user's trailing events, so the base conversion
     probability is a property of that definition, not a general visitor conversion rate.
  4. A first-order chain fitted to complete journeys reproduces the observed share of converting
     journeys exactly. That is an implementation sanity check, not evidence the model is right.
  5. 'new'/'returning' segments are structural under the reset-after-conversion rule (every journey
     after a user's first is 'returning') and only see the 48 hours in the data.
  6. This is an attribution method over observed correlations, not proof of causal marketing lift.
  7. A removal effect credits channel k with every conversion whose path runs THROUGH k, including
     the work later channels did after k. Because the chain is first-order it can also recombine
     transitions seen in different journeys, so it is not the same as "drop every journey that
     contains k". Removal effects need not sum to 1 (see the Stage 5 notes in the reporting).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from mta.journey_builder import CONVERSION, DEFAULT_OUTPUT_PATH, NULL, PATH_SEP, START

TERMINAL_STATES = (CONVERSION, NULL)
SEGMENTS = ("new", "returning")
MIN_JOURNEYS_PER_SEGMENT = 200  # heuristic, not a statistical guarantee; adjustable
MIN_CONVERSIONS_PER_SEGMENT = 30  # heuristic; adjustable
SPARSE_ROW_THRESHOLD = 10  # states with fewer departures than this are flagged as noisy
TOL = 1e-9

# Real journeys (from the Kaggle file) used for the hand-calculable worked example.
EXAMPLE_JOURNEY_IDS = (
    "10062#1", "10167#1", "10068#1", "10068#2", "10068#3", "10068#4", "21704#1", "21704#2", "21704#3",
)


# --------------------------------------------------------------------------- 1. load / parse
def load_journeys(path=DEFAULT_OUTPUT_PATH) -> pd.DataFrame:
    """Load journeys.csv written by Stage 3 and check it has the columns this stage needs."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python scripts/run_stage3.py` first.")
    df = pd.read_csv(path, dtype={"journey_id": str, "user_id": str, "path": str, "segment": str})
    missing = [c for c in ("journey_id", "path", "segment", "converted") if c not in df.columns]
    if missing:
        raise ValueError(f"journeys file is missing required column(s): {missing}")
    return df


def validate_state_sequence(states: list, journey_id: str = "?") -> None:
    """A valid path is Start, then channels only, then exactly one terminal state."""
    if len(states) < 2:
        raise ValueError(f"journey {journey_id}: path needs at least a Start and a terminal state")
    if any(s == "" for s in states):
        raise ValueError(f"journey {journey_id}: path contains an empty state name")
    if states[0] != START:
        raise ValueError(f"journey {journey_id}: path must begin with {START!r}, got {states[0]!r}")
    if states[-1] not in TERMINAL_STATES:
        raise ValueError(f"journey {journey_id}: path must end in Conversion or Null, got {states[-1]!r}")
    for s in states[1:-1]:
        if s in (START, CONVERSION, NULL):
            raise ValueError(f"journey {journey_id}: {s!r} appears inside the path; it is a model state, not a channel")


def parse_path(path: str, journey_id: str = "?") -> list:
    """'Start > A > B > Conversion' -> ['Start', 'A', 'B', 'Conversion'] (validated)."""
    if not isinstance(path, str) or not path.strip():
        raise ValueError(f"journey {journey_id}: empty path")
    states = path.split(PATH_SEP)
    validate_state_sequence(states, journey_id)
    return states


# --------------------------------------------------------------------------- 2. counts
def count_transitions(paths) -> Counter:
    """Count every consecutive (from_state, to_state) pair across all paths.

    Repeated channels are real transitions: A > A counts as the pair (A, A), a self-loop.
    """
    counts = Counter()
    for states in paths:
        validate_state_sequence(states)
        counts.update(zip(states[:-1], states[1:]))
    return counts


def state_order(counts) -> list:
    """Deterministic order: Start, channels alphabetically, Conversion, Null."""
    seen = {s for pair in counts for s in pair}
    channels = sorted(seen - {START, CONVERSION, NULL})
    return [START] + channels + [CONVERSION, NULL]


def transition_table(counts, states=None) -> pd.DataFrame:
    """Long-format table (from, to, count, from_total, probability) for hand checking."""
    states = states or state_order(counts)
    rank = {s: i for i, s in enumerate(states)}
    totals = Counter()
    for (a, _), n in counts.items():
        totals[a] += n
    rows = [(a, b, n, totals[a], n / totals[a]) for (a, b), n in counts.items()]
    out = pd.DataFrame(rows, columns=["from_state", "to_state", "count", "from_total", "probability"])
    out["_f"] = out["from_state"].map(rank)
    out["_t"] = out["to_state"].map(rank)
    return out.sort_values(["_f", "_t"]).drop(columns=["_f", "_t"]).reset_index(drop=True)


# --------------------------------------------------------------------------- 3. count matrix
def build_count_matrix(counts, states=None) -> pd.DataFrame:
    """Square matrix of raw counts: row = from state, column = to state.

    Terminal rows stay all zero here (counts are observations, and nothing leaves a terminal
    state in real journeys). Absorbing behaviour is added in the PROBABILITY matrix.
    """
    states = states or state_order(counts)
    matrix = pd.DataFrame(0, index=states, columns=states, dtype=int)
    for (a, b), n in counts.items():
        if a not in matrix.index or b not in matrix.columns:
            raise ValueError(f"transition ({a!r} -> {b!r}) uses a state that is not in the state list")
        matrix.at[a, b] = n
    problems = []
    for t in TERMINAL_STATES:
        if matrix.loc[t].sum() > 0:
            problems.append(f"transitions leave terminal state {t!r}")
    if matrix[START].sum() > 0:
        problems.append("transitions enter Start")
    if problems:
        raise ValueError("invalid count matrix: " + "; ".join(problems))
    return matrix


# --------------------------------------------------------------------------- 4/5. probabilities
def matrix_problems(P: pd.DataFrame, tol: float = TOL) -> list:
    """Every way a probability matrix can be wrong, as readable strings (empty list = fine)."""
    problems = []
    if list(P.index) != list(P.columns):
        return ["row and column labels differ"]
    values = P.to_numpy(dtype=float)
    if (values < -tol).any() or (values > 1 + tol).any():
        problems.append("some probabilities are outside [0, 1]")
    for state, total in P.sum(axis=1).items():
        if abs(total - 1.0) > tol:
            problems.append(f"row {state!r} sums to {total:.12f}, not 1")
    for t in TERMINAL_STATES:
        if t in P.index:
            expected = pd.Series(0.0, index=P.columns)
            expected[t] = 1.0
            if not np.allclose(P.loc[t].to_numpy(dtype=float), expected.to_numpy(), atol=tol):
                problems.append(f"terminal state {t!r} is not absorbing")
    if START in P.columns and P[START].abs().sum() > tol:
        problems.append("some state transitions into Start")
    return problems


def validate_probability_matrix(P: pd.DataFrame, tol: float = TOL) -> None:
    problems = matrix_problems(P, tol)
    if problems:
        raise ValueError("invalid probability matrix: " + "; ".join(problems))


def build_probability_matrix(count_matrix: pd.DataFrame) -> pd.DataFrame:
    """Row-normalise each NON-terminal state's counts; make Conversion and Null absorbing.

    P(i -> j) = n(i, j) / sum_k n(i, k). A non-terminal state with zero departures cannot be
    normalised, so we raise instead of silently inventing a distribution.
    """
    P = count_matrix.astype(float)
    totals = count_matrix.sum(axis=1)
    for state in count_matrix.index:
        if state in TERMINAL_STATES:
            continue
        if totals[state] == 0:
            raise ValueError(f"state {state!r} has no outgoing transitions; cannot normalise its row")
        P.loc[state] = count_matrix.loc[state] / totals[state]
    for t in TERMINAL_STATES:
        P.loc[t] = 0.0
        P.loc[t, t] = 1.0
    validate_probability_matrix(P)
    return P


# --------------------------------------------------------------------------- 6. eventual conversion
def absorption_probabilities(P: pd.DataFrame) -> pd.Series:
    """h(i) = probability of eventually being absorbed in Conversion, starting from state i.

    h satisfies h(i) = sum_j P(i, j) h(j), with h(Conversion) = 1 and h(Null) = 0. Splitting P into
    Q (transient -> transient) gives (I - Q) h = r, where r(i) = P(i, Conversion). We solve that
    linear system directly, which sums over all path lengths including loops.
    """
    validate_probability_matrix(P)
    transient = [s for s in P.index if s not in TERMINAL_STATES]
    Q = P.loc[transient, transient].to_numpy(dtype=float)
    r = P.loc[transient, CONVERSION].to_numpy(dtype=float)
    try:
        h = np.linalg.solve(np.eye(len(transient)) - Q, r)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "some states can never reach Conversion or Null (a closed loop), so absorption "
            "probabilities are undefined"
        ) from exc
    if (h < -TOL).any() or (h > 1 + TOL).any():
        raise ValueError("absorption probabilities fell outside [0, 1]; the matrix is not a valid absorbing chain")
    return pd.Series(h, index=transient, name="p_eventual_conversion")


def conversion_probability(P: pd.DataFrame) -> float:
    """Probability of EVENTUALLY reaching Conversion from Start (not just Start -> Conversion)."""
    return float(absorption_probabilities(P)[START])


def conversion_probability_iterative(P: pd.DataFrame, max_steps: int = 100_000, tol: float = 1e-12) -> float:
    """Teaching cross-check: push probability mass forward one step at a time from Start.

    After k steps the vector holds P(state at step k). Mass in Conversion only grows (absorbing).
    We stop when almost no mass remains in non-terminal states.
    """
    validate_probability_matrix(P)
    values = P.to_numpy(dtype=float)
    states = list(P.index)
    v = np.zeros(len(states))
    v[states.index(START)] = 1.0
    transient_idx = [i for i, s in enumerate(states) if s not in TERMINAL_STATES]
    for _ in range(max_steps):
        v = v @ values
        if v[transient_idx].sum() < tol:
            return float(v[states.index(CONVERSION)])
    raise ValueError("probability mass did not settle into terminal states; check for closed loops")


# --------------------------------------------------------------------------- 7. models
@dataclass
class MarkovModel:
    segment: str
    n_journeys: int
    n_converting: int
    counts: pd.DataFrame
    probs: pd.DataFrame
    absorption: pd.Series
    base_conversion_probability: float
    observed_conversion_rate: float
    direct_start_to_conversion: float
    sparse_states: dict = field(default_factory=dict)  # state -> departures, when below threshold

    @property
    def reproduces_observed_rate(self) -> bool:
        return abs(self.base_conversion_probability - self.observed_conversion_rate) < 1e-9


@dataclass
class ModelSet:
    models: dict  # 'all' plus any fitted segments
    skipped: dict  # segment -> reason it was not fitted


def fit_markov_model(journeys: pd.DataFrame, segment=None) -> MarkovModel:
    """Fit one model on all journeys, or on one segment ('new' / 'returning')."""
    if segment is not None and segment not in SEGMENTS:
        raise ValueError(f"segment must be one of {SEGMENTS} or None, got {segment!r}")
    df = journeys if segment is None else journeys[journeys["segment"] == segment]
    if df.empty:
        raise ValueError(f"no journeys to fit for segment {segment or 'all'!r}")
    paths = []
    for jid, path, converted in zip(df["journey_id"], df["path"], df["converted"]):
        states = parse_path(path, jid)
        if (states[-1] == CONVERSION) != (int(converted) == 1):
            raise ValueError(f"journey {jid}: 'converted' flag disagrees with the path's terminal state")
        paths.append(states)
    counts = count_transitions(paths)
    count_matrix = build_count_matrix(counts)
    probs = build_probability_matrix(count_matrix)
    absorption = absorption_probabilities(probs)
    departures = count_matrix.sum(axis=1)
    sparse = {s: int(departures[s]) for s in probs.index
              if s not in TERMINAL_STATES and departures[s] < SPARSE_ROW_THRESHOLD}
    n_conv = int(sum(p[-1] == CONVERSION for p in paths))
    return MarkovModel(
        segment=segment or "all",
        n_journeys=len(paths),
        n_converting=n_conv,
        counts=count_matrix,
        probs=probs,
        absorption=absorption,
        base_conversion_probability=float(absorption[START]),
        observed_conversion_rate=n_conv / len(paths),
        direct_start_to_conversion=float(probs.loc[START, CONVERSION]),
        sparse_states=sparse,
    )


def fit_segment_models(journeys: pd.DataFrame, min_journeys: int = MIN_JOURNEYS_PER_SEGMENT,
                       min_conversions: int = MIN_CONVERSIONS_PER_SEGMENT) -> ModelSet:
    """Fit 'all', then 'new' and 'returning' only when each has enough data.

    A segment that is too small is reported in `skipped` with the reason, never dropped silently.
    """
    unknown = set(journeys["segment"].unique()) - set(SEGMENTS)
    if unknown:
        raise ValueError(f"unknown segment value(s): {sorted(unknown)}")
    models = {"all": fit_markov_model(journeys)}
    skipped = {}
    for seg in SEGMENTS:
        sub = journeys[journeys["segment"] == seg]
        n, conv = len(sub), int(sub["converted"].sum())
        if n < min_journeys:
            skipped[seg] = f"only {n} journeys (minimum {min_journeys})"
        elif conv < min_conversions:
            skipped[seg] = f"only {conv} converting journeys (minimum {min_conversions})"
        else:
            models[seg] = fit_markov_model(journeys, seg)
    return ModelSet(models=models, skipped=skipped)


# --------------------------------------------------------------------------- reporting
def summarize_model(model: MarkovModel) -> str:
    checks = matrix_problems(model.probs)
    lines = [
        f"--- model: {model.segment} ---",
        f"Journeys: {model.n_journeys}  (converting: {model.n_converting})",
        f"Direct P(Start -> Conversion):             {model.direct_start_to_conversion:.4f}",
        f"P(eventually reach Conversion from Start): {model.base_conversion_probability:.4f}",
        f"Observed share of converting journeys:     {model.observed_conversion_rate:.4f}",
        f"Model reproduces observed rate (sanity check): {model.reproduces_observed_rate}",
        f"Probability matrix problems: {'none' if not checks else checks}",
    ]
    if model.sparse_states:
        lines.append(f"Sparse states (< {SPARSE_ROW_THRESHOLD} departures, noisy): {model.sparse_states}")
    return "\n".join(lines)


# =========================================================================== Stage 5: removal effects
DEFAULT_RESULTS_PATH = Path("data/processed/markov_results.csv")
ZERO_BASELINE_TOL = 1e-12
STATUS_OK = "ok"
STATUS_ZERO_BASELINE = "zero_baseline"
STATUS_NOT_IN_JOURNEYS = "not_in_journeys"
STATUS_NOT_REACHABLE = "not_reachable_at_baseline"
RESULT_COLUMNS = [
    "segment", "channel", "baseline_conversion_probability", "conversion_probability_without_channel",
    "removal_effect", "status", "n_visits", "unreachable_states", "cut_off_states",
]


def eligible_channels(P: pd.DataFrame) -> list:
    """Marketing channels in a matrix: every state except Start, Conversion and Null."""
    return [s for s in P.index if s not in (START, CONVERSION, NULL)]


def reachable_states(P: pd.DataFrame, start: str = START) -> set:
    """States reachable from `start` by following transitions with positive probability."""
    values = P.to_numpy(dtype=float)
    labels = list(P.columns)
    row_of = {s: i for i, s in enumerate(P.index)}
    seen, stack = {start}, [start]
    while stack:
        row = values[row_of[stack.pop()]]
        for j, p in enumerate(row):
            if p > 0 and labels[j] not in seen:
                seen.add(labels[j])
                stack.append(labels[j])
    return seen


def remove_channel(P: pd.DataFrame, channel: str) -> pd.DataFrame:
    """Return a NEW valid probability matrix with `channel` removed (redirect-to-Null policy).

    Every transition into the channel (from Start, from other channels, from the channel itself)
    is added to the Null column, then the channel's column is zeroed. The channel's own row is
    replaced by 'go to Null' so it is a valid distribution; since nothing can reach it any more it
    never affects the result. Terminal rows are untouched, so Conversion and Null stay absorbing.
    The input matrix is not modified.
    """
    validate_probability_matrix(P)
    if channel in (START, CONVERSION, NULL):
        raise ValueError(f"{channel!r} is a model state and cannot be removed")
    if channel not in P.index:
        raise ValueError(f"{channel!r} is not a state in this matrix")
    out = P.copy()
    out[NULL] = out[NULL] + out[channel]
    out[channel] = 0.0
    out.loc[channel] = 0.0
    out.loc[channel, NULL] = 1.0
    validate_probability_matrix(out)
    return out


def removal_effect_from_probabilities(cr_base: float, cr_without: float) -> float:
    """RE = 1 - CR_without / CR_base, with the edge cases made explicit.

    * CR_base = 0: the ratio is 0/0, so there is nothing to attribute; return 0.0 (status is
      reported separately as 'zero_baseline').
    * Under redirect-to-Null, CR_without can never exceed CR_base. If it does (beyond rounding
      noise) the matrix is inconsistent and we raise instead of reporting a negative effect.
    """
    if not (np.isfinite(cr_base) and np.isfinite(cr_without)):
        raise ValueError("conversion probabilities must be finite")
    if cr_base <= ZERO_BASELINE_TOL:
        return 0.0
    if cr_without < -TOL or cr_without > cr_base + TOL:
        raise ValueError(
            f"CR_without ({cr_without}) outside [0, CR_base={cr_base}]; removal cannot increase conversion"
        )
    return float(min(max(1.0 - cr_without / cr_base, 0.0), 1.0))


def removal_effects(P: pd.DataFrame, channels=None, counts: pd.DataFrame = None, segment: str = "all") -> pd.DataFrame:
    """Removal effect of each channel, as one tidy table (see RESULT_COLUMNS).

    channels=None uses every channel in the matrix. A requested channel that is not in the matrix
    (never seen in the journeys) gets RE = 0 and status 'not_in_journeys': removing something that
    never occurs changes nothing. Diagnostics per channel:
      unreachable_states : states reachable from Start at baseline but orphaned by the removal
      cut_off_states     : states that could reach Conversion at baseline but can no longer
    """
    validate_probability_matrix(P)
    requested = eligible_channels(P) if channels is None else list(channels)
    if len(set(requested)) != len(requested):
        raise ValueError("duplicate channel names requested")
    reserved = [c for c in requested if c in (START, CONVERSION, NULL)]
    if reserved:
        raise ValueError(f"model states cannot be removed: {reserved}")

    h_base = absorption_probabilities(P)
    cr_base = float(h_base[START])
    reach_base = reachable_states(P)
    rows = []
    for ch in requested:
        if ch not in P.index:
            rows.append(dict(channel=ch, baseline_conversion_probability=cr_base,
                             conversion_probability_without_channel=cr_base, removal_effect=0.0,
                             status=STATUS_NOT_IN_JOURNEYS, n_visits=0, unreachable_states="", cut_off_states=""))
            continue
        P2 = remove_channel(P, ch)
        h2 = absorption_probabilities(P2)
        cr_without = float(h2[START])
        unreachable = sorted((reach_base - reachable_states(P2)) - {ch})
        cut_off = sorted(s for s in h_base.index if s != ch and h_base[s] > TOL and h2[s] <= TOL)
        if cr_base <= ZERO_BASELINE_TOL:
            status = STATUS_ZERO_BASELINE
        elif ch not in reach_base:
            status = STATUS_NOT_REACHABLE
        else:
            status = STATUS_OK
        rows.append(dict(
            channel=ch, baseline_conversion_probability=cr_base,
            conversion_probability_without_channel=cr_without,
            removal_effect=removal_effect_from_probabilities(cr_base, cr_without),
            status=status, n_visits=int(counts[ch].sum()) if counts is not None else pd.NA,
            unreachable_states="; ".join(unreachable), cut_off_states="; ".join(cut_off),
        ))
    table = pd.DataFrame(rows)
    table["n_visits"] = table["n_visits"].astype("Int64")
    table["_key"] = table["removal_effect"].round(12)  # rounding makes tie order identical on every machine
    table = table.sort_values(["_key", "channel"], ascending=[False, True]).drop(columns="_key").reset_index(drop=True)
    table.insert(0, "segment", segment)
    table = table[RESULT_COLUMNS]
    numeric = ["baseline_conversion_probability", "conversion_probability_without_channel", "removal_effect"]
    if not np.isfinite(table[numeric].to_numpy(dtype=float)).all():
        raise ValueError("non-finite value in removal-effect results")
    return table


def model_removal_effects(model: MarkovModel) -> pd.DataFrame:
    """Removal effects for one fitted model (uses its counts for n_visits)."""
    return removal_effects(model.probs, counts=model.counts, segment=model.segment)


def all_removal_effects(model_set: ModelSet) -> pd.DataFrame:
    """One table covering every fitted model: 'all', then 'new' and 'returning' if they were fitted."""
    return pd.concat([model_removal_effects(m) for m in model_set.models.values()], ignore_index=True)


def export_markov_results(table: pd.DataFrame, path=DEFAULT_RESULTS_PATH) -> Path:
    """Write the results CSV (full float precision), creating the folder. Overwrites the old file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return path


def example_probability_matrix(journeys: pd.DataFrame):
    """Probability + count matrices for the 9 real worked-example journeys (same as Stage 4)."""
    ex = journeys[journeys["journey_id"].isin(EXAMPLE_JOURNEY_IDS)]
    missing = set(EXAMPLE_JOURNEY_IDS) - set(ex["journey_id"])
    if missing:
        raise ValueError(f"worked-example journeys not found: {sorted(missing)}")
    paths = [parse_path(p, j) for j, p in zip(ex["journey_id"], ex["path"])]
    counts = build_count_matrix(count_transitions(paths))
    return build_probability_matrix(counts), counts


def main() -> None:
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    journeys = load_journeys(DEFAULT_OUTPUT_PATH)

    # ---- Part A: worked example on 9 real journeys we already traced by hand in Stage 3
    ex = journeys[journeys["journey_id"].isin(EXAMPLE_JOURNEY_IDS)].copy()
    missing = set(EXAMPLE_JOURNEY_IDS) - set(ex["journey_id"])
    if missing:
        raise ValueError(f"worked-example journeys not found in {DEFAULT_OUTPUT_PATH}: {sorted(missing)}")
    ex["_order"] = ex["journey_id"].map({j: i for i, j in enumerate(EXAMPLE_JOURNEY_IDS)})
    ex = ex.sort_values("_order")
    print("=== PART A: WORKED EXAMPLE (9 real journeys from your file) ===")
    for _, r in ex.iterrows():
        print(f"{r['journey_id']:>8}  {r['path']}")
    paths = [parse_path(p, j) for j, p in zip(ex["journey_id"], ex["path"])]
    counts = count_transitions(paths)
    print(f"\nTotal transitions counted: {sum(counts.values())}")
    print("\nTransition table (count / from_total = probability):")
    print(transition_table(counts).round(4).to_string(index=False))
    count_matrix = build_count_matrix(counts)
    print("\nCount matrix (row = from, column = to):")
    print(count_matrix.to_string())
    probs = build_probability_matrix(count_matrix)
    print("\nProbability matrix:")
    print(probs.round(4).to_string())
    print("\nRow sums:", {state: round(float(total), 10) for state, total in probs.sum(axis=1).items()})
    print("\nP(eventually reach Conversion) from each non-terminal state:")
    print(absorption_probabilities(probs).round(4).to_string())
    print(f"\nDirect P(Start -> Conversion):             {probs.loc[START, CONVERSION]:.4f}")
    print(f"P(eventually reach Conversion from Start): {conversion_probability(probs):.4f}")
    print(f"Same, by stepping the chain forward:       {conversion_probability_iterative(probs):.4f}")
    print(f"Observed share of converting journeys:     {sum(p[-1] == CONVERSION for p in paths)}/{len(paths)}")

    # ---- Part B: all journeys, then segments when the sample size permits
    print("\n=== PART B: FULL DATA ===")
    model_set = fit_segment_models(journeys)
    full = model_set.models["all"]
    print("Count matrix:")
    print(full.counts.to_string())
    print("\nProbability matrix:")
    print(full.probs.round(4).to_string())
    print()
    for name, model in model_set.models.items():
        print(summarize_model(model))
        print()
    for seg, reason in model_set.skipped.items():
        print(f"Segment {seg!r} NOT modelled: {reason}")


def main_removal() -> None:
    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 30)
    journeys = load_journeys(DEFAULT_OUTPUT_PATH)

    print("=== PART A: WORKED EXAMPLE (the same 9 real journeys as Stage 4) ===")
    P, counts = example_probability_matrix(journeys)
    print(f"Baseline P(eventually reach Conversion) = {conversion_probability(P):.4f}  (7/9)")
    print("\nRemoving 'Search Ads': Start -> Search Ads (3/9) is redirected to Null.")
    print("Start row BEFORE:", {k: round(float(v), 4) for k, v in P.loc[START].items() if v > 0})
    P_without = remove_channel(P, "Search Ads")
    print("Start row AFTER: ", {k: round(float(v), 4) for k, v in P_without.loc[START].items() if v > 0})
    print("\nMatrix after removing Search Ads (note: every row still sums to 1):")
    print(P_without.round(4).to_string())
    print("Row sums:", {s: round(float(t), 10) for s, t in P_without.sum(axis=1).items()})
    print(f"\nCR_without Search Ads = {conversion_probability(P_without):.4f}  (4/9)")
    print("\nRemoval effects for all five channels in the example:")
    print(removal_effects(P, counts=counts, segment="worked_example").round(4).to_string(index=False))

    print("\n=== PART B: FULL DATA ===")
    model_set = fit_segment_models(journeys)
    table = all_removal_effects(model_set)
    for seg in table["segment"].unique():
        sub = table[table["segment"] == seg]
        print(f"\n--- segment: {seg} ---")
        print(sub.drop(columns="segment").round(4).to_string(index=False))
        print(f"Sum of removal effects: {sub['removal_effect'].sum():.4f}  (not constrained to equal 1)")
    for seg, reason in model_set.skipped.items():
        print(f"\nSegment {seg!r} NOT modelled: {reason}")
    out = export_markov_results(table, DEFAULT_RESULTS_PATH)
    print(f"\nExported {len(table)} rows to {out}")
    print("Status counts:", table["status"].value_counts().to_dict())


if __name__ == "__main__":
    main()
