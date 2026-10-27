"""AttributionLab Flask backend.

Thin integration layer only: every attribution/budget number comes from the existing
src/ modules (data_pipeline, journey_builder, markov_engine, shapley_engine, hybrid_engine,
budget_allocator). This file does not reimplement any algorithm - it loads an uploaded CSV,
calls the real pipeline functions in order, and serializes their outputs to JSON for the
single-page frontend in templates/index.html.

Pipeline wiring (see each module's own docstring for the algorithm):
  clean_events -> build_journeys -> fit_markov_model -> model_removal_effects (Markov)
                                   -> extract_channels_from_journeys -> generate_coalitions
                                      -> calculate_all_coalition_values -> compute_shapley_values
                                      -> integrate_shapley_frames (Shapley)
                                   -> hybrid_attribution (Markov + Shapley -> Hybrid)
                                   -> allocate_budget (Hybrid weights + user budget -> INR split)

Session state: an uploaded dataset is validated once and its reconstructed journeys are
cached in memory under a token; "Run Analysis" uses that cache; "Allocate Budget" reuses the
cached hybrid weights so changing only the budget never reruns the attribution pipeline.
"""
from __future__ import annotations

import os
import sys
import tempfile
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd
from flask import Flask, jsonify, render_template, request

from mta.data_pipeline import KAGGLE_CONFIG, SchemaError, clean_events
from mta.journey_builder import build_journeys
from mta.markov_engine import MarkovModel, fit_segment_models, model_removal_effects
from shapley_engine.coalition_generator import (
    CoalitionLimitError,
    coalitions_to_frame,
    extract_channels_from_journeys,
    generate_coalitions,
    validate_coalitions,
)
from shapley_engine.coalition_value import calculate_all_coalition_values, validate_coalition_values
from shapley_engine.shapley_engine import (
    build_value_lookup,
    compute_shapley_values,
    shapley_to_frame,
    validate_shapley_values,
)
from shapley_engine.shapley_integration import IntegrationBlockedError, integrate_shapley_frames
from hybrid_engine import hybrid_attribution
from budget_allocator import ALLOCATION_NOTICE, allocate_budget, allocation_summary
from baseline_attribution import all_heuristic_attributions

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB, generous for a classroom CSV

# In-memory per-upload cache. A demo-scale single-process app; no DB needed.
ANALYSES: dict = {}

REQUIRED_RAW_COLUMNS = list(KAGGLE_CONFIG.column_map.keys())  # as stated in the raw file


# --------------------------------------------------------------------------- helpers
def _error(message: str, status: int = 400):
    return jsonify({"error": message}), status


def _dataframe_preview(path: Path, n: int = 5) -> list:
    preview = pd.read_csv(path, dtype=str, keep_default_na=False, nrows=n)
    return preview.to_dict(orient="records"), list(preview.columns)


def run_attribution_pipeline(journeys: pd.DataFrame, model: MarkovModel) -> dict:
    """Markov -> Shapley -> Hybrid for one already-fitted segment model.

    `model` comes from fit_segment_models() and carries its own `.segment` label
    ("all", "new" or "returning"); the Shapley game is built from exactly the
    journeys that segment was fitted on, so the channel sets and baselines agree
    (required by hybrid_attribution's consistency check).
    """
    markov_results = model_removal_effects(model)

    seg_journeys = journeys if model.segment == "all" else journeys[journeys["segment"] == model.segment]
    channels = extract_channels_from_journeys(seg_journeys)
    if not channels:
        raise ValueError(f"No marketing channels were found in the '{model.segment}' journeys.")

    coalitions = generate_coalitions(channels)
    validate_coalitions(coalitions, channels)
    coalition_table = coalitions_to_frame(coalitions)

    coalition_values = calculate_all_coalition_values(model.probs, coalition_table)
    validate_coalition_values(coalition_values, coalition_table, channels, model.base_conversion_probability)

    lookup = build_value_lookup(coalition_values)
    shapley = compute_shapley_values(lookup)
    validate_shapley_values(shapley, lookup)
    shapley_df = shapley_to_frame(shapley)

    integration = integrate_shapley_frames(
        coalition_values, shapley_df, baseline=model.base_conversion_probability
    )
    final_shapley = integration.table

    hybrid_df = hybrid_attribution(markov_results, final_shapley, alpha=0.5, segment=model.segment)
    heuristics = all_heuristic_attributions(seg_journeys)

    return {
        "markov_results": markov_results,
        "final_shapley": final_shapley,
        "hybrid": hybrid_df,
        "heuristics": heuristics,
        "baseline": model.base_conversion_probability,
        "channels": channels,
    }


SEGMENT_LABELS = {"all": "All Users", "new": "New Users", "returning": "Returning Users"}


# --------------------------------------------------------------------------- routes
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/upload", methods=["POST"])
def upload():
    file = request.files.get("file")
    if file is None or file.filename == "":
        return _error("No file was selected.")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            file.save(tmp.name)
            tmp_path = Path(tmp.name)

        try:
            preview_rows, columns = _dataframe_preview(tmp_path)
        except Exception as exc:  # pandas parser errors, encoding errors, etc.
            return _error(f"Could not read the file as CSV: {exc}")

        missing = [c for c in REQUIRED_RAW_COLUMNS if c not in columns and c != "Campaign"]
        if missing:
            return _error(
                f"Missing required column(s): {missing}. Expected columns: {REQUIRED_RAW_COLUMNS}."
            )

        try:
            validated = clean_events(tmp_path, KAGGLE_CONFIG)
        except SchemaError as exc:
            return _error(str(exc))
        except Exception as exc:
            return _error(f"Could not validate the file: {exc}")

        if validated.summary.rows_clean == 0:
            return _error(
                "Every row in this file was rejected during validation, so no journeys can be "
                "built. Check the column formats (date format DD-MM-YYYY HH:MM, Conversion "
                "Yes/No)."
            )

        try:
            journey_result = build_journeys(validated.clean, window_hours=None)
        except Exception as exc:
            return _error(f"Could not reconstruct journeys: {exc}")

        token = uuid.uuid4().hex
        ANALYSES[token] = {"journeys": journey_result.journeys, "hybrid_by_segment": {}}

        reject_reasons = [
            f"{name}: {count}" for name, count in validated.summary.check_counts.items() if count
        ]

        return jsonify(
            {
                "token": token,
                "columns": columns,
                "preview": preview_rows,
                "stats": {
                    "n_events": validated.summary.rows_clean,
                    "n_rejected": validated.summary.rows_rejected,
                    "n_users": validated.summary.n_users,
                    "n_journeys": len(journey_result.journeys),
                },
                "warnings": reject_reasons,
            }
        )
    finally:
        if tmp_path is not None:
            try:
                os.remove(tmp_path)
            except OSError:
                pass


@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.get_json(silent=True) or {}
    token = data.get("token")
    entry = ANALYSES.get(token)
    if entry is None:
        return _error("Unknown or expired session. Please upload the dataset again.")

    journeys = entry["journeys"]
    try:
        model_set = fit_segment_models(journeys)
    except ValueError as exc:
        return _error(f"Attribution analysis failed: {exc}")

    segments_response = {}
    hybrid_by_segment = {}

    for seg_name, model in model_set.models.items():
        try:
            result = run_attribution_pipeline(journeys, model)
        except (ValueError, IntegrationBlockedError) as exc:
            segments_response[seg_name] = {
                "label": SEGMENT_LABELS.get(seg_name, seg_name),
                "available": False,
                "reason": f"Could not compute attribution for this segment: {exc}",
            }
            continue

        hybrid_by_segment[seg_name] = result["hybrid"]
        first_pct = dict(zip(result["heuristics"]["first"]["channel"], result["heuristics"]["first"]["percentage"]))
        last_pct = dict(zip(result["heuristics"]["last"]["channel"], result["heuristics"]["last"]["percentage"]))
        linear_pct = dict(zip(result["heuristics"]["linear"]["channel"], result["heuristics"]["linear"]["percentage"]))
        rows = [
            {
                "channel": r["channel"],
                "first_touch_pct": round(first_pct.get(r["channel"], 0.0), 2),
                "last_touch_pct": round(last_pct.get(r["channel"], 0.0), 2),
                "linear_pct": round(linear_pct.get(r["channel"], 0.0), 2),
                "markov_removal_effect": round(float(r["markov_removal_effect"]), 6),
                "markov_weight_pct": round(float(r["markov_weight"]) * 100, 2),
                "shapley_value": round(float(r["shapley_raw_value"]), 6),
                "shapley_weight_pct": round(float(r["shapley_weight"]) * 100, 2),
                "hybrid_weight_pct": round(float(r["hybrid_percentage"]), 2),
            }
            for _, r in result["hybrid"].iterrows()
        ]
        segments_response[seg_name] = {
            "label": SEGMENT_LABELS.get(seg_name, seg_name),
            "available": True,
            "n_journeys": model.n_journeys,
            "n_converting": model.n_converting,
            "baseline_conversion_probability": round(result["baseline"], 6),
            "n_channels": len(result["channels"]),
            "rows": rows,
        }

    for seg_name, reason in model_set.skipped.items():
        segments_response[seg_name] = {
            "label": SEGMENT_LABELS.get(seg_name, seg_name),
            "available": False,
            "reason": f"Not enough data for this segment ({reason}).",
        }

    entry["hybrid_by_segment"] = hybrid_by_segment

    return jsonify({"segments": segments_response})


@app.route("/api/allocate", methods=["POST"])
def allocate():
    data = request.get_json(silent=True) or {}
    token = data.get("token")
    budget = data.get("budget")
    segment = data.get("segment", "all")

    entry = ANALYSES.get(token)
    if entry is None:
        return _error("Unknown or expired session. Please upload the dataset again.")
    hybrid_by_segment = entry.get("hybrid_by_segment") or {}
    if segment not in hybrid_by_segment:
        return _error(
            f"Attribution weights for the '{SEGMENT_LABELS.get(segment, segment)}' segment are not "
            "available. Run the attribution analysis first, or pick a different segment."
        )
    if budget is None:
        return _error("Enter a budget amount.")

    weights = hybrid_by_segment[segment][["channel", "hybrid_weight"]]
    try:
        table = allocate_budget(weights, budget)
        summary = allocation_summary(table, budget)
    except (TypeError, ValueError) as exc:
        return _error(str(exc))

    rows = [
        {
            "channel": r["channel"],
            "attribution_weight_pct": round(float(r["hybrid_percentage"]), 2),
            "amount": int(r["recommended_budget"]),
        }
        for _, r in table.iterrows()
    ]

    return jsonify(
        {
            "rows": rows,
            "summary": {
                "total_budget": summary["total_budget"],
                "allocated_total": summary["allocated_total"],
                "difference": summary["difference"],
                "reconciles": summary["reconciles"],
            },
            "notice": list(ALLOCATION_NOTICE),
        }
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
