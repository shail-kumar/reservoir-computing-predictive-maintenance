"""Maintenance-alert worked example for WRITEUP.md Table 3.

Run compare_models first to produce results_FD004_nlags-19.pkl.

    python -m rcpm.experiments.maintenance_alert_demo
"""

import pickle

from rcpm.data import load_cmapss
from rcpm.paths import result_path

# Illustrative only, matching Ozcan (2025) Table 17's convention - not derived from cost modeling.
THRESHOLD = 15

# Hand-picked for illustration, not random - see WRITEUP.md Table 3.
EXAMPLE_INDICES = [99, 124, 182, 7]


def alert(value: float) -> str:
    return "Maintenance Required" if value <= THRESHOLD else "No Maintenance Required"


def main():
    with open(result_path("results_FD004_nlags-19.pkl"), "rb") as f:
        results = pickle.load(f)
    by_name = {r["name"]: r for r in results}

    # Test units are 1..248 with no gaps, and load_cmapss()/build_sequences() preserve
    # ascending unit order, same as RUL_FD004.txt's row order (verified directly against
    # the raw file) - so test-sequence index i is simply engine unit i + 1.
    _, test, _ = load_cmapss("FD004")
    units = sorted(test["unit"].unique())
    assert units == list(range(1, len(units) + 1)), (
        "Unit-ID mapping assumes units are 1..N with no gaps - re-derive if this fails"
    )

    print("| Test engine | Actual RUL | Baseline | ESN | NG-RC |")
    print("|---|---|---|---|---|")
    for i in EXAMPLE_INDICES:
        y_true = by_name["Baseline (GBM)"]["y_test"][i]
        true_alert = alert(y_true)
        cells = [f"Unit {i + 1}", f"{y_true}"]
        for name in ["Baseline (GBM)", "Classical ESN", "NG-RC"]:
            r = by_name[name]
            pred = r["y_pred"][i]
            margin = r.get("margin")
            decision_value = pred if margin is None else pred - margin
            model_alert = alert(decision_value)
            mark = "✓" if model_alert == true_alert else "✗"
            # To make the alert based on conformal threshold.
            shown = (
                f"{pred:.1f}" if margin is None else f"{pred:.1f} -> {decision_value:.1f}"
            )
            cells.append(f"{model_alert} ({shown}) {mark}")
        print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
