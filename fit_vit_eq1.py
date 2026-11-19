"""Offline GPU-frequency slice of SparseDVFS Eq. (1). Frequencies in MHz, times in ms."""
import argparse
import csv
import json
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Curve:
    intercept: float
    slope: float
    knee: float | None = None

    def predict(self, freq):
        x = 1.0 / freq
        if self.knee is not None:
            x = np.maximum(x, 1.0 / self.knee)
        return self.intercept + self.slope * x


def nonnegative_line(x, y):
    # Solve the two-variable constrained least-squares problem, including both edges.
    xc = x - x.mean()
    slope = np.dot(xc, y - y.mean()) / np.dot(xc, xc)
    intercept = y.mean() - slope * x.mean()
    if slope <= 0:
        return float(y.mean()), 0.0
    if intercept < 0:
        return 0.0, float(np.dot(x, y) / np.dot(x, x))
    return float(intercept), float(slope)


def fit_curves(freq, latency):
    # freq contains distinct values in ascending order. Both one-branch limits
    # are included in the reciprocal fit (a zero slope gives the constant fit).
    x = 1.0 / freq
    reciprocal = Curve(*nonnegative_line(x, latency))
    best = reciprocal
    best_sse = np.sum((best.predict(freq) - latency) ** 2)
    tolerance = np.finfo(float).eps * np.dot(latency, latency) * 16
    for knee in freq[1:-1]:
        feature = np.maximum(x, 1.0 / knee)
        intercept, slope = nonnegative_line(feature, latency)
        sse = np.sum((intercept + slope * feature - latency) ** 2)
        if sse < best_sse - tolerance:
            best = Curve(intercept, slope, float(knee))
            best_sse = sse
    return best, reciprocal


def supported_knee(curve, freq):
    knee = curve.knee
    return (knee is not None and np.sum(freq < knee) >= 3
            and np.sum(freq > knee) >= 3)


def assess_knee(best, freq, latency):
    if not supported_knee(best, freq):
        return "insufficient_two_branch_support", []
    # Repeat on two disjoint halves of training frequencies only.
    knees = []
    for start in (0, 1):
        subset = freq[start::2]
        curve, _ = fit_curves(subset, latency[start::2])
        if not supported_knee(curve, subset):
            return "unstable", knees
        knees.append(curve.knee)
    all_knees = [best.knee, *knees]
    stable = max(all_knees) / min(all_knees) <= 1.10
    return "stable_on_training_splits" if stable else "unstable", knees


def load_cells(path):
    values = defaultdict(lambda: defaultdict(list))
    uuids = set()
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source, skipinitialspace=True)
        reader.fieldnames = [name.strip().lower() for name in (reader.fieldnames or [])]
        required = {"layer_name", "requested_gpu_freq", "latency_ms", "gpu_uuid"}
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"Missing CSV columns: {sorted(missing)}")
        for number, row in enumerate(reader, 2):
            try:
                layer = row["layer_name"].strip()
                uuid = row["gpu_uuid"].strip()
                freq = float(row["requested_gpu_freq"])
                latency = float(row["latency_ms"])
                if not (uuid and layer and np.isfinite([freq, latency]).all()
                        and min(freq, latency) > 0):
                    raise ValueError("expected nonempty identity and positive finite measurements")
            except (ValueError, AttributeError, TypeError) as exc:
                raise ValueError(f"Invalid CSV row {number}: {exc}") from exc
            values[layer][freq].append(latency)
            uuids.add(uuid)
    if len(uuids) != 1:
        raise ValueError(f"Expected one GPU UUID, found {sorted(uuids)}")
    return values, uuids.pop()


def write_csv(path, records):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)


def analyze(path):
    cells, uuid = load_cells(path)
    freq_grid = sorted({f for layer in cells.values() for f in layer})
    held_out = set(freq_grid[2::5])
    rows, predictions = [], []
    total_error = np.zeros(3)
    total_time = 0.0
    for layer, by_freq in cells.items():
        freq = np.array(sorted(by_freq))
        latency = np.array([np.mean(by_freq[f]) for f in freq])
        test = np.array([f in held_out for f in freq])
        if not test.any() or (~test).sum() < 2:
            raise ValueError(f"{layer}: insufficient training or held-out frequencies")
        train_f, train_y = freq[~test], latency[~test]
        best, reciprocal = fit_curves(train_f, train_y)
        support, split_knees = assess_knee(best, train_f, train_y)
        curves = [best, reciprocal, Curve(float(train_y.mean()), 0.0)]
        predicted = np.array([c.predict(freq[test]) for c in curves])
        errors = np.abs(predicted - latency[test]).sum(axis=1)
        wape = 100 * errors / latency[test].sum()
        total_error += errors
        total_time += latency[test].sum()
        rows.append(dict(layer_name=layer, train_count=len(train_f), test_count=int(test.sum()),
                         intercept_ms=best.intercept, C_ms_MHz=best.slope,
                         M_ms=best.slope / best.knee if best.knee is not None else None,
                         knee_MHz=best.knee, knee_support=support, split_knees_MHz=json.dumps(split_knees),
                         eq1_WAPE_pct=wape[0], reciprocal_WAPE_pct=wape[1], constant_WAPE_pct=wape[2],
                         eq1_MAE_ms=errors[0]/test.sum()))
        for f, observed, (eq1, rec, const) in zip(freq[test], latency[test], predicted.T):
            predictions.append(dict(layer_name=layer, requested_gpu_freq=f, measured_ms=observed,
                                    eq1_ms=eq1, reciprocal_ms=rec, constant_ms=const))
    metrics = 100 * total_error / total_time
    report = dict(gpu_uuid=uuid, layers=len(rows), frequencies=len(freq_grid),
                  cells=sum(map(len, cells.values())), held_out_cells=len(predictions),
                  eq1_WAPE_pct=float(metrics[0]), reciprocal_WAPE_pct=float(metrics[1]),
                  constant_WAPE_pct=float(metrics[2]),
                  stable_knees=sum(r["knee_support"] == "stable_on_training_splits" for r in rows),
                  stable_and_lower_test_error=sum(
                      1 for r in rows if r["knee_support"] == "stable_on_training_splits"
                      and r["eq1_WAPE_pct"] < min(r["reciprocal_WAPE_pct"], r["constant_WAPE_pct"])))
    return rows, predictions, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="profiling_results_ViT-Base.csv")
    args = parser.parse_args()
    try:
        rows, predictions, report = analyze(args.csv)
        out = Path(tempfile.mkdtemp(prefix="fit-eq1-", dir=args.csv.resolve().parent))
        write_csv(out / "models.csv", rows)
        write_csv(out / "held_out_predictions.csv", predictions)
        (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Fit failed: {exc}\n")
    print(json.dumps(report, indent=2))
    print(f"Output: {out}")


if __name__ == "__main__":
    main()
