"""Fit a*f + c/f or a*f + b + c/f, with MHz inputs and ms outputs."""
import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from fit_vit_eq1 import load_cells, split_cells, write_csv
from fit_vit_eq1_fast import calibrated_scale, fit_relative_calibrated, hit_mask


def fit_coefficients(freq, observed, relative=False, intercept=False):
    # GHz keeps the design columns on comparable numerical scales.
    ghz = freq / 1000.0
    columns = [ghz, np.ones_like(ghz), 1 / ghz] if intercept else [ghz, 1 / ghz]
    design = np.column_stack(columns)
    target = observed
    if relative:
        design = design / observed[:, None]
        target = np.ones_like(observed)
    coeff, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
    if rank != len(columns):
        raise ValueError(f"Need {len(columns)} independent design columns at distinct frequencies")
    if intercept:
        return coeff[0] / 1000, coeff[1], coeff[2] * 1000
    return coeff[0] / 1000, coeff[1] * 1000


def predict(freq, coeff):
    b = coeff[1] if len(coeff) == 3 else 0.0
    return coeff[0] * freq + b + coeff[-1] / freq


def positive_on_sweep(low, high, coeff):
    a, c = coeff[0], coeff[-1]
    points = [low, high]
    if a > 0 and c > 0:
        minimum = np.sqrt(c / a)
        if low < minimum < high:
            points.append(minimum)
    return bool(np.all(predict(np.array(points), coeff) > 0))


def relative_calibrated(freq, observed, intercept=False):
    if len(freq) < 5:
        raise ValueError("Need at least five training frequencies for calibration")
    oof = np.empty_like(observed)
    folds = np.arange(len(freq)) % 3
    for fold in range(3):
        valid = folds == fold
        coeff = fit_coefficients(freq[~valid], observed[~valid], relative=True, intercept=intercept)
        oof[valid] = predict(freq[valid], coeff)
    # Nonpositive predictions cannot hit a positive observation for any positive scale.
    pos = oof > 0
    scale = calibrated_scale(oof[pos], observed[pos]) if pos.any() else 1.0
    coeff = fit_coefficients(freq, observed, relative=True, intercept=intercept)
    return tuple(value * scale for value in coeff)


def analyze(path, intercept=False):
    cells, uuid = load_cells(path)
    models, predictions = [], []
    for layer, f, y, test in split_cells(cells):
        train_f, train_y = f[~test], y[~test]
        baseline, _ = fit_relative_calibrated(train_f, train_y)
        base = baseline.predict(f[test])
        columns = {}
        for name, coeff in (("ols", fit_coefficients(train_f, train_y, intercept=intercept)),
                            ("calibrated", relative_calibrated(train_f, train_y, intercept=intercept))):
            pred = predict(f[test], coeff)
            columns[name] = pred
            models.append(dict(layer_name=layer, fit=name, a_ms_per_MHz=coeff[0],
                               **({"b_ms": coeff[1]} if intercept else {}), c_ms_MHz=coeff[-1],
                               train_count=len(train_f), test_count=len(pred),
                               positive_on_sweep=positive_on_sweep(f[0], f[-1], coeff),
                               hit_pct=100 * hit_mask(pred, y[test]).mean(),
                               WAPE_pct=100 * np.abs(pred-y[test]).sum() / y[test].sum()))
        for freq, obs, ref, ols, cal in zip(f[test], y[test], base, columns["ols"], columns["calibrated"]):
            predictions.append(dict(layer_name=layer, requested_gpu_freq=freq, measured_ms=obs,
                                    eq1_baseline_ms=ref, ols_ms=ols, calibrated_ms=cal))
    obs = np.array([r["measured_ms"] for r in predictions])
    metrics = {}
    for key in ("eq1_baseline_ms", "ols_ms", "calibrated_ms"):
        pred = np.array([r[key] for r in predictions])
        hits = int(hit_mask(pred, obs).sum())
        metrics[key] = dict(hits=hits, hit_pct=100 * hits / len(obs),
                            WAPE_pct=float(100*np.abs(pred-obs).sum()/obs.sum()),
                            nonpositive_predictions=int((pred <= 0).sum()))
    report = dict(gpu_uuid=uuid, layers=len(cells), held_out=len(obs), metrics=metrics)
    if intercept:
        reference, candidate = metrics["eq1_baseline_ms"], metrics["calibrated_ms"]
        report.update(formula="a*f_MHz + b + c/f_MHz",
                      net_hits=candidate["hits"] - reference["hits"],
                      hit_change_pp=candidate["hit_pct"] - reference["hit_pct"],
                      WAPE_change_pp=candidate["WAPE_pct"] - reference["WAPE_pct"])
    return models, predictions, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--intercept", action="store_true", help="fit an independent constant b in ms")
    args = parser.parse_args()
    try:
        models, predictions, report = analyze(args.csv, intercept=args.intercept)
        prefix = "fit-af-b-" if args.intercept else "fit-af-"
        out = Path(tempfile.mkdtemp(prefix=prefix, dir=args.csv.resolve().parent))
        write_csv(out / "models.csv", models)
        write_csv(out / "held_out_predictions.csv", predictions)
        (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Fit failed: {exc}\n")
    print(json.dumps(report, indent=2))
    print(f"Output: {out}")


if __name__ == "__main__":
    main()
