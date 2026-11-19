"""Fit T_ms = a*f_MHz + c/f_MHz to each ViT module on the existing frequency split."""
import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from fit_vit_eq1 import load_cells, split_cells, write_csv
from fit_vit_eq1_fast import calibrated_scale, fit_relative_calibrated, hit_mask


def fit_coefficients(freq, observed, relative=False):
    # GHz keeps the two design columns on comparable numerical scales.
    ghz = freq / 1000.0
    design = np.column_stack([ghz, 1 / ghz])
    target = observed
    if relative:
        design = design / observed[:, None]
        target = np.ones_like(observed)
    coeff, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
    if rank != 2:
        raise ValueError("Need at least two distinct training frequencies")
    return coeff[0] / 1000, coeff[1] * 1000


def predict(freq, coeff):
    a, c = coeff
    return a * freq + c / freq


def relative_calibrated(freq, observed):
    if len(freq) < 5:
        raise ValueError("Need at least five training frequencies for calibration")
    oof = np.empty_like(observed)
    folds = np.arange(len(freq)) % 3
    for fold in range(3):
        valid = folds == fold
        coeff = fit_coefficients(freq[~valid], observed[~valid], relative=True)
        oof[valid] = predict(freq[valid], coeff)
    # Nonpositive predictions cannot hit a positive observation for any positive scale.
    pos = oof > 0
    scale = calibrated_scale(oof[pos], observed[pos]) if pos.any() else 1.0
    a, c = fit_coefficients(freq, observed, relative=True)
    return a * scale, c * scale


def analyze(path):
    cells, uuid = load_cells(path)
    models, predictions = [], []
    for layer, f, y, test in split_cells(cells):
        train_f, train_y = f[~test], y[~test]
        baseline, _ = fit_relative_calibrated(train_f, train_y)
        base = baseline.predict(f[test])
        columns = {}
        for name, coeff in (("ols", fit_coefficients(train_f, train_y)),
                            ("calibrated", relative_calibrated(train_f, train_y))):
            a, c = coeff
            pred = predict(f[test], coeff)
            columns[name] = pred
            models.append(dict(layer_name=layer, fit=name, a_ms_per_MHz=a, c_ms_MHz=c,
                               train_count=len(train_f), test_count=len(pred),
                               positive_on_sweep=bool(np.all(predict(f[[0, -1]], coeff) > 0)),
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
    return models, predictions, dict(gpu_uuid=uuid, layers=len(cells), held_out=len(obs), metrics=metrics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    args = parser.parse_args()
    try:
        models, predictions, report = analyze(args.csv)
        out = Path(tempfile.mkdtemp(prefix="fit-af-", dir=args.csv.resolve().parent))
        write_csv(out / "models.csv", models)
        write_csv(out / "held_out_predictions.csv", predictions)
        (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Fit failed: {exc}\n")
    print(json.dumps(report, indent=2))
    print(f"Output: {out}")


if __name__ == "__main__":
    main()
