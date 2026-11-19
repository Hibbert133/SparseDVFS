"""Compare two fits of the GPU-frequency slice of Eq. (1) on the original split."""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

from fit_vit_eq1 import Curve, fit_curves, load_cells, split_cells, write_csv


def hit_mask(predicted, observed):
    return (predicted >= 0.9 * observed) & (predicted <= 1.1 * observed)


def calibrated_scale(predicted, observed):
    # A point's hit status can change only at the ends of its scale interval.
    bounds = np.unique(np.r_[0.9 * observed / predicted, 1.1 * observed / predicted])
    candidates = np.r_[1.0, (bounds[1:] + bounds[:-1]) / 2, bounds]
    # Count closed intervals in O(n log n). Include 1 and prefer the least change on ties.
    left = np.sort(0.9 * observed / predicted)
    right = np.sort(1.1 * observed / predicted)
    counts = (np.searchsorted(left, candidates, side="right")
              - np.searchsorted(right, candidates, side="left"))
    best = candidates[counts == counts.max()]
    return float(best[np.argmin(np.abs(np.log(best)))])


def fit_relative_calibrated(freq, observed):
    if len(freq) < 5:
        raise ValueError("At least five distinct training frequencies are required for calibration")
    folds = np.arange(len(freq)) % 3
    oof = np.empty_like(observed, dtype=float)
    for fold in range(3):
        valid = folds == fold
        curve, _ = fit_curves(freq[~valid], observed[~valid], relative=True)
        oof[valid] = curve.predict(freq[valid])
    scale = calibrated_scale(oof, observed)
    curve, _ = fit_curves(freq, observed, relative=True)
    return Curve(scale * curve.intercept, scale * curve.slope, curve.knee), scale


def search_shape(freq, observed, baseline):
    # At the lowest frequency, u is the fraction of predicted time in C/f.
    # h * ((1-u) + u * max(f_min/f, f_min/knee)) is exactly an Eq. (1) slice.
    # Include both the constant and the no-knee limit as well as the existing fit.
    def score(curve):
        predicted = curve.predict(freq)
        return int(hit_mask(predicted, observed).sum()), -float(np.abs(predicted - observed).sum())

    best, best_score = baseline, score(baseline)
    indices = np.unique(np.linspace(1, len(freq)-2, min(15, len(freq)-2), dtype=int))
    for knee in [None, *freq[indices]]:
        effective = freq if knee is None else np.minimum(freq, knee)
        for u in np.linspace(0, 1, 21):
            if u == 0 and knee is not None:
                continue
            shape = (1-u) + u * freq[0] / effective
            initial = np.median(observed / shape)
            height = initial * calibrated_scale(initial * shape, observed)
            curve = Curve(float(height * (1-u)), float(height * u * freq[0]), knee)
            quality = score(curve)
            if quality > best_score:
                best, best_score = curve, quality
    return best


def select_shape(freq, observed, baseline):
    if len(freq) < 8:
        raise ValueError("Shape selection requires eight or more training frequencies")
    folds = np.arange(len(freq)) % 3
    old = np.empty_like(observed, dtype=float)
    new = np.empty_like(old)
    for fold in range(3):
        valid = folds == fold
        ff, yy = freq[~valid], observed[~valid]
        inner_base, _ = fit_relative_calibrated(ff, yy)
        candidate = search_shape(ff, yy, inner_base)
        old[valid], new[valid] = inner_base.predict(freq[valid]), candidate.predict(freq[valid])
    gain = int(hit_mask(new, observed).sum()) - int(hit_mask(old, observed).sum())
    error_change = np.abs(new-observed).sum() - np.abs(old-observed).sum()
    use_search = gain > 0 and error_change <= 0
    selected = search_shape(freq, observed, baseline) if use_search else baseline
    details = dict(shape_changed=bool(selected != baseline), inner_hit_gain=gain,
                   inner_change_WAPE_pp=float(100 * error_change / observed.sum()))
    return selected, details


def compare(path, refine_shape=False):
    cells, uuid = load_cells(path)
    models, predictions = [], []
    for layer, freq, latency, test in split_cells(cells):
        fitted, scale = fit_relative_calibrated(freq[~test], latency[~test])
        details = {}
        if refine_shape:
            baseline = fitted
            fitted, details = select_shape(freq[~test], latency[~test], baseline)
            if details["shape_changed"]:
                scale = None  # Export final coefficients directly.
        else:
            baseline, _ = fit_curves(freq[~test], latency[~test])
        old, new = baseline.predict(freq[test]), fitted.predict(freq[test])
        models.append(dict(layer_name=layer, intercept_ms=fitted.intercept, C_ms_MHz=fitted.slope,
                           M_ms=fitted.slope/fitted.knee if fitted.knee is not None else None,
                           knee_MHz=fitted.knee, calibration_scale=scale,
                           train_count=int((~test).sum()), test_count=int(test.sum()), **details))
        for f, y, a, b in zip(freq[test], latency[test], old, new):
            predictions.append(dict(layer_name=layer, requested_gpu_freq=f, measured_ms=y,
                                    baseline_ms=a, fitted_ms=b))
    observed, old, new = (np.array([row[key] for row in predictions])
                          for key in ("measured_ms", "baseline_ms", "fitted_ms"))
    old_hits, new_hits = hit_mask(old, observed), hit_mask(new, observed)
    n = len(observed)
    old_count, new_count = int(old_hits.sum()), int(new_hits.sum())
    old_wape, new_wape = [float(100*np.abs(pred-observed).sum()/observed.sum()) for pred in (old, new)]
    report = dict(gpu_uuid=uuid, modules=len(models), held_out=n,
                  old_hits=old_count, new_hits=new_count, gained=int((new_hits & ~old_hits).sum()),
                  lost=int((old_hits & ~new_hits).sum()), old_hit_pct=100*old_count/n,
                  new_hit_pct=100*new_count/n, change_pp=100*(new_count-old_count)/n,
                  old_WAPE_pct=old_wape, new_WAPE_pct=new_wape)
    if refine_shape:
        report.update(baseline="relative_calibrated", candidate="training_cv_selected_shape",
                      shape_changed_modules=sum(row["shape_changed"] for row in models))
    return models, predictions, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--refine-shape", action="store_true",
                        help="compare the calibrated fit against training-CV-selected Eq. (1) shapes")
    args = parser.parse_args()
    if sys.platform == "win32" and not sys.stdout.isatty():
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        models, predictions, r = compare(args.csv, args.refine_shape)
        prefix = "fit-eq1-shape-" if args.refine_shape else "fit-eq1-fast-"
        out = Path(tempfile.mkdtemp(prefix=prefix, dir=args.csv.resolve().parent))
        write_csv(out / "models.csv", models)
        write_csv(out / "held_out_predictions.csv", predictions)
        (out / "summary.json").write_text(json.dumps(r, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Fit failed: {exc}\n")
    print(f"留出点 {r['held_out']}，模块 {r['modules']}。")
    if args.refine_shape:
        print(f"基线：相对误差拟合加校准；训练集内验证后换用新曲线的模块数：{r['shape_changed_modules']}。")
    print(f"±10% 命中率：{r['old_hit_pct']:.2f}% → {r['new_hit_pct']:.2f}%"
          f"（{r['change_pp']:+.2f} 个百分点），新增 {r['gained']} 个，丢失 {r['lost']} 个。")
    print(f"WAPE：{r['old_WAPE_pct']:.2f}% → {r['new_WAPE_pct']:.2f}%。")
    if r['new_hits'] > r['old_hits']:
        print("结论：本次留出点按论文命中率，新拟合更好。"
              + ("代价是整体 WAPE 上升。" if r['new_WAPE_pct'] > r['old_WAPE_pct'] else "整体 WAPE 也未上升。"))
    else:
        print("结论：本次留出点的论文命中率未提高，保留原拟合。")
    print(f"Output: {out}")


if __name__ == "__main__":
    main()
