#!/usr/bin/env python3
"""Protocol II (ML anomaly detection) - plaintext model and public inputs.

Model: Prophet as configured by Jaiswal et al. (AICCC 2021) in their published notebook
(github.com/ritukajaiswal/Anomaly-Detection-Code-and-Anomaly-Labelled-Dataset-on-Ausgrid-Data):
  piecewise-linear trend (25 changepoints in the first 80% of history), daily seasonality
  (Fourier order 4), yearly seasonality (order 10), weekly seasonality conditioned on
  weekend / weekday (order 3 each), Australian holidays, 99% interval.
  Anomaly: reading outside [yhat_lower, yhat_upper], or reading == 0.

With Prophet's Gaussian priors the MAP fit is linear in y:  w = A y,  A = (P'P + L)^-1 P'.
The design matrix P and A depend only on timestamps (public), so each broker can apply A to
its share of y. (Prophet's Laplace changepoint prior is replaced by a Gaussian with the same
variance, and hyperparameters are Prophet's defaults since the tuned values were not published.)
In-sample, Prophet's interval is yhat +/- z * sigma_obs with z = 2.576 for 99%.

Writes iot/data/p2_c<ID>/: readings.i32, design.f64, A.f64, meta.txt, ref_anomaly.u8, paper_label.u8
"""
import pathlib, sys
import numpy as np, pandas as pd, holidays

ROOT = pathlib.Path(__file__).resolve().parent
CSV = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] else pathlib.Path(
    "/home/claude/datasets/Anomaly-Detection-Code-and-Anomaly-Labelled-Dataset-on-Ausgrid-Data/Anomaly labelled dataset.csv")
CUSTOMERS = [int(c) for c in sys.argv[2].split(",")] if len(sys.argv) > 2 else [7]
Z = 2.5758293035489004          # two-sided 99%
N_CP, CP_RANGE = 25, 0.8         # Prophet defaults
SIGMA_OBS = 0.25                 # kWh, public noise scale used for the prior weights
TAU = dict(trend=5.0, cp=0.05 * np.sqrt(2), seas=10.0, hol=10.0)   # Prophet default prior scales

def fourier(tdays, period, order):
    cols = []
    for k in range(1, order + 1):
        a = 2 * np.pi * k * tdays / period
        cols += [np.sin(a), np.cos(a)]
    return np.column_stack(cols)

def design(ds):
    n = len(ds)
    t = (ds - ds.iloc[0]) / (ds.iloc[-1] - ds.iloc[0])
    t = t.to_numpy(float)
    tdays = (ds - pd.Timestamp("1970-01-01")).dt.total_seconds().to_numpy() / 86400.0
    hist = int(np.floor(n * CP_RANGE))
    cp_idx = np.linspace(0, hist - 1, N_CP + 1).round().astype(int)[1:]
    s = t[cp_idx]
    trend = np.column_stack([np.ones(n), t] + [np.maximum(t - sj, 0) for sj in s])
    taus = [TAU["trend"], TAU["trend"]] + [TAU["cp"]] * N_CP
    wkend = (ds.dt.weekday >= 5).to_numpy()
    blocks = [trend, fourier(tdays, 1.0, 4), fourier(tdays, 365.25, 10),
              fourier(tdays, 7.0, 3) * wkend[:, None], fourier(tdays, 7.0, 3) * (~wkend)[:, None]]
    taus += [TAU["seas"]] * (8 + 20 + 6 + 6)
    au = holidays.Australia(years=sorted(set(ds.dt.year)))
    names = sorted(set(au.values()))
    dates = ds.dt.date
    hol = np.column_stack([dates.map(lambda d, nm=nm: float(au.get(d) == nm)).to_numpy() for nm in names])
    keep = hol.sum(0) > 0
    blocks.append(hol[:, keep]); taus += [TAU["hol"]] * int(keep.sum())
    P = np.column_stack(blocks)
    return P, np.array(taus)

def main():
    df = pd.read_csv(CSV)
    for cid in CUSTOMERS:
        d = df[df.costumer == cid].copy()
        d["ds"] = pd.to_datetime(dict(year=d.year, month=d.monthofyear, day=d.dayofmonth, hour=d.hourofday, minute=d.minuteofhour))
        d = d.sort_values("ds").reset_index(drop=True)
        y = np.rint(d.GC.to_numpy() * 1000).astype(np.int32)        # Wh, as published by the meter
        P, taus = design(d.ds)
        lam = (SIGMA_OBS / taus) ** 2
        A = np.linalg.solve(P.T @ P + np.diag(lam), P.T)           # K x N, public
        w = A @ y
        r = y - P @ w
        s2 = np.mean(r ** 2)
        ref = ((r ** 2 > Z * Z * s2) | (y == 0)).astype(np.uint8)
        lab = d.Anomaly.to_numpy().astype(np.uint8)
        out = ROOT / "data" / f"p2_c{cid}"; out.mkdir(parents=True, exist_ok=True)
        y.tofile(out / "readings.i32"); P.astype("<f8").tofile(out / "design.f64"); A.astype("<f8").tofile(out / "A.f64")
        ref.tofile(out / "ref_anomaly.u8"); lab.tofile(out / "paper_label.u8")
        (out / "meta.txt").write_text(f"customer {cid}\nN {len(y)}\nK {P.shape[1]}\nz {Z:.16f}\nstart {d.ds.iloc[0]}\nend {d.ds.iloc[-1]}\n")
        tp = int(((ref == 1) & (lab == 1)).sum()); fp = int(((ref == 1) & (lab == 0)).sum()); fn = int(((ref == 0) & (lab == 1)).sum())
        print(f"customer {cid}: N={len(y)} K={P.shape[1]} sigma={np.sqrt(s2):.1f} Wh  anomalies ours={ref.sum()} paper={lab.sum()}  "
              f"agreement={(ref == lab).mean():.4f} precision={tp / max(tp + fp, 1):.3f} recall={tp / max(tp + fn, 1):.3f}")

if __name__ == "__main__":
    main()
