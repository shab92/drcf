#!/usr/bin/env python3
"""DRCF prototype: try privacy-preserving forecasting / anomaly detection on your own time series.

The tool plays the model owner (trains in the clear on the training window), then runs the real two-broker
DRCF computation (build/drcf_secure, both brokers on this machine) on secret-shared data, and reports
accuracy, how closely the secure result matches the plaintext model, and the cost.

Examples
  # load forecasting with an AR model on lags 1, 2, 24 plus hour-of-day and weekday effects
  python3 drcf.py --data load.csv --time timestamp --value kwh --task forecast --model linear \
                  --lags 1,2,24 --public intercept,hour,weekday

  # anomaly detection with the Prophet-style model (Table A1, Protocol II), labels for scoring
  python3 drcf.py --data meter.csv --time ts --value kwh --label anomaly --task anomaly --model prophet --k 3

  # price forecasting with RBF SVR + ARIMA on its residuals (Table A1, Protocol VI)
  python3 drcf.py --data prices.csv --time date --value price --task forecast --model svr \
                  --lags 1,2,3 --svr-C 1 --svr-delta 20 --svr-epsilon 0.001 --arima 24,0,1

Models
  linear   y_hat = sum beta_j y(t - lag_j) + public features (intercept, hour, weekday, --exog columns).
           Covers AR / ARIMA-style predictors and regression models (PARX with weather columns via --exog).
  prophet  piecewise-linear trend + daily / yearly / weekday / weekend seasonality, fitted on each broker's share;
           anomaly if |y - y_hat| > k sigma (and optionally y == 0). Anomaly task only.
  svr      RBF SVR on min-max normalised lags (exact kernel in a garbled circuit); optional --arima p,d,q on the SVR
           residuals (d may be 0 or the seasonal period, e.g. 24) evaluated with CKKS.
Tasks
  forecast one step ahead for every test point (actual past values as inputs)
  anomaly  flag test points with |y - y_hat| > threshold; linear / svr use threshold = k * std(training residuals)

Cost guide: linear and prophet are cheap (seconds for tens of thousands of points). svr costs about
2,100 AND gates per (test point x support vector): keep --test x support vectors around 30,000 or below.
"""
import argparse
import os
import subprocess
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))


def ar_inf_weights(ar_lags, ar, d_period, ma, n):
    """One-step predictor e_hat_t = c' + sum_{j=1..n} w_j e_{t-j} for phi(B)(1 - B^s)^d e_t = c + theta(B) eps_t."""
    phi = np.zeros(n + 1); phi[0] = 1.0
    for lag, a in zip(ar_lags, ar):
        if lag <= n: phi[lag] -= a
    if d_period:
        phi = phi - np.concatenate([np.zeros(d_period), phi[:-d_period]])
    theta = np.zeros(n + 1); theta[0] = 1.0
    for j, b in enumerate(ma, start=1): theta[j] += b
    pi = np.zeros(n + 1)
    for k in range(n + 1):
        pi[k] = phi[k] - sum(theta[j] * pi[k - j] for j in range(1, min(k, len(ma)) + 1))
    return -pi[1:]


def public_features(ts, names, exog):
    cols = []
    for n in names:
        if n == "intercept": cols.append(np.ones(len(ts)))
        elif n == "hour":
            for h in range(1, 24): cols.append((ts.dt.hour == h).astype(float).to_numpy())
        elif n == "weekday":
            for d in range(1, 7): cols.append((ts.dt.dayofweek == d).astype(float).to_numpy())
        elif n: sys.exit(f"unknown public feature '{n}' (use intercept, hour, weekday)")
    for e in exog: cols.append(e)
    return np.column_stack(cols) if cols else np.zeros((len(ts), 0))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True); ap.add_argument("--time", required=True); ap.add_argument("--value", required=True)
    ap.add_argument("--series", help="column identifying the series (meter, customer)"); ap.add_argument("--series-id")
    ap.add_argument("--label", help="0/1 anomaly label column, used to score anomaly detection")
    ap.add_argument("--task", choices=["forecast", "anomaly"], required=True)
    ap.add_argument("--model", choices=["linear", "prophet", "svr"], required=True)
    ap.add_argument("--train", type=int, default=0, help="training points (default: 4 weeks of hourly data, or all before the test window)")
    ap.add_argument("--test", type=int, default=168, help="test points at the end of the series (default 168)")
    ap.add_argument("--lags", default="1,2,24"); ap.add_argument("--public", default="intercept")
    ap.add_argument("--exog", default="", help="comma-separated public covariate columns (e.g. temperature)")
    ap.add_argument("--k", type=float, default=3.0, help="anomaly threshold in standard deviations (Table A1: 3)")
    ap.add_argument("--zero-rule", action="store_true", help="prophet: also flag readings equal to 0 (Jaiswal et al.)")
    ap.add_argument("--au-holidays", action="store_true", help="prophet: Australian holidays 2011-2013")
    ap.add_argument("--svr-C", type=float, default=1.0); ap.add_argument("--svr-delta", type=float, default=20.0)
    ap.add_argument("--svr-epsilon", type=float, default=0.001)
    ap.add_argument("--arima", default="", help="svr: p,d,q for ARIMA on residuals; p may be 'a+b' for lags a and b, d is 0 or a period")
    ap.add_argument("--scale", type=float, default=100.0, help="fixed-point scale for values inside garbled circuits")
    ap.add_argument("--build", default=os.path.join(HERE, "build")); ap.add_argument("--work", default=".")
    ap.add_argument("--out", help="write a markdown report here")
    a = ap.parse_args()

    df = pd.read_csv(a.data)
    if a.series:
        sid = a.series_id if a.series_id is not None else df[a.series].iloc[0]
        df = df[df[a.series].astype(str) == str(sid)]
    df[a.time] = pd.to_datetime(df[a.time]); df = df.sort_values(a.time).dropna(subset=[a.value]).reset_index(drop=True)
    y = df[a.value].to_numpy(float); ts = df[a.time]; n = len(y)
    lags = [int(v) for v in a.lags.split(",") if v] if a.model != "prophet" else []
    maxlag = max(lags) if lags else 0
    T0 = n - a.test
    if T0 < 0: sys.exit(f"--test {a.test} is longer than the series ({n} points)")
    train = n if a.model == "prophet" else (a.train or min(T0 - maxlag, 672))
    tr0 = T0 - train
    if a.model != "prophet" and (tr0 < maxlag or T0 <= tr0): sys.exit("not enough data for the requested lags / train / test windows")
    labels = df[a.label].to_numpy(int)[T0:] if a.label else None
    exog = [df[c].to_numpy(float) for c in a.exog.split(",") if c]
    job = [f"TASK {a.task}", f"MODEL {a.model}", f"SCALE {a.scale:g}"]
    ref = None; info = ""

    if a.model == "linear":
        P = public_features(ts, [v for v in a.public.split(",") if v], exog)
        X = np.column_stack([y[t - np.array(lags)] for t in range(tr0, T0)]).T if lags else np.zeros((train, 0))
        A = np.hstack([X, P[tr0:T0]]); coef, *_ = np.linalg.lstsq(A, y[tr0:T0], rcond=None)
        beta, pc = coef[:len(lags)], coef[len(lags):]
        res = y[tr0:T0] - A @ coef; sigma = float(res.std())
        Xt = np.column_stack([y[t - np.array(lags)] for t in range(T0, n)]).T if lags else np.zeros((a.test, 0))
        ref = np.hstack([Xt, P[T0:]]) @ coef
        job += [f"TEST {T0} {a.test}", f"LAGS {len(lags)} " + " ".join(map(str, lags)), f"COEF {len(beta)} " + " ".join(f"{v:.17g}" for v in beta),
                f"PUBLIC {P.shape[1]} " + " ".join(f"{v:.17g}" for v in pc)] + [" ".join(f"{v:.17g}" for v in row) for row in P[T0:]]
        info = f"lags {lags}, {P.shape[1]} public features, OLS on {train} points, residual std {sigma:.4g}"
        if a.task == "anomaly":
            job.append(f"THRESH {a.k * sigma:.17g}"); ref = (np.abs(y[T0:] - ref) > a.k * sigma).astype(int)
    elif a.model == "prophet":
        if a.task != "anomaly": sys.exit("prophet is available for --task anomaly")
        st = [f"{t.year} {t.month} {t.day} {t.hour} {t.minute} {t.dayofweek}" for t in ts]
        job += [f"TEST {T0} {a.test}", f"PROPHET {a.k:g} {int(a.zero_rule)} {int(a.au_holidays)} {float(np.percentile(np.abs(y), 99)) or 1.0:.6g}", f"STAMPS {n}"] + st
        info = f"Prophet-style fit on all {n} points (shares), threshold {a.k:g} sigma" + (", zero rule" if a.zero_rule else "")
    else:
        from sklearn.svm import SVR
        lo, hi = y[tr0:T0].min(), y[tr0:T0].max(); z = (y - lo) / (hi - lo)
        Xtr = np.array([[z[t - l] for l in lags] for t in range(tr0, T0)]); ytr = z[tr0:T0]
        gamma = 1.0 / (2 * a.svr_delta ** 2)
        svr = SVR(kernel="rbf", C=a.svr_C, epsilon=a.svr_epsilon, gamma=gamma).fit(Xtr, ytr)
        svs, alpha, b = svr.support_vectors_, svr.dual_coef_[0], svr.intercept_[0]
        if len(alpha) * a.test > 60000: print(f"warning: {len(alpha)} support vectors x {a.test} test points is a heavy garbled circuit")
        f_n = lambda t: float(alpha @ np.exp(-gamma * ((svs - np.array([z[t - l] for l in lags])) ** 2).sum(1)) + b)
        e = {t: z[t] - f_n(t) for t in range(tr0, n)}
        Lw, cp, w = 0, 0.0, []
        if a.arima:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            p_s, d_s, q_s = a.arima.split(","); p_l = [int(v) for v in p_s.split("+") if v and v != "0"]; d = int(d_s); q = int(q_s)
            er = np.array([e[t] for t in range(tr0, T0)])
            mod = SARIMAX(er, order=(p_l, 0, q), seasonal_order=(0, 1, 0, d) if d else (0, 0, 0, 0), trend="n" if d else "c").fit(disp=False)
            pr = list(mod.params); c0 = 0.0 if d else pr.pop(0)
            ar, ma = pr[:len(p_l)], pr[len(p_l):len(p_l) + q]
            Lw = min(train - 1, 144); w = ar_inf_weights(p_l, ar, d, ma, Lw); cp = c0 / (1 + sum(ma))
        fz = np.array([f_n(t) + (cp + sum(w[j - 1] * e[t - j] for j in range(1, Lw + 1)) if Lw else 0.0) for t in range(T0, n)])
        ref = fz * (hi - lo) + lo
        job += [f"TEST {T0} {a.test}", f"LAGS {len(lags)} " + " ".join(map(str, lags)), f"NORM {lo:.17g} {hi:.17g}",
                f"SVR {gamma:.17g} {b:.17g} {len(alpha)}"] + [f"{al:.17g} " + " ".join(f"{v:.17g}" for v in sv) for al, sv in zip(alpha, svs)]
        if Lw: job += [f"ARIMA {Lw} {cp:.17g} " + " ".join(f"{v:.17g}" for v in w), f"RESID {Lw} " + " ".join(f"{e[t]:.17g}" for t in range(T0 - Lw, T0))]
        info = f"RBF SVR (C={a.svr_C:g}, delta={a.svr_delta:g}, eps={a.svr_epsilon:g}) on lags {lags}, {len(alpha)} support vectors" + (f", ARIMA({a.arima}) on residuals" if a.arima else "")
        if a.task == "anomaly":
            sigma = float(np.std([y[t] - (f_n(t) * (hi - lo) + lo) for t in range(tr0, T0)]))
            job.append(f"THRESH {a.k * sigma:.17g}"); ref = (np.abs(y[T0:] - ref) > a.k * sigma).astype(int)

    job.insert(3, f"SERIES {n}\n" + "\n".join(f"{v:.17g}" for v in y))
    jf, rf = os.path.join(a.work, "drcf_job.txt"), os.path.join(a.work, "drcf_result.txt")
    with open(jf, "w") as f: f.write("\n".join(job) + "\n")
    eng = os.path.join(a.build, "drcf_secure")
    if not os.path.exists(eng): sys.exit(f"{eng} not found: build the project first (see README)")
    t0 = time.time()
    subprocess.run([os.path.join(HERE, "run.sh"), eng, jf, rf], check=True)
    wall = time.time() - t0
    toks = open(rf).read().split()
    ms, ands, mib = float(toks[1]), float(toks[2]), float(toks[3])
    nout = int(toks[5]); sec = np.array([float(v) for v in toks[6:6 + nout]])
    if len(toks) > 6 + nout and toks[6 + nout] == "REF": ref = np.array([int(v) for v in toks[8 + nout:8 + 2 * nout]])

    lines = [f"# DRCF prototype run", "", f"- data: {a.data} ({n} points, value column `{a.value}`)",
             f"- task: {a.task}, model: {a.model} ({info})",
             f"- training: {'the whole series (fit on shares)' if a.model == 'prophet' else f'{train} points'}, test: last {a.test} points", ""]
    if a.task == "forecast":
        act = y[T0:]
        mape = lambda f: float(np.mean(np.abs((f - act) / np.where(act == 0, np.nan, act))) * 100)
        rmse = lambda f: float(np.sqrt(np.mean((f - act) ** 2)))
        lines += ["| | MAPE % | RMSE |", "|---|---|---|", f"| secure (two brokers) | {mape(sec):.3f} | {rmse(sec):.4g} |",
                  f"| plaintext model | {mape(ref):.3f} | {rmse(ref):.4g} |", "",
                  f"Largest |secure - plaintext| over the test points: {np.max(np.abs(sec - ref)):.4g} (value units)."]
    else:
        flags = sec.astype(int)
        lines.append(f"Flagged {int(flags.sum())} of {a.test} test points.")
        if ref is not None: lines.append(f"Secure flags equal the plaintext model on {100 * np.mean(flags == ref):.2f}% of points.")
        if labels is not None:
            tp = int(((flags == 1) & (labels == 1)).sum()); fp = int(((flags == 1) & (labels == 0)).sum())
            fn = int(((flags == 0) & (labels == 1)).sum()); tn = int(((flags == 0) & (labels == 0)).sum())
            prec = tp / (tp + fp) if tp + fp else 0; rec = tp / (tp + fn) if tp + fn else 0; spec = tn / (tn + fp) if tn + fp else 0
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
            lines += ["", "| precision | recall | F1 | G-mean |", "|---|---|---|---|", f"| {prec:.3f} | {rec:.3f} | {f1:.3f} | {np.sqrt(rec * spec):.3f} |"]
    lines += ["", f"Cost: {ms / 1000:.2f} s of secure computation ({wall:.1f} s wall), {ands / 1e6:.1f} M AND gates, {mib:.1f} MiB between the brokers (localhost)."]
    report = "\n".join(lines)
    print(report)
    if a.out:
        with open(a.out, "w") as f: f.write(report + "\n")


if __name__ == "__main__":
    main()
