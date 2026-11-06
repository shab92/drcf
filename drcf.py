#!/usr/bin/env python3
"""DRCF prototype: try privacy-preserving forecasting / anomaly detection on your own time series.

The tool plays the model owner (trains in the clear on the training window), then runs the real two-broker
DRCF computation (build/drcf_secure, both brokers on this machine) on secret-shared data, and reports
accuracy, how closely the secure result matches the plaintext model, and the cost.

Quick start: pick a Table A1 protocol, point it at your CSV. Paper settings are the defaults; every one can be changed.
  python3 drcf.py --protocol 1 --data meter.csv --time ts --value kwh --temp temperature     # PARX anomaly detection
  python3 drcf.py --protocol 2 --data meter.csv --time ts --value kwh                        # Prophet anomaly detection
  python3 drcf.py --protocol 3 --data load.csv  --time ts --value kw --test 720              # sliding-window SARIMA, 30 days
  python3 drcf.py --protocol 4 --data load.csv  --time ts --value kw --series client          # daily-peak SVR, 31 days
  python3 drcf.py --protocol 5 --data price.csv --time ts --value price                      # log-ARIMA price forecast
  python3 drcf.py --protocol 6 --data price.csv --time ts --value price                      # SVR + ARIMA price forecast

Protocols (paper Table A1, model from the cited paper)
  1  PARX, Ardakanian et al. 2014: one OLS model per (hour, weekday / weekend-holiday), lags = same hour on the
     previous p = 3 days of that type, temperature terms (T-20)+, (16-T)+, (5-T)+ from --temp; anomaly if
     |y - y_hat| > k sigma of that season (k = 3). Their occupancy terms need the current reading and are left out.
  2  Prophet as configured by Jaiswal et al. 2021 (99% interval, zero readings flagged).
  3  SWHSA, Alberg & Last 2018: SARIMA(0,0,1)(0,1,1)24 on a 96-day window, 24 h ahead, slide 24 h, refit when
     MAPE_val / MAPE_tr > th (th = 1).
  4  Chen, Chang & Lin 2004: daily maximum of the (summed) load, 31 days ahead recursively, 7 previous maxima +
     weekday bits + holiday bit, RBF SVR, winter months for training (--months), C and gamma by validation.
  5  Contreras et al. 2003, Eq. (3) (Spain): multiplicative ARIMA on log prices, 24 h ahead each day.
     Eq. (4) (California): --ar "1,2;23,24,47,48,72,96,120,144;167,168,169,192" --diff 1,24,168
     --ma "1,2;24,48,72,96;144;168,336,504". The publisher takes the log of its own reading before sharing; the
     subscriber exponentiates the forecast it receives.
  6  SVRARIMA, Che & Wang 2010: RBF SVR on lags 1-3 + ARIMA on its residuals, one hour ahead.
Protocols 1, 3 and 5 need hourly data (other spacings are resampled to hourly means); 4 needs hourly data too.

Examples (generic models)
  # load forecasting with an AR model on lags 1, 2, 24 plus hour-of-day and weekday effects
  python3 drcf.py --data load.csv --time timestamp --value kwh --task forecast --model linear \
                  --lags 1,2,24 --public intercept,hour,weekday

  # anomaly detection with the Prophet-style model (Table A1, Protocol II), labels for scoring
  python3 drcf.py --data meter.csv --time ts --value kwh --label anomaly --task anomaly --model prophet --k 3

  # price forecasting with RBF SVR + ARIMA on its residuals (Table A1, Protocol VI)
  python3 drcf.py --data prices.csv --time date --value price --task forecast --model svr \
                  --lags 1,2,3 --svr-C 1 --svr-delta 20 --svr-epsilon 0.001 --arima 24,0,1

Models
  parx, arima, svr-daily   the models behind protocols 1, 3/5 and 4 (see above)
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
    ap.add_argument("--protocol", type=int, choices=range(1, 7), help="Table A1 protocol with the cited paper's settings")
    ap.add_argument("--task", choices=["forecast", "anomaly"])
    ap.add_argument("--model", choices=["linear", "prophet", "svr", "parx", "arima", "svr-daily"])
    ap.add_argument("--train", type=int, default=0, help="training points (default: 4 weeks of hourly data, or all before the test window)")
    ap.add_argument("--test", type=int, default=None, help="test points at the end of the series (default 168)")
    ap.add_argument("--lags", default="1,2,24"); ap.add_argument("--public", default="intercept")
    ap.add_argument("--exog", default="", help="comma-separated public covariate columns (e.g. temperature)")
    ap.add_argument("--temp", help="parx: temperature column in degrees C (public weather data)")
    ap.add_argument("--p", type=int, default=3, help="parx: previous days of the same type used as lags (paper: 3)")
    ap.add_argument("--holidays", default="", help="parx / svr-daily: country code for public holidays, e.g. PT, CA (needs the holidays package)")
    ap.add_argument("--ar", default=None, help="arima: AR lags, one factor per ';', e.g. '1,2,3,4,5;23,24;168'")
    ap.add_argument("--ma", default=None, help="arima: MA lags, same format"); ap.add_argument("--diff", default=None, help="arima: difference periods, e.g. 24 or 1,24,168")
    ap.add_argument("--const", type=int, default=None, help="arima: 1 to fit a constant"); ap.add_argument("--log", action="store_true", help="arima: model log(value)")
    ap.add_argument("--window", type=int, default=None, help="arima: training window in points (0 = all history)")
    ap.add_argument("--horizon", type=int, default=24, help="arima: points forecast from each origin (24 = day ahead)")
    ap.add_argument("--truncate", type=int, default=2304, help="arima: lags kept in the linear predictor")
    ap.add_argument("--refit", choices=["drift", "every", "once"], default=None, help="arima: when to re-estimate (drift = Alberg & Last rule)")
    ap.add_argument("--th", type=float, default=1.0, help="arima: drift threshold on MAPE_val / MAPE_tr")
    ap.add_argument("--months", default="1,2,3,10,11,12", help="svr-daily: training months (paper: winter)")
    ap.add_argument("--k", type=float, default=None, help="anomaly threshold in standard deviations (Table A1: 3)")
    ap.add_argument("--zero-rule", action="store_true", help="prophet: also flag readings equal to 0 (Jaiswal et al.)")
    ap.add_argument("--au-holidays", action="store_true", help="prophet: Australian holidays 2011-2013")
    ap.add_argument("--svr-C", type=float, default=1.0); ap.add_argument("--svr-delta", type=float, default=20.0)
    ap.add_argument("--svr-eps-daily", type=float, default=0.1, help="svr-daily: epsilon on [0,1]-scaled load (LIBSVM default)")
    ap.add_argument("--svr-epsilon", type=float, default=0.001)
    ap.add_argument("--arima", default="", help="svr: p,d,q for ARIMA on residuals; p may be 'a+b' for lags a and b, d is 0 or a period")
    ap.add_argument("--scale", type=float, default=None, help="fixed-point scale for values inside garbled circuits")
    ap.add_argument("--build", default=os.path.join(HERE, "build")); ap.add_argument("--work", default=".")
    ap.add_argument("--out", help="write a markdown report here")
    a = ap.parse_args()
    P = {1: dict(model="parx", task="anomaly", k=3.0, test=168),
         2: dict(model="prophet", task="anomaly", k=2.576, zero_rule=True, test=None),
         3: dict(model="arima", task="forecast", ar="", ma="1;24", diff="24", const=0, window=2304, refit="drift", test=720),
         4: dict(model="svr-daily", task="forecast", test=31),
         5: dict(model="arima", task="forecast", ar="1,2,3,4,5;23,24,47,48,72,96,120,144;168,336,504", ma="1,2;24;168,336,504",
                 diff="", const=1, log=True, window=0, refit="once", test=168),
         6: dict(model="svr", task="forecast", lags="1,2,3", svr_C=1.0, svr_delta=20.0, svr_epsilon=0.001, arima="24,0,1", train=168, test=168)}
    if a.protocol:
        given = {x.lstrip("-").split("=")[0].replace("-", "_") for x in sys.argv[1:] if x.startswith("--")}
        for kk, v in P[a.protocol].items():
            if kk not in given: setattr(a, kk, v)
    if not a.model or not a.task: sys.exit("give --protocol N, or --model and --task")
    for kk, v in dict(test=168, k=3.0, ar="", ma="", diff="", const=1, window=0, refit="once").items():
        if getattr(a, kk) is None: setattr(a, kk, v)
    auto_scale = a.scale is None
    if auto_scale: a.scale = 100.0
    if a.model == "prophet" and a.test is None: a.test = 168

    df = pd.read_csv(a.data)
    if a.series and not (a.model == "svr-daily" and a.series_id is None):
        sid = a.series_id if a.series_id is not None else df[a.series].iloc[0]
        df = df[df[a.series].astype(str) == str(sid)]
    df[a.time] = pd.to_datetime(df[a.time]); df = df.sort_values(a.time).dropna(subset=[a.value]).reset_index(drop=True)
    if a.model == "svr-daily":
        return run_svr_daily(a, df)
    if a.model in ("parx", "arima"):           # hourly series
        num = [a.value] + ([a.temp] if a.temp else []) + ([a.label] if a.label else [])
        step = df[a.time].diff().median()
        if step != pd.Timedelta(hours=1):
            df = df.set_index(a.time)[num].resample("h").mean().interpolate(limit_direction="both").reset_index()
            print(f"resampled to hourly means ({len(df)} hours)")
    y = df[a.value].to_numpy(float); ts = df[a.time]; n = len(y)
    lags = [int(v) for v in a.lags.split(",") if v] if a.model in ("linear", "svr") else []
    maxlag = max(lags) if lags else 0
    T0 = n - a.test
    if T0 < 0: sys.exit(f"--test {a.test} is longer than the series ({n} points)")
    train = n if a.model == "prophet" else (a.train or min(T0 - maxlag, 672))
    if a.model in ("parx", "arima"): train = a.train or T0
    tr0 = T0 - train
    if a.model in ("linear", "svr") and (tr0 < maxlag or T0 <= tr0): sys.exit("not enough data for the requested lags / train / test windows")
    labels = df[a.label].to_numpy(int)[T0:] if a.label else None
    mask = 40
    if a.model in ("parx", "arima"):
        # CKKS fixed point: shares are 30-bit integers at --scale; keep about 13 bits of statistical hiding on both the
        # shares (scale * max|y| <= 2^17) and the masks on the products (mask >= 2^13 * share * coefficient share)
        top = float(np.max(np.abs(np.log(y) if (a.model == "arima" and a.log and (y > 0).all()) else y))) or 1.0
        if auto_scale: a.scale = 10.0 ** np.floor(np.log10(2 ** 17 / top))
        mask = int(np.ceil(np.log2(2 ** 30 / a.scale * 16))) + 13
    exog = [df[c].to_numpy(float) for c in a.exog.split(",") if c]
    job = [f"TASK {a.task}", f"MODEL {'linear' if a.model in ('parx', 'arima') else a.model}", f"SCALE {a.scale:g}"]
    if a.model in ("parx", "arima"): info_fx = f"; scale {a.scale:g}, CKKS mask 2^{mask}"
    else: info_fx = ""
    ref = None; info = ""; yser = y; unit = None

    if a.model == "parx":
        import drcf_models as dm
        hol = holiday_set(a.holidays, ts)
        temp = df[a.temp].to_numpy(float) if a.temp else None
        rows, th, f_ref, ns = dm.parx_rows(y, ts, tr0, T0, n, a.p, temp, hol, a.k)
        job += [f"TEST {T0} {a.test}", f"MASK {mask}"] + dm.rows_job(rows, th if a.task == "anomaly" else None)
        ref = (np.abs(y[T0:] - f_ref) > th).astype(int) if a.task == "anomaly" else f_ref
        info = f"PARX p={a.p}, {ns} seasons, {'temperature terms' if a.temp else 'no temperature'}, OLS on {train} points" + (f", threshold {a.k:g} sigma per season" if a.task == "anomaly" else "")
    elif a.model == "arima":
        import drcf_models as dm
        if a.task != "forecast": sys.exit("arima is available for --task forecast")
        if a.log:
            if (y <= 0).any(): sys.exit("--log needs positive values")
            yser = np.log(y); unit = np.exp
        ar, ma = dm.parse_factors(a.ar), dm.parse_factors(a.ma)
        diffs = [int(v) for v in a.diff.split(",") if v]
        window = a.window or T0; train = min(window, T0)
        L = min(a.truncate, T0 - a.horizon)
        rows, ref, fits = dm.arima_rows(yser, T0, n, ar, ma, diffs, bool(a.const), window, a.horizon, L, a.refit, a.th, a.log)
        job += [f"TEST {T0} {a.test}", f"MASK {mask}"] + dm.rows_job(rows)
        if unit is not None: ref = unit(ref)
        info = (f"ARIMA AR {a.ar or '-'} | diff {a.diff or '-'} | MA {a.ma or '-'}{' | const' if a.const else ''}{' on log values' if a.log else ''}; "
                f"window {window} points, {a.horizon} ahead per origin, refit {a.refit} ({fits} fits), {L} lags")
    elif a.model == "linear":
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

    job.insert(3, f"SERIES {n}\n" + "\n".join(f"{v:.17g}" for v in yser))
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
    if unit is not None: sec = unit(sec)
    info += info_fx
    report(a, n, info, train, y[T0:], sec, ref, labels, ms, ands, mib, wall)


def holiday_set(code, ts):
    if not code: return frozenset()
    import holidays
    return frozenset(holidays.country_holidays(code, years=sorted(set(ts.dt.year))).keys())


def run_svr_daily(a, df):
    import drcf_models as dm
    if a.series and a.series_id is None:          # every series is a publisher; the brokers add their shares
        wide = df.pivot_table(index=a.time, columns=a.series, values=a.value, aggfunc="sum")
    else:
        wide = df.set_index(a.time)[[a.value]]
    wide = wide.resample("h").mean().interpolate(limit_direction="both").fillna(0)
    wide = wide[wide.index >= wide.index[0].normalize() + pd.Timedelta(days=int(wide.index[0].hour != 0))]
    nd = len(wide) // 24; wide = wide.iloc[:nd * 24]
    dates = [d.date() for d in wide.index[::24]]
    ntest = a.test
    if nd - ntest < 30: sys.exit("svr-daily needs more history (at least a month before the test days)")
    hol = holiday_set(a.holidays, pd.Series(wide.index))
    months = {int(v) for v in a.months.split(",") if v}
    scale = a.scale if a.scale != 100.0 else (1.0 if np.allclose(wide.values, np.round(wide.values)) else 1000.0)
    lines, ref, act, info = dm.svr_daily(wide.to_numpy(float), dates, ntest, months, hol, a.svr_eps_daily, scale)
    mf, rf = os.path.join(a.work, "drcf_model_p4.txt"), os.path.join(a.work, "drcf_result.txt")
    with open(mf, "w") as f: f.write("\n".join(lines) + "\n")
    eng = os.path.join(a.build, "protocol4_daily")
    if not os.path.exists(eng): sys.exit(f"{eng} not found: build the project first (see README)")
    t0 = time.time(); subprocess.run([os.path.join(HERE, "run.sh"), eng, mf, rf], check=True, stdout=subprocess.DEVNULL); wall = time.time() - t0
    toks = open(rf).read().split()
    ms, ands, mib = float(toks[1]), float(toks[2]), float(toks[3]); sec = np.array([float(v) for v in toks[6:6 + int(toks[5])]])
    info = f"daily-peak RBF SVR, {wide.shape[1]} publisher(s) summed, {info}"
    report(a, nd, info, None, act, sec, ref, None, ms, ands, mib, wall, unit_name="days")


def report(a, n, info, train, act, sec, ref, labels, ms, ands, mib, wall, unit_name="points"):
    tr = "the whole series (fit on shares)" if a.model == "prophet" else ("model-specific (see above)" if train is None else f"{train} points")
    lines = [f"# DRCF prototype run", "", f"- data: {a.data} ({n} {unit_name}, value column `{a.value}`)",
             f"- {'protocol ' + str(a.protocol) + ', ' if a.protocol else ''}task: {a.task}, model: {a.model} ({info})",
             f"- training: {tr}, test: last {a.test} {unit_name}", ""]
    if a.task == "forecast":
        mape = lambda f: float(np.mean(np.abs((f - act) / np.where(act == 0, np.nan, act))) * 100)
        rmse = lambda f: float(np.sqrt(np.mean((f - act) ** 2)))
        lines += ["| | MAPE % | RMSE |", "|---|---|---|", f"| secure (two brokers) | {mape(sec):.3f} | {rmse(sec):.4g} |",
                  f"| plaintext model | {mape(ref):.3f} | {rmse(ref):.4g} |", "",
                  f"Largest |secure - plaintext| over the test points: {np.max(np.abs(sec - ref)):.4g} (value units)."]
    else:
        flags = sec.astype(int)
        lines.append(f"Flagged {int(flags.sum())} of {len(flags)} test points.")
        if ref is not None: lines.append(f"Secure flags equal the plaintext model on {100 * np.mean(flags == ref):.2f}% of points.")
        if labels is not None:
            tp = int(((flags == 1) & (labels == 1)).sum()); fp = int(((flags == 1) & (labels == 0)).sum())
            fn = int(((flags == 0) & (labels == 1)).sum()); tn = int(((flags == 0) & (labels == 0)).sum())
            prec = tp / (tp + fp) if tp + fp else 0; rec = tp / (tp + fn) if tp + fn else 0; spec = tn / (tn + fp) if tn + fp else 0
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
            lines += ["", "| precision | recall | F1 | G-mean |", "|---|---|---|---|", f"| {prec:.3f} | {rec:.3f} | {f1:.3f} | {np.sqrt(rec * spec):.3f} |"]
    lines += ["", f"Cost: {ms / 1000:.2f} s of secure computation ({wall:.1f} s wall), {ands / 1e6:.1f} M AND gates, {mib:.1f} MiB between the brokers (localhost)."]
    text = "\n".join(lines)
    print(text)
    if a.out:
        with open(a.out, "w") as f: f.write(text + "\n")


if __name__ == "__main__":
    main()
