#!/usr/bin/env python3
"""Protocol III (Table A1): sliding-window hourly seasonal ARIMA, SWHSA of Alberg & Last (2018),
on the UCI Individual Household Electric Power Consumption data (hourly mean of Global_active_power, kW).

Model owner side, in plaintext (Table A1: the owner trains and shares the coefficients):
  - SARIMA(0,0,1)(0,1,1)24, the paper's best hourly model (MAPE 9.19% on their meters)
  - training window Wtr = 96 days (2304 h), validation window Wval = 24 h, slide by Wval
  - concept drift rule: re-induce the model when MAPE_val / MAPE_tr > Th (Th = 1 by default; the paper does not give Th)
The 24 forecasts of each validation window are written as linear predictors in past readings
(truncated AR(inf) form, L lags), so the brokers evaluate them on shares with CKKS.

Usage: python3 scripts/train_protocol3.py hourly.csv job.txt reference.txt [days=365] [L=504] [Th=1]
hourly.csv has columns time,kw (examples/prepare_uci_household.py makes it)."""
import sys, warnings
import numpy as np, pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
warnings.filterwarnings("ignore")

csv, job, ref = sys.argv[1:4]
DAYS = int(sys.argv[4]) if len(sys.argv) > 4 else 365
L = int(sys.argv[5]) if len(sys.argv) > 5 else 504
TH = float(sys.argv[6]) if len(sys.argv) > 6 else 1.0
WTR, WVAL, S = 96 * 24, 24, 24

y = pd.read_csv(csv)["kw"].to_numpy()
n = len(y)
n = n - (n % 24)                       # whole days
T_first = n - DAYS * WVAL              # first forecast hour
assert T_first - WTR - L >= 0, "series too short"

def fit(t1, t2):
    m = SARIMAX(y[t1:t2], order=(0, 0, 1), seasonal_order=(0, 1, 1, S)).fit(disp=False)
    return m

def ar_weights(theta, Theta):
    # pi(B) = (1 - B^24) / ((1 + theta B)(1 + Theta B^24)), y_t = sum_{j>=1} c_j y_{t-j} + e_t
    ma = np.zeros(L + 26); ma[0] = 1; ma[1] += theta; ma[S] += Theta; ma[S + 1] += theta * Theta
    num = np.zeros(L + 1); num[0] = 1; num[S] = -1
    pi = np.zeros(L + 1)
    for k in range(L + 1):              # pi = num / ma (power series division)
        v = num[k] - sum(ma[i] * pi[k - i] for i in range(1, min(k, S + 1) + 1))
        pi[k] = v
    return -pi[1:]                      # c_j, j = 1..L

def rows_for(c):
    # expand the recursion directly: each forecast is a vector over the observed past y_{T-1}, y_{T-2}, ...
    R = []
    for h in range(WVAL):
        w = np.zeros(L + WVAL)          # w[k] = weight of y_{T-1-k}
        for j in range(1, L + 1):
            src = h - j                 # forecast index being used, or observed if < 0
            if src >= 0: w += c[j - 1] * R[src]
            else:
                k = -src - 1
                if k < L + WVAL: w[k] += c[j - 1]
        R.append(w)
    return np.array(R)

rows, ref_sm, ref_lin, refits = [], [], [], 0
model = None; mape_tr = None
for d in range(DAYS):
    T = T_first + d * WVAL
    t1, t2 = T - WTR, T
    if model is None or (mape_tr > 0 and mape_val / mape_tr > TH):
        model = fit(t1, t2); refits += 1
        theta, Theta = model.params[0], model.params[1]
        c = ar_weights(theta, Theta)
        Rw = rows_for(c)
        fitted = model.fittedvalues[S + 1:]
        mape_tr = np.mean(np.abs((fitted - y[t1 + S + 1:t2]) / y[t1 + S + 1:t2])) * 100
    # forecasts of the current model for hours T..T+23 from data up to T-1
    past = y[T - 1::-1][:L + WVAL]      # y_{T-1}, y_{T-2}, ...
    lin = Rw @ past
    sm = model.apply(y[T - WTR:T], refit=False).forecast(WVAL)
    mape_val = np.mean(np.abs((lin - y[T:T + WVAL]) / y[T:T + WVAL])) * 100
    ref_lin += list(lin); ref_sm += list(sm)
    for h in range(WVAL):
        rows.append([(k + 1 + h, Rw[h][k]) for k in range(L + WVAL) if abs(Rw[h][k]) > 1e-7])

K = max(len(r) for r in rows)
for r in rows:
    while len(r) < K: r.append((r[0][0], 0.0))
with open(job, "w") as f:
    f.write("TASK forecast\nMODEL linear\nSCALE 10000\nMASK 34\n")   # MASK: CKKS mask 2^34 over products of at most 2^21
    f.write(f"SERIES {n}\n" + "\n".join(f"{v:.6f}" for v in y[:n]) + "\n")
    f.write(f"TEST {T_first} {DAYS * WVAL}\nROWS {K}\n")
    for r in rows: f.write("0 " + " ".join(f"{lag} {cf:.9g}" for lag, cf in r) + "\n")
act = y[T_first:T_first + DAYS * WVAL]
with open(ref, "w") as f:
    f.write(f"# hour actual linear_ref statsmodels_ref\n")
    for i in range(len(act)): f.write(f"{T_first + i} {act[i]:.6f} {ref_lin[i]:.6f} {ref_sm[i]:.6f}\n")
naive = y[T_first - 24:T_first - 24 + DAYS * WVAL]
mape = lambda p: np.mean(np.abs((np.array(p) - act) / act)) * 100
print(f"windows {DAYS}, refits {refits}, rows {len(rows)} x {K} lags")
print(f"MAPE linear (truncated L={L}) {mape(ref_lin):.3f}%   statsmodels {mape(ref_sm):.3f}%")
print(f"naive hourly baseline (y[t-24]) {mape(naive):.3f}%")
print(f"max |linear - statsmodels| {np.max(np.abs(np.array(ref_lin) - np.array(ref_sm))):.4f} kW")
