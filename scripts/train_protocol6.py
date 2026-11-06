#!/usr/bin/env python3
"""Model owner for DRCF Table A1, Protocol VI (ML price forecast), following
J. Che and J. Wang, "Short-term electricity prices forecasting based on support vector regression and
Auto-regressive integrated moving average modeling", Energy Conversion and Management 51(10), 2010.

SVRARIMA, as in the paper:
  * training data: hourly prices of the 7 days before the forecast week (Sec. 3.1)
  * min-max normalisation, Eq. (3.1), using the training window
  * SVR on the phase-space reconstruction with m = 3, tau = 1, Eq. (2.1)-(2.3):  X(j+1) = F(X(j), X(j-1), X(j-2))
  * RBF kernel K(x, x') = exp(-||x - x'||^2 / (2 delta^2)), Eq. (2.9), with C = 1, delta = 20, epsilon = 0.001 (Figs. 4, 7)
  * ARIMA on the SVR residuals, Eqs. (2.11)-(2.13): week type 1 uses p = (24), d = (0), q = (1) (Fig. 4);
    week type 2 uses p = (1, 24), d = (24), q = (1) (Fig. 7)
  * forecast = SVR forecast + ARIMA forecast of the residual, one hour ahead; RMSE and MAPE, Eqs. (3.2)-(3.3)
Test weeks: the same calendar weeks as the paper (Feb 26 - Mar 3 and Apr 1 - 7) in 2018, 2019 and 2020, on the
Polish day-ahead market data (DRCF Table 3, dataset 6).

The model owner trains in the clear and writes the parameters; the brokers receive them as secret shares
(Table A1 input beta) and run the forecast on shared prices. The ARIMA one-step forecast is written as its
equivalent linear predictor in past residuals (AR(infinity) form, truncated at L lags).

Usage: python3 scripts/train_protocol6.py <path to data/data.xlsx> <out dir>
Writes <out>/prices_p6.txt, <out>/model_p6.txt, <out>/reference_p6.txt (plain text, whitespace separated).
The model file also carries the owner's in-sample residuals for the last L training hours (needed by the ARIMA part).
"""
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.svm import SVR
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")
M, C, DELTA, EPS = 3, 1.0, 20.0, 0.001
GAMMA = 1.0 / (2 * DELTA ** 2)
L = 144                                    # truncation of the AR(infinity) predictor (< 165 available residuals)
WEEKS = [(f"{y}-02-26", 1) for y in (2018, 2019, 2020)] + [(f"{y}-04-01", 2) for y in (2018, 2019, 2020)]


def series_weights(ar_lags, ar, d24, ma, n):
    """One-step predictor e_hat_t = c' + sum_{j=1..n} w_j e_{t-j} for phi(B)(1-B^24)^d e_t = c + theta(B) eps_t."""
    phi = np.zeros(n + 1); phi[0] = 1.0
    for lag, a in zip(ar_lags, ar): phi[lag] -= a
    if d24:
        phi = phi - np.concatenate([np.zeros(24), phi[:-24]])
    theta = np.zeros(n + 1); theta[0] = 1.0
    for j, b in enumerate(ma, start=1): theta[j] += b
    pi = np.zeros(n + 1)                    # pi(B) = phi(B) / theta(B), eps_t = pi(B) e_t - c / theta(1)
    for k in range(n + 1):
        pi[k] = phi[k] - sum(theta[j] * pi[k - j] for j in range(1, min(k, len(ma)) + 1))
    return -pi[1:]                          # e_hat_t = e_t - eps_t = c' - sum_{j>=1} pi_j e_{t-j}


def main(xlsx, out):
    df = pd.read_excel(xlsx)
    df["date"] = pd.to_datetime(df["date"])
    price = df["electricity_price (PLN/MWh)"].to_numpy(float)
    with open(f"{out}/prices_p6.txt", "w") as f:
        for p in price: f.write(f"{p:.2f}\n")
    mo = open(f"{out}/model_p6.txt", "w"); rf = open(f"{out}/reference_p6.txt", "w")
    print(f"{'week':<12} {'type':>4} {'nSV':>4} | {'SVR RMSE':>9} {'SVR MAPE%':>9} | {'SVRARIMA RMSE':>13} {'MAPE%':>7}")
    for start, wtype in WEEKS:
        t0 = int(np.searchsorted(df["date"].to_numpy(), np.datetime64(start)))
        tr0 = t0 - 168                                       # 7 previous days
        train = price[tr0:t0]
        lo, hi = train.min(), train.max()
        z = (price - lo) / (hi - lo)                         # Eq. (3.1), training min/max
        X = np.array([[z[j], z[j - 1], z[j - 2]] for j in range(tr0 + 2, t0 - 1)])
        y = np.array([z[j + 1] for j in range(tr0 + 2, t0 - 1)])
        svr = SVR(kernel="rbf", C=C, epsilon=EPS, gamma=GAMMA).fit(X, y)
        svs, coef, b = svr.support_vectors_, svr.dual_coef_[0], svr.intercept_[0]

        def svr_at(t):                                       # one-step forecast of z[t]
            x = np.array([z[t - 1], z[t - 2], z[t - 3]])
            return float(coef @ np.exp(-GAMMA * ((svs - x) ** 2).sum(1)) + b)

        hist = list(range(tr0 + 3, t0))                      # in-sample residuals on the training window
        e = {t: z[t] - svr_at(t) for t in range(tr0 + 3, t0 + 168)}
        er = np.array([e[t] for t in hist])
        if wtype == 1:
            mod = SARIMAX(er, order=([24], 0, 1), trend="c").fit(disp=False)
            c, ar_l, ar, ma, d24 = mod.params[0], [24], [mod.params[1]], [mod.params[2]], False
        else:
            mod = SARIMAX(er, order=([1, 24], 0, 1), seasonal_order=(0, 1, 0, 24), trend="n").fit(disp=False)
            c, ar_l, ar, ma, d24 = 0.0, [1, 24], list(mod.params[0:2]), [mod.params[2]], True
        w = series_weights(ar_l, ar, d24, ma, L)
        cp = c / (1 + sum(ma))
        # reference forecasts (clear), same formulas the brokers evaluate on shares
        f_svr, f_hyb, act = [], [], []
        for t in range(t0, t0 + 168):
            n_hat = svr_at(t)
            e_hat = cp + sum(w[j - 1] * e[t - j] for j in range(1, L + 1))
            f_svr.append(n_hat * (hi - lo) + lo); f_hyb.append((n_hat + e_hat) * (hi - lo) + lo); act.append(price[t])
        f_svr, f_hyb, act = map(np.array, (f_svr, f_hyb, act))
        rmse = lambda f: float(np.sqrt(np.mean((f - act) ** 2)))
        mape = lambda f: float(np.mean(np.abs((f - act) / act)) * 100)
        print(f"{start:<12} {wtype:>4} {len(coef):>4} | {rmse(f_svr):>9.3f} {mape(f_svr):>9.3f} | {rmse(f_hyb):>13.3f} {mape(f_hyb):>7.3f}")
        mo.write(f"WEEK {start} {wtype} {t0} {tr0}\nNORM {lo:.17g} {hi:.17g}\nSVR {GAMMA:.17g} {b:.17g} {len(coef)}\n")
        for a, sv in zip(coef, svs): mo.write(f"{a:.17g} {sv[0]:.17g} {sv[1]:.17g} {sv[2]:.17g}\n")
        mo.write(f"ARIMA {L} {cp:.17g}\n" + " ".join(f"{v:.17g}" for v in w) + "\n")
        mo.write(f"RESID {L}\n" + " ".join(f"{e[t]:.17g}" for t in range(t0 - L, t0)) + "\n")   # owner's training residuals
        rf.write(f"WEEK {start}\n" + "\n".join(f"{a:.2f} {s:.17g} {h:.17g}" for a, s, h in zip(act, f_svr, f_hyb)) + "\n")
    mo.close(); rf.close()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else ".")
