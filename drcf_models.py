"""Model-owner side of the Table A1 protocols for drcf.py: fits the paper's model in the clear on your data and
writes what the brokers need (secret-shared by the harness). Nothing here sees the brokers' computation.

  parx_rows      Protocol I   PARX of Ardakanian et al. (2014)
  arima_rows     Protocol III sliding-window (S)ARIMA of Alberg & Last (2018)
                 Protocol V   multiplicative ARIMA on log prices of Contreras et al. (2003)
  svr_daily      Protocol IV  daily-peak RBF SVR of Chen, Chang & Lin (2004)
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import lfilter


# ---------------------------------------------------------------- Protocol I: PARX
def temp_terms(temp):
    """Ardakanian et al. Eqs. 2-4: cooling above 20 C, heating below 16 C, overheating below 5 C."""
    T = np.asarray(temp, float)
    return np.column_stack([np.maximum(T - 20, 0), np.maximum(16 - T, 0), np.maximum(5 - T, 0)])


def parx_rows(y, ts, tr0, T0, n, p=3, temp=None, holiday_days=frozenset(), k=3.0):
    """One season per (hour of day, weekday vs weekend/holiday). Lags are the same hour on the previous p days of
    the same day type. OLS per season. Returns per-test-point rows (intercept, [(lag, coef)]), thresholds, reference."""
    days = ts.dt.normalize()
    off = np.array([(d.dayofweek >= 5) or (d.date() in holiday_days) for d in days])      # weekend / holiday
    hour = ts.dt.hour.to_numpy()
    X3 = temp_terms(temp) if temp is not None else np.zeros((n, 0))
    # index of the same hour on the previous days of the same type
    prev = {}
    last = {}
    for t in range(n):
        key = (hour[t], off[t])
        prev[t] = list(last.get(key, []))[-p:][::-1]          # most recent first
        last.setdefault(key, []).append(t)
    seasons = {}
    for t in range(tr0, T0):
        if len(prev[t]) == p: seasons.setdefault((hour[t], off[t]), []).append(t)
    model = {}
    for s, idx in seasons.items():
        A = np.array([[y[j] for j in prev[t]] + list(X3[t]) + [1.0] for t in idx]); b = y[idx]
        c, *_ = np.linalg.lstsq(A, b, rcond=None)
        model[s] = (c, float((b - A @ c).std()) or 1e-9)
    rows, th, ref = [], [], []
    for t in range(T0, n):
        s = (hour[t], off[t])
        if s not in model or len(prev[t]) < p: raise SystemExit(f"no PARX season / history for test point {t}: use a longer training window")
        c, sd = model[s]
        icpt = float(c[-1] + (X3[t] @ c[p:p + X3.shape[1]] if X3.shape[1] else 0.0))
        rows.append((icpt, [(t - j, float(c[i])) for i, j in enumerate(prev[t])]))
        th.append(k * sd); ref.append(icpt + sum(c[i] * y[j] for i, j in enumerate(prev[t])))
    return rows, np.array(th), np.array(ref), len(model)


# ---------------------------------------------------------------- Protocols III and V: factored ARIMA
def parse_factors(spec):
    """'1,2;24;168,336' -> [[1,2],[24],[168,336]] (one polynomial factor per ';')."""
    return [[int(v) for v in f.split(",") if v] for f in spec.split(";") if f.strip()] if spec else []


def poly(factors, coefs, sign):
    """prod_f (1 + sign * sum_l c_l B^l)."""
    out = np.array([1.0]); i = 0
    for f in factors:
        q = np.zeros(max(f) + 1); q[0] = 1.0
        for l in f: q[l] += sign * coefs[i]; i += 1
        out = np.convolve(out, q)
    return out


def diff_poly(periods):
    out = np.array([1.0])
    for s in periods:
        q = np.zeros(s + 1); q[0] = 1; q[s] = -1; out = np.convolve(out, q)
    return out


class Arima:
    """Phi(B) D(B) y_t = c + Theta(B) e_t, Phi and Theta products of sparse factors, D seasonal/regular differences.
    Fitted by conditional least squares."""
    def __init__(self, ar, ma, diffs, const):
        self.ar, self.ma, self.diffs, self.const = ar, ma, diffs, const
        self.nar, self.nma = sum(map(len, ar)), sum(map(len, ma))

    def _polys(self, th):
        A = np.convolve(poly(self.ar, th[:self.nar], -1), diff_poly(self.diffs))
        M = poly(self.ma, th[self.nar:self.nar + self.nma], +1)
        c = th[-1] if self.const else 0.0
        return A, M, c

    def resid(self, th, y):
        A, M, c = self._polys(th)
        return lfilter(A, M, y) - c / M.sum()

    def fit(self, y):
        burn = len(self._polys(np.zeros(self.nar + self.nma + self.const))[0]) - 1
        if burn >= len(y) - 10: raise SystemExit("training window too short for this ARIMA")
        f = lambda th: self.resid(th, y)[burn:]
        x0 = np.zeros(self.nar + self.nma + self.const)
        if self.const: x0[-1] = float(np.mean(lfilter(diff_poly(self.diffs), [1.0], y)[burn:]))
        self.th = least_squares(f, x0, method="lm" if len(x0) < len(y) - burn else "trf").x
        self.burn = burn
        return self

    def ar_inf(self, L):
        """y_t = c' + sum_{j=1..L} w_j y_{t-j} + e_t (truncated AR(inf) form)."""
        A, M, c = self._polys(self.th)
        imp = np.zeros(L + 1); imp[0] = 1.0
        pi = lfilter(A, M, imp)
        return -pi[1:], c / M.sum()


def forecast_rows(w, c, H):
    """Rows for y_{T+h}, h = 0..H-1, from y_{T-1}, y_{T-2}, ...: weight vectors over y_{T-1-k}, plus intercepts."""
    L = len(w); R, C = [], []
    for h in range(H):
        v = np.zeros(L + H); ic = c
        for j in range(1, L + 1):
            src = h - j
            if src >= 0: v += w[j - 1] * R[src]; ic += w[j - 1] * C[src]
            else:
                kk = -src - 1
                if kk < L + H: v[kk] += w[j - 1]
        R.append(v); C.append(ic)
    return np.array(R), np.array(C)


def arima_rows(y, T0, n, ar, ma, diffs, const, window, H, L, refit="drift", th=1.0, log=None):
    """Sliding origin every H points from T0. refit: 'drift' (Alberg & Last: when MAPE_val / MAPE_tr > th),
    'every' origin, or 'once' (fit on the window before T0). Returns rows, reference forecasts, number of fits."""
    model = Arima(ar, ma, diffs, const)
    unit = (lambda v: np.exp(v)) if log else (lambda v: v)
    rows, ref, fits = [], [], 0
    W = mtr = mval = None
    for T in range(T0, n, H):
        h_n = min(H, n - T)
        need = W is None or refit == "every" or (refit == "drift" and mtr > 0 and mval / mtr > th)
        if need:
            t1 = max(0, T - window)
            model.fit(y[t1:T]); fits += 1
            W, c = model.ar_inf(L); Rw, Cw = forecast_rows(W, c, H)
            e = model.resid(model.th, y[t1:T])[model.burn:]; act = y[t1 + model.burn:T]
            mtr = float(np.mean(np.abs((unit(act - e) - unit(act)) / unit(act))) * 100)
        k = min(len(Rw[0]), T)
        past = y[T - 1::-1][:k]
        f = Rw[:h_n, :k] @ past + Cw[:h_n]
        mval = float(np.mean(np.abs((unit(f) - unit(y[T:T + h_n])) / unit(y[T:T + h_n]))) * 100)
        ref += list(f)
        for h in range(h_n):
            rows.append((float(Cw[h]), [(kk + 1 + h, float(Rw[h][kk])) for kk in range(k) if abs(Rw[h][kk]) > 1e-9]))
    return rows, np.array(ref), fits


def rows_job(rows, rowth=None):
    K = max(len(r[1]) for r in rows)
    out = [f"ROWS {K}"]
    for c, r in rows:
        r = r + [(r[0][0], 0.0)] * (K - len(r))
        out.append(f"{c:.12g} " + " ".join(f"{l} {v:.12g}" for l, v in r))
    if rowth is not None: out.append(f"ROWTH {len(rowth)} " + " ".join(f"{v:.12g}" for v in rowth))
    return out


# ---------------------------------------------------------------- Protocol IV: daily-peak SVR
def svr_daily(hourly, dates, ntest, months, holiday_days, eps=0.1, scale=1.0):
    """hourly: (days*24, clients) readings; dates: date per day. Chen, Chang & Lin: daily max of the total,
    7 previous maxima scaled to [0,1] + 6 weekday bits + holiday bit, RBF eps-SVR, recursive forecast.
    C, gamma by recursive prediction of the last validation block of ntest days before the test block."""
    from sklearn.svm import SVR
    nd = len(dates)
    dmax = hourly.sum(axis=1).reshape(nd, 24).max(axis=1)
    test0 = nd - ntest
    cal = lambda d, hol=True: [1.0 if d.weekday() == i else 0.0 for i in range(6)] + [1.0 if hol and d in holiday_days else 0.0]
    train = [d for d in range(7, test0) if dates[d].month in months]
    lo, hi = dmax[train].min(), dmax[train].max(); s = (dmax - lo) / (hi - lo)
    feats = lambda ds: (np.array([list(s[d - 7:d][::-1]) + cal(dates[d]) for d in ds]), s[ds])

    def recursive(m, first):
        v = list(s[:first])
        for d in range(first, first + ntest):
            v.append(float(m.predict([v[d - 7:d][::-1] + cal(dates[d], False)])[0]))
        return np.array(v[first:]) * (hi - lo) + lo
    val0 = test0 - 365 if test0 - 365 >= 14 else test0 - ntest
    fit_days = [d for d in train if not (val0 <= d < val0 + ntest)]
    best = None
    for lc in range(-2, 9):
        for lg in range(-8, 3):
            F, t = feats(fit_days)
            m = SVR(kernel="rbf", C=2.0 ** lc, gamma=2.0 ** lg, epsilon=eps).fit(F, t)
            e = float(np.mean(np.abs(recursive(m, val0) - dmax[val0:val0 + ntest]) / dmax[val0:val0 + ntest]))
            if best is None or e < best[0]: best = (e, 2.0 ** lc, 2.0 ** lg)
    F, t = feats(train)
    m = SVR(kernel="rbf", C=best[1], gamma=best[2], epsilon=eps).fit(F, t)
    ref = recursive(m, test0)
    lines = [f"SCALE {scale:g}", f"NORM {lo:.6f} {hi:.6f}", f"SVR {best[2]:.9g} {m.intercept_[0]:.9g} {len(m.dual_coef_[0])}"]
    lines += [f"{a:.9g} " + " ".join(f"{x:.9g}" for x in v) for a, v in zip(m.dual_coef_[0], m.support_vectors_)]
    lines += [f"CAL {ntest}"] + [" ".join(str(int(b)) for b in cal(dates[d], False)) for d in range(test0, nd)]
    lines += [f"HIST {hourly.shape[1]} 168"] + [" ".join(str(int(round(v * scale))) for v in hourly[(test0 - 7) * 24:test0 * 24, c]) for c in range(hourly.shape[1])]
    info = f"C={best[1]:g}, gamma={best[2]:g}, {len(m.dual_coef_[0])} support vectors, {len(train)} training days"
    return lines, ref, dmax[test0:], info
