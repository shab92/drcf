#!/usr/bin/env python3
"""Protocol IV (Table A1): ML load forecasting with the SVR model of Chen, Chang & Lin (IEEE TPWRS 19(4), 2004),
the EUNITE 2001 winner, on ElectricityLoadDiagrams20112014 (Trindade, UCI), hourly 321-client version of Lai et al.

The paper's model, applied to the system load (sum of all clients):
  - target: daily maximum load; predict the next 31 days recursively (each prediction feeds the next day)
  - 14 attributes: maxima of the previous 7 days (scaled linearly to [0,1]) + 6 weekday bits (Mon..Sat, Sunday = all
    zero) + 1 holiday bit; no temperature (the paper found it unhelpful)
  - training data: the "winter" months only (Jan-Mar, Oct-Dec)
  - RBF kernel, LIBSVM epsilon-SVR (epsilon = 0.1, the LIBSVM default); C and gamma chosen on a January validation
    month with recursive prediction (paper: January 1997 and 1998; here January 2013)
  - holiday flags are removed in the forecast month (the paper did this for January 1999)
Test: January 2014, trained on the winter months of 2012-2013 (Portuguese national holidays from `holidays`).

Model owner side, in plaintext. Writes the model, the 321 clients' hourly readings for the 7 days before the test
month (what the publishers send), the actual daily maxima and the plaintext reference forecast.
Usage: python3 scripts/train_protocol4.py electricity.txt.gz model_p4.txt"""
import sys, gzip, datetime as dt
import numpy as np, holidays
from sklearn.svm import SVR

src, out = sys.argv[1], sys.argv[2]
X = np.loadtxt(gzip.open(src), delimiter=",")              # 26304 h x 321 clients, from 2012-01-01
day0 = dt.date(2012, 1, 1)
nd = X.shape[0] // 24
total = X.sum(axis=1)
dmax = total[:nd * 24].reshape(nd, 24).max(axis=1)
dates = [day0 + dt.timedelta(d) for d in range(nd)]
hol = holidays.Portugal(years=[2012, 2013, 2014])
winter = lambda d: d.month in (1, 2, 3, 10, 11, 12)

def idx(y, m, d): return (dt.date(y, m, d) - day0).days
def cal(d, holiday=True):
    v = [0.0] * 7
    if d.weekday() < 6: v[d.weekday()] = 1.0               # Mon..Sat; Sunday all zero
    if holiday and d in hol: v[6] = 1.0
    return v

def dataset(days, lo, hi):
    s = (dmax - lo) / (hi - lo)
    F = [list(s[d - 7:d][::-1]) + cal(dates[d]) for d in days]   # previous day first
    return np.array(F), s[days]

def recursive(model, first, n, lo, hi):
    s = list((dmax[:first] - lo) / (hi - lo))
    for d in range(first, first + n):
        f = s[d - 7:d][::-1] + cal(dates[d], holiday=False)
        s.append(float(model.predict([f])[0]))
    return np.array(s[first:]) * (hi - lo) + lo

def mape(p, a): return float(np.mean(np.abs((p - a) / a)) * 100)

test0, ntest = idx(2014, 1, 1), 31
train_days = [d for d in range(7, test0) if winter(dates[d])]
lo, hi = dmax[train_days].min(), dmax[train_days].max()

# C, gamma on January 2013 (recursive), trained on the other winter days before 2014
val0 = idx(2013, 1, 1)
fit_days = [d for d in train_days if not (val0 <= d < val0 + 31)]
best = None
for lc in range(-2, 9):
    for lg in range(-8, 3):
        Fv, yv = dataset(fit_days, lo, hi)
        m = SVR(kernel="rbf", C=2.0 ** lc, gamma=2.0 ** lg, epsilon=0.1).fit(Fv, yv)
        e = mape(recursive(m, val0, 31, lo, hi), dmax[val0:val0 + 31])
        if best is None or e < best[0]: best = (e, 2.0 ** lc, 2.0 ** lg)
print(f"validation (Jan 2013): MAPE {best[0]:.3f}%  C={best[1]:g} gamma={best[2]:g}")

F, y = dataset(train_days, lo, hi)
model = SVR(kernel="rbf", C=best[1], gamma=best[2], epsilon=0.1).fit(F, y)
ref = recursive(model, test0, ntest, lo, hi)
act = dmax[test0:test0 + ntest]
print(f"test (Jan 2014): {len(model.support_)} SVs, plaintext MAPE {mape(ref, act):.3f}%  (paper, EUNITE Jan 1999: 1.95%)")

with open(out, "w") as f:
    f.write(f"NORM {lo:.6f} {hi:.6f}\n")
    sv, al = model.support_vectors_, model.dual_coef_[0]
    f.write(f"SVR {best[2]:.9g} {model.intercept_[0]:.9g} {len(al)}\n")
    for a, v in zip(al, sv): f.write(f"{a:.9g} " + " ".join(f"{x:.9g}" for x in v) + "\n")
    f.write(f"CAL {ntest}\n")
    for d in range(test0, test0 + ntest): f.write(" ".join(str(int(b)) for b in cal(dates[d], holiday=False)) + "\n")
    f.write(f"HIST {X.shape[1]} 168\n")                    # clients x hours, the 7 days before the test month
    for c in range(X.shape[1]): f.write(" ".join(str(int(round(v))) for v in X[(test0 - 7) * 24:test0 * 24, c]) + "\n")
    f.write(f"ACT {ntest}\n" + "\n".join(f"{v:.1f}" for v in act) + "\n")
    f.write(f"REF {ntest}\n" + "\n".join(f"{v:.3f}" for v in ref) + "\n")
