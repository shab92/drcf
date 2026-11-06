// Prophet's additive model (Taylor & Letham, "Forecasting at scale", 2018) as a linear least-squares fit,
// configured as in Jaiswal et al. (AICCC 2021) / their published notebook:
//   daily_seasonality = True (order 4), yearly_seasonality = True (order 10), weekly_seasonality = False,
//   add_seasonality('week_ends',  period 7, order 3, condition = weekend),
//   add_seasonality('week_days',  period 7, order 3, condition = weekday),
//   add_country_holidays('AU'), additive mode, 25 changepoints in the first 80% of the history.
// MAP estimate with Prophet's default prior scales (k, m: 5; seasonality and holidays: 10; changepoints: 0.05).
// Prophet puts a Laplace prior on the changepoint deltas; here it is a Gaussian with the same scale, which makes
// the whole fit LINEAR in y. That is what lets each DRCF broker fit its own share locally (Table A1, Protocol II):
//   theta_i = G^{-1} A^T y_i,  yhat_i = A theta_i,  and theta_1 + theta_2 = theta(y).
#pragma once
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include "au_holidays.h"

namespace prophet_lite {

inline int64_t days_from_civil(int y, int m, int d) {   // H. Hinnant's algorithm
    y -= m <= 2; const int64_t era = (y >= 0 ? y : y - 399) / 400; const unsigned yoe = (unsigned)(y - era * 400);
    const unsigned doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1; const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return era * 146097 + (int64_t)doe - 719468;
}

struct Stamp { int y, mo, d, h, mi, dow; };   // dow: 0 = Monday ... 6 = Sunday

struct Config {
    int n_changepoints = 25; double changepoint_range = 0.8;
    int daily_order = 4, yearly_order = 10, weekly_cond_order = 3;
    double prior_trend = 5.0, prior_cp = 0.05, prior_season = 10.0, prior_holiday = 10.0;
    double y_scale = 2000.0;   // public scale standing in for Prophet's max|y|, which would depend on the data
    double noise_var = 0.01;   // observation variance in scaled units for the MAP ridge weights
    bool au_holidays = true;   // Australian national holidays (Jaiswal et al.); off for other datasets
};

struct Model {
    Config cfg; size_t N = 0; int P = 0;
    std::vector<double> A;       // N x P design matrix (public: depends only on timestamps)
    std::vector<double> Lc;      // Cholesky factor of G = A^T A + Lambda
    std::vector<double> lambda;

    void build(const std::vector<Stamp>& ts) {
        N = ts.size();
        std::vector<double> days(N);
        for (size_t i = 0; i < N; i++) days[i] = days_from_civil(ts[i].y, ts[i].mo, ts[i].d) + (ts[i].h * 60 + ts[i].mi) / 1440.0;
        double t0 = days.front(), t1 = days.back();
        std::vector<double> cps;
        size_t last = (size_t)std::floor(cfg.changepoint_range * N);
        for (int j = 1; j <= cfg.n_changepoints; j++) cps.push_back((days[(size_t)std::llround((double)j * last / (cfg.n_changepoints + 1))] - t0) / (t1 - t0));
        int H = cfg.au_holidays ? (int)au_holidays::names.size() : 0;
        P = 2 + cfg.n_changepoints + 2 * cfg.daily_order + 2 * cfg.yearly_order + 4 * cfg.weekly_cond_order + H;
        A.assign(N * (size_t)P, 0.0); lambda.assign(P, 0.0);
        const double tw = 2 * M_PI;
        for (size_t i = 0; i < N; i++) {
            double* r = &A[i * P]; int c = 0;
            double t = (days[i] - t0) / (t1 - t0);
            r[c++] = 1.0; r[c++] = t;                                                   // m, k
            for (double s : cps) r[c++] = t > s ? t - s : 0.0;                          // changepoint deltas
            for (int n = 1; n <= cfg.daily_order; n++) { r[c++] = std::cos(tw * n * days[i]); r[c++] = std::sin(tw * n * days[i]); }
            for (int n = 1; n <= cfg.yearly_order; n++) { r[c++] = std::cos(tw * n * days[i] / 365.25); r[c++] = std::sin(tw * n * days[i] / 365.25); }
            bool we = ts[i].dow >= 5;
            for (int cond = 0; cond < 2; cond++)                                         // week_ends, week_days
                for (int n = 1; n <= cfg.weekly_cond_order; n++) {
                    double on = (cond == 0) == we ? 1.0 : 0.0;
                    r[c++] = on * std::cos(tw * n * days[i] / 7.0); r[c++] = on * std::sin(tw * n * days[i] / 7.0);
                }
            if (cfg.au_holidays)
                for (auto& hd : au_holidays::days)
                    if (hd.y == ts[i].y && hd.m == ts[i].mo && hd.d == ts[i].d) r[c + hd.name] = 1.0;
        }
        // ridge weights = noise variance / prior variance, per parameter group
        int c = 0;
        lambda[c++] = cfg.noise_var / (cfg.prior_trend * cfg.prior_trend); lambda[c++] = lambda[0];
        for (int j = 0; j < cfg.n_changepoints; j++) lambda[c++] = cfg.noise_var / (cfg.prior_cp * cfg.prior_cp);
        for (int j = 0; j < 2 * cfg.daily_order + 2 * cfg.yearly_order + 4 * cfg.weekly_cond_order; j++) lambda[c++] = cfg.noise_var / (cfg.prior_season * cfg.prior_season);
        for (int j = 0; j < H; j++) lambda[c++] = cfg.noise_var / (cfg.prior_holiday * cfg.prior_holiday);
        // G = A^T A + Lambda, Cholesky
        std::vector<double> G((size_t)P * P, 0.0);
        for (size_t i = 0; i < N; i++) { const double* r = &A[i * P]; for (int a = 0; a < P; a++) if (r[a] != 0) for (int b = 0; b <= a; b++) G[a * P + b] += r[a] * r[b]; }
        for (int a = 0; a < P; a++) { G[a * P + a] += lambda[a]; for (int b = 0; b < a; b++) G[b * P + a] = G[a * P + b]; }
        Lc.assign((size_t)P * P, 0.0);
        for (int j = 0; j < P; j++) {
            double s = G[j * P + j]; for (int k = 0; k < j; k++) s -= Lc[j * P + k] * Lc[j * P + k];
            if (s <= 0) throw std::runtime_error("design matrix not positive definite");
            Lc[j * P + j] = std::sqrt(s);
            for (int i = j + 1; i < P; i++) { double v = G[i * P + j]; for (int k = 0; k < j; k++) v -= Lc[i * P + k] * Lc[j * P + k]; Lc[i * P + j] = v / Lc[j * P + j]; }
        }
    }
    // theta = G^{-1} A^T (y / y_scale): linear in y
    std::vector<double> fit(const std::vector<double>& y) const {
        std::vector<double> b(P, 0.0), z(P), th(P);
        for (size_t i = 0; i < N; i++) { const double* r = &A[i * P]; double yi = y[i] / cfg.y_scale; for (int a = 0; a < P; a++) b[a] += r[a] * yi; }
        for (int i = 0; i < P; i++) { double v = b[i]; for (int k = 0; k < i; k++) v -= Lc[i * P + k] * z[k]; z[i] = v / Lc[i * P + i]; }
        for (int i = P - 1; i >= 0; i--) { double v = z[i]; for (int k = i + 1; k < P; k++) v -= Lc[k * P + i] * th[k]; th[i] = v / Lc[i * P + i]; }
        return th;
    }
    std::vector<double> predict(const std::vector<double>& th) const {
        std::vector<double> yh(N, 0.0);
        for (size_t i = 0; i < N; i++) { const double* r = &A[i * P]; double s = 0; for (int a = 0; a < P; a++) s += r[a] * th[a]; yh[i] = s * cfg.y_scale; }
        return yh;
    }
};

}  // namespace prophet_lite
