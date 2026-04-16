import numpy as np
from scipy.stats import chi2
from math import sqrt, exp
from scipy.special import erf


def standard_normal_cdf(x):
    return 0.5 * (1.0 + erf(np.asarray(x) / sqrt(2.0)))


def normal_cdf(x, mu=1.0, sigma=1.0):
    return 0.5 * (1.0 + erf((np.asarray(x) - mu) / (sigma * sqrt(2.0))))


def kolmogorov_smirnov(x_vals):
    x = np.asarray(x_vals, dtype=float)
    N = x.size
    mu = np.mean(x)
    sigma = np.sqrt((1.0 / (N - 1)) * np.sum((x - mu) ** 2))
    if sigma == 0:
        return 0.0, 1.0

    sort_ind = np.argsort(x)
    Y = (x[sort_ind] - mu) / sigma

    f0 = np.arange(N) / N
    fn = (np.arange(N) + 1.0) / N
    ff = standard_normal_cdf(Y)
    D = np.max([np.max(np.abs(f0 - ff)), np.max(np.abs(fn - ff))])

    # prob_ks series
    eps1 = 0.001
    eps2 = 1e-8
    en = np.sqrt(N)
    lam = (en + 0.12 + 0.11 / en) * D

    a2 = -2.0 * lam * lam
    probks = 0.0
    termbf = 0.0
    sign = 1.0
    converged = False
    for j in range(1, 101):
        term = sign * 2.0 * exp(a2 * (j ** 2))
        probks += term
        if (abs(term) <= eps1 * termbf) or (abs(term) <= eps2 * probks):
            converged = True
            break
        sign = -sign
        termbf = abs(term)

    if not converged:
        probks = 1.0

    return D, probks


def anderson_darling(x_vals, signif_level=0.95, adjusted=False, alt_2d_interp=False):
    x = np.asarray(x_vals, dtype=float)
    N = x.size
    mu = np.mean(x)
    sigma = np.sqrt((1.0 / (N - 1)) * np.sum((x - mu) ** 2))
    if sigma == 0:
        if adjusted and N >= 8:
            crit_val = np.interp(
                signif_level,
                [0.90, 0.95, 0.975, 0.99, 0.995],
                [0.631, 0.752, 0.873, 1.035, 1.159],
            )
        else:
            crit_val = 0.0
        return 0.0, crit_val

    sort_ind = np.argsort(x)
    Y = (x[sort_ind] - mu) / sigma
    norm_cdf = standard_normal_cdf(Y)

    i_arr = np.arange(1, N + 1)
    A_sqrd = -N - (1.0 / N) * np.sum((2 * i_arr - 1) * (np.log(norm_cdf) + np.log(1 - norm_cdf[::-1])))

    if adjusted and N >= 8:
        A_sqrd = A_sqrd * (1 + 0.75 / N + 2.25 / (N ** 2))
        crit_val = np.interp(signif_level, [0.90, 0.95, 0.975, 0.99, 0.995], [0.631, 0.752, 0.873, 1.035, 1.159])
        return A_sqrd, crit_val

    # Stephens 1974 table interpolation
    N_coords = np.array([10, 20, 50, 100, 1e37], dtype=float)
    S_coords = np.array([0.85, 0.90, 0.95, 0.975, 0.99], dtype=float)
    crits = np.array(
        [
            [0.514, 0.578, 0.683, 0.779, 0.926],
            [0.528, 0.591, 0.704, 0.815, 0.969],
            [0.546, 0.616, 0.735, 0.861, 1.021],
            [0.559, 0.631, 0.754, 0.884, 1.047],
            [0.576, 0.656, 0.787, 0.918, 1.092],
        ],
        dtype=float,
    ).T  # shape (5,5) over S then N

    if alt_2d_interp:
        interp_crit_table = np.array([np.interp(N, N_coords, crits[s, :]) for s in range(5)])
        crit_val = np.interp(signif_level, S_coords, interp_crit_table)
    else:
        # simple bilinear interpolation via two 1D steps (more stable than trigrid)
        interp_crit_table = np.array([np.interp(N, N_coords, crits[s, :]) for s in range(5)])
        crit_val = np.interp(signif_level, S_coords, interp_crit_table)

    return A_sqrd, crit_val


def ljung_box(residuals, signif_level=0.95, num_fit_params=0, max_lag=None):
    x = np.asarray(residuals, dtype=float)
    N = x.size
    half_n = int(np.floor(N / 2.0))

    if max_lag is None:
        max_lag = half_n
    if max_lag > half_n:
        max_lag = half_n

    lags = np.arange(1, max_lag + 1)
    mu = np.mean(x)
    mean_diff = x - mu
    AR = np.zeros(max_lag, dtype=float)
    for k in range(max_lag):
        lag = lags[k]
        AR[k] = np.sum(mean_diff[: N - lag] * mean_diff[lag:])

    denom = np.sum(mean_diff ** 2)
    if denom != 0:
        AR = AR / denom

    Q = N * (N + 2) * np.sum(AR ** 2 / (N - lags))

    if max_lag >= num_fit_params:
        crit_val = chi2.ppf(1.0 - signif_level, max_lag - num_fit_params)
    else:
        crit_val = 0.0

    return Q, crit_val


def test_residuals(residuals, signif_level=0.95, num_fit_params=0, max_lag=None):
    KS_stat, KS_prob = kolmogorov_smirnov(residuals)
    AD_stat, AD_crit = anderson_darling(residuals, signif_level=signif_level)
    LB_stat, LB_chisqrd = ljung_box(residuals, signif_level=signif_level, num_fit_params=num_fit_params, max_lag=max_lag)
    return {
        "signif_level": signif_level,
        "KS_stat": KS_stat,
        "KS_prob": KS_prob,
        "AD_stat": AD_stat,
        "AD_crit": AD_crit,
        "LB_stat": LB_stat,
        "LB_chisqrd": LB_chisqrd,
    }


def signif_noise_spec(data_in, p, num_fft, color="white", bonferroni=False):
    data = np.asarray(data_in, dtype=float)
    sigma = np.std(data, ddof=0)
    N = data.size
    dof = 2.0

    a = 1.0 - p
    num_for_scaling = min(N / 2.0, num_fft)
    if bonferroni:
        a_all = a
        a = 1.0 - (1.0 - a_all) ** (1.0 / float(num_for_scaling))

    chi_sq_val = chi2.ppf(a, dof)

    if color == "white":
        P_k = np.ones(num_fft, dtype=float)
        alpha = 0.0
    else:
        mu = np.mean(data)
        mean_diff = data - mu
        ar_1 = (1.0 / np.sum(mean_diff ** 2)) * np.sum(mean_diff[:-1] * mean_diff[1:])
        ar_2 = (1.0 / np.sum(mean_diff ** 2)) * np.sum(mean_diff[:-2] * mean_diff[2:])
        alpha = (ar_1 + np.sqrt(abs(ar_2))) / 2.0
        k = np.arange(num_fft)
        P_k = (1 - alpha ** 2) / (1 + alpha ** 2 - 2 * alpha * np.cos(2 * np.pi * k / N))

    signif = (sigma ** 2) * chi_sq_val / (N * dof) * P_k
    return signif


def signif_noise_spec_bootstrap(data_in, p, num_fft, color="white"):
    data = np.asarray(data_in, dtype=float)
    # data shape: (N, N_boot)
    N = data.shape[0]
    N_boot = data.shape[1]

    s = np.std(data, axis=0)
    s = np.broadcast_to(s, (num_fft, N_boot))

    dof = 2.0
    chi = chi2.ppf(1.0 - p, dof)

    if color == "white":
        P_k = np.ones((num_fft, N_boot), dtype=float)
        alpha = 0.0
    else:
        mu = np.mean(data, axis=0)
        mu = np.broadcast_to(mu, (N, N_boot))
        sigma = np.std(data, axis=0)
        data_i_plus_1 = np.roll(data, -1, axis=0)
        data_i_plus_2 = np.roll(data, -2, axis=0)
        a1 = (1.0 / ((N - 1) * sigma ** 2)) * np.sum(((data - mu) * (data_i_plus_1 - mu))[:-1, :], axis=0)
        a2 = (1.0 / ((N - 2) * sigma ** 2)) * np.sum(((data - mu) * (data_i_plus_2 - mu))[:-2, :], axis=0)
        a1 = np.broadcast_to(a1, (num_fft, N_boot))
        a2 = np.broadcast_to(a2, (num_fft, N_boot))
        alpha = (a1 + np.sqrt(abs(a2))) / 2.0
        k = np.broadcast_to(np.arange(num_fft)[:, None], (num_fft, N_boot))
        P_k = (1 - alpha ** 2) / (1 + alpha ** 2 - 2 * alpha * np.cos(2 * np.pi * k / N))

    signif = (s ** 2) * chi / ((N / 2.0) * dof) * P_k
    signif = np.mean(signif, axis=1)
    return signif
