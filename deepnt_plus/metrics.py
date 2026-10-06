"""Regression and calibration metrics. RTT metrics are always reported in milliseconds."""
import numpy as np
from scipy.stats import norm as _norm

LEVELS = (0.5, 0.8, 0.9, 0.95)


def regression_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = y_pred - y_true
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return {
        'n': int(len(y_true)),
        'MAE': float(np.mean(np.abs(err))),
        'RMSE': float(np.sqrt(np.mean(err ** 2))),
        'R2': float(1 - np.sum(err ** 2) / ss_tot) if ss_tot > 0 else float('nan'),
        'MedAE': float(np.median(np.abs(err))),
        'sMAPE': float(np.mean(2 * np.abs(err) / (np.abs(y_true) + np.abs(y_pred) + 1e-9)) * 100),
        'MdAPE': float(np.median(np.abs(err) / np.maximum(np.abs(y_true), 1e-9)) * 100),
        'MAE_log': float(np.mean(np.abs(np.log1p(np.clip(y_pred, 0, None)) - np.log1p(y_true)))),
    }


def calibration_metrics(y_true_ms, mu_log, sigma_log):
    """Predictive distribution: log1p(RTT) ~ N(mu_log, sigma_log^2). Returns mean NLL (log space), sharpness,
    empirical coverage of central intervals, and the mean absolute coverage error (regression ECE)."""
    z = np.log1p(np.asarray(y_true_ms, float))
    mu, sd = np.asarray(mu_log, float), np.maximum(np.asarray(sigma_log, float), 1e-6)
    out = {'NLL': float(np.mean(0.5 * np.log(2 * np.pi * sd ** 2) + (z - mu) ** 2 / (2 * sd ** 2))),
           'sharpness_log': float(sd.mean())}
    gaps = []
    for level in LEVELS:
        k = _norm.ppf(0.5 + level / 2)
        cov = float(np.mean(np.abs(z - mu) <= k * sd))
        out[f'coverage_{int(level * 100)}'] = cov
        gaps.append(abs(cov - level))
    out['ECE_reg'] = float(np.mean(gaps))
    return out


def reliability_curve(y_true_ms, mu_log, sigma_log, levels=np.linspace(0.05, 0.95, 19)):
    z = np.log1p(np.asarray(y_true_ms, float))
    mu, sd = np.asarray(mu_log, float), np.maximum(np.asarray(sigma_log, float), 1e-6)
    return levels, np.array([np.mean(np.abs(z - mu) <= _norm.ppf(0.5 + l / 2) * sd) for l in levels])
