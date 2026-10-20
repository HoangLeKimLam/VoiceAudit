# -*- coding: utf-8 -*-
"""Thong ke cho v6 (khong phu thuoc phan con lai cua v6): khoang Wilson, Clopper-Pearson (chinh xac),
McNemar chinh xac, bootstrap theo cum (nguoi noi / series / tep nguon). v6.common xuat lai cac ham nay."""
from __future__ import annotations

import math
from statistics import NormalDist
from typing import Any, Callable, Sequence

import numpy as np

__all__ = ["wilson", "clopper_pearson", "mcnemar", "bootstrap_ci_by_group"]


def _check_kn(k: int, n: int) -> None:
    if int(k) != k or int(n) != n or not 0 <= k <= n:
        raise ValueError(f"can so nguyen 0 <= k <= n, nhan k={k}, n={n}")


def wilson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Khoang Wilson (1-alpha) cho ti le k/n. n=0 -> (0, 1)."""
    _check_kn(k, n)
    if n == 0:
        return 0.0, 1.0
    z = NormalDist().inv_cdf(1 - alpha / 2)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    # mep dung 0 / 1 khi k = 0 / n (tranh sai so tron 1e-17)
    return (0.0 if k == 0 else max(0.0, c - h)), (1.0 if k == n else min(1.0, c + h))


def _binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k), X ~ Bin(n, p), tinh trong khong gian log (on dinh khi n lon)."""
    if k < 0:
        return 0.0
    if k >= n or p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    lp, lq = math.log(p), math.log1p(-p)
    t = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq
         for i in range(k + 1)]
    mx = max(t)
    return min(1.0, math.exp(mx) * sum(math.exp(x - mx) for x in t))


def _bisect(f: Callable[[float], float], target: float, increasing: bool, iters: int = 80) -> float:
    lo, hi = 0.0, 1.0
    for _ in range(iters):
        mid = (lo + hi) / 2
        if (f(mid) < target) == increasing:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _clopper_pearson_exact(k: int, n: int, alpha: float) -> tuple[float, float]:
    """Clopper-Pearson bang chia doi tren CDF nhi thuc (khong can scipy)."""
    lo = 0.0 if k == 0 else _bisect(lambda p: 1 - _binom_cdf(k - 1, n, p), alpha / 2, True)
    hi = 1.0 if k == n else _bisect(lambda p: _binom_cdf(k, n, p), alpha / 2, False)
    return lo, hi


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Khoang chinh xac Clopper-Pearson (1-alpha). scipy.stats.beta neu co, neu khong thi chia doi."""
    _check_kn(k, n)
    if n == 0:
        return 0.0, 1.0
    try:
        from scipy.stats import beta
    except ImportError:
        return _clopper_pearson_exact(int(k), int(n), alpha)
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def mcnemar(b: int, c: int) -> float:
    """p hai phia McNemar chinh xac: b, c = so cap bat dong (A dung B sai / A sai B dung); Bin(b+c, 0,5)."""
    if int(b) != b or int(c) != c or b < 0 or c < 0:
        raise ValueError(f"b, c phai la so nguyen >= 0, nhan {b}, {c}")
    n, m = int(b + c), int(min(b, c))
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(m + 1))     # so nguyen chinh xac
    return min(1.0, 2 * tail / 2 ** n)


def bootstrap_ci_by_group(values: Any, groups: Sequence[Any], fn: Callable[[Any], float], n: int = 2000,
                          seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    """Bootstrap theo CUM (nguoi noi / series / tep nguon): boc lai nhom co hoan lai, ghep moi dong cua nhom, tinh fn.
    values: mang (N, ...) — vd cot [dung_A, dung_B] cho so sanh cap; fn nhan values con. Tra (uoc luong, lo, hi)
    phan vi (1-alpha). < 2 nhom hoac fn toan NaN -> lo = hi = NaN."""
    v = np.asarray(values)
    g = np.asarray(groups)
    if len(v) != len(g):
        raise ValueError(f"values ({len(v)}) va groups ({len(g)}) khac do dai")
    est = float(fn(v))
    _, inv = np.unique(g, return_inverse=True)
    idx = [np.flatnonzero(inv == j) for j in range(inv.max() + 1)] if len(g) else []
    if len(idx) < 2:
        return est, math.nan, math.nan
    rng = np.random.default_rng(seed)
    stats = np.array([float(fn(v[np.concatenate([idx[j] for j in rng.integers(0, len(idx), len(idx))])]))
                      for _ in range(int(n))])
    stats = stats[np.isfinite(stats)]
    if not len(stats):
        return est, math.nan, math.nan
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return est, float(lo), float(hi)
