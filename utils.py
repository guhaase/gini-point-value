"""Funções do modelo binormal calibrado e dos canais de custo (V06).

Convenção: escore s alto = mais risco. Bons ~ N(0,1), maus ~ N(d,1).
AUC = Phi(d/sqrt2); KS = 2 Phi(d/2) - 1; Gini = 2 AUC - 1.
PD calibrada: logit PD(s) = logit(pi) + d s - d^2/2.
"""
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

GRID = np.linspace(-9.0, 12.0, 120_001)
DS = GRID[1] - GRID[0]


# ---------------------------------------------------------------- métricas
def d_from_auc(auc):
    return np.sqrt(2.0) * norm.ppf(auc)

def auc_from_d(d):
    return norm.cdf(d / np.sqrt(2.0))

def ks_from_d(d):
    return 2.0 * norm.cdf(d / 2.0) - 1.0

def d_from_ks(ks):
    return 2.0 * norm.ppf((ks + 1.0) / 2.0)


# ------------------------------------------------------- modelo calibrado
def densities(d, pi):
    fG = norm.pdf(GRID)
    fB = norm.pdf(GRID - d)
    mix = (1 - pi) * fG + pi * fB
    return fG, fB, mix

def pd_of_s(d, pi, s=GRID):
    z = np.log(pi / (1 - pi)) + d * s - 0.5 * d * d
    return 1.0 / (1.0 + np.exp(-z))


# ------------------------------------------------------------ capital IRB
def irb_corr(pd, segment):
    pd = np.asarray(pd)
    if segment == "mortgage":
        return np.full_like(pd, 0.15, dtype=float)
    if segment == "revolving":
        return np.full_like(pd, 0.04, dtype=float)
    if segment == "other_retail":      # Basileia: R cai de 0,16 (PD->0) para 0,03
        w = (1 - np.exp(-35 * pd)) / (1 - np.exp(-35))
        return 0.03 * w + 0.16 * (1 - w)
    raise ValueError(segment)

def irb_k(pd, lgd, segment="other_retail"):
    """Requerimento de capital K (fração da EAD), varejo, sem ajuste de prazo."""
    pd = np.clip(np.asarray(pd, dtype=float), 1e-7, 1 - 1e-7)
    R = irb_corr(pd, segment)
    cond = norm.cdf((norm.ppf(pd) + np.sqrt(R) * norm.ppf(0.999)) / np.sqrt(1 - R))
    return lgd * (cond - pd)


# --------------------------------------------------------------- canais
def channel_selection(d, pi, lgd, accept):
    """C1: taxa de aprovação fixa; devolve bad rate e EL (fração) dos aprovados."""
    f = lambda c: (1 - pi) * norm.cdf(c) + pi * norm.cdf(c - d) - accept
    c = brentq(f, -20, 20)
    br = pi * norm.cdf(c - d) / accept
    return {"cutoff": c, "bad_rate": br, "el": br * lgd}

def channel_optimal_cutoff(d, pi, lgd, margin):
    """C1': regra ótima de Stein (aprova se PD < m/(m+LGD)); lucro por unidade de aplicante."""
    p_star = margin / (margin + lgd)
    _, _, mix = densities(d, pi)
    pd = pd_of_s(d, pi)
    acc = pd < p_star
    profit = (((1 - pd) * margin - pd * lgd) * mix * acc).sum() * DS
    rate = (mix * acc).sum() * DS
    return {"profit": profit, "accept": rate, "p_star": p_star}

def channel_capital(d, pi, lgd, segment="other_retail"):
    """C3: capital médio da carteira inteira com PD calibrada."""
    _, _, mix = densities(d, pi)
    return (irb_k(pd_of_s(d, pi), lgd, segment) * mix).sum() * DS

def lifetime_pd(pd12, horizon_years):
    return 1.0 - (1.0 - pd12) ** horizon_years

def channel_provision(d, pi, lgd, threshold, horizon_years):
    """C4: ECL IFRS 9 com estágio 2 por limiar absoluto de PD de 12 meses."""
    _, _, mix = densities(d, pi)
    pd = pd_of_s(d, pi)
    st2 = pd > threshold
    ecl = np.where(st2, lifetime_pd(pd, horizon_years), pd) * lgd
    return {"ecl": (ecl * mix).sum() * DS, "share_stage2": (mix * st2).sum() * DS}


# ------------------------------------------- formas fechadas (Proposição 3)
def jacobians(d):
    """d(delta)/dX para X = AUC, Gini, KS."""
    from scipy.stats import norm as _n
    return {"AUC": np.sqrt(2) / _n.pdf(d / np.sqrt(2)),
            "Gini": 1 / (np.sqrt(2) * _n.pdf(d / np.sqrt(2))),
            "KS": 1 / _n.pdf(d / 2)}

def dbr_ddelta(d, pi, accept):
    c = channel_selection(d, pi, 1.0, accept)["cutoff"]
    f = (1 - pi) * norm.pdf(c) + pi * norm.pdf(c - d)
    p = pi * norm.pdf(c - d) / f
    return -f * p * (1 - p) / accept, p

def dprofit_ddelta(d, pi, lgd, margin):
    ps = margin / (margin + lgd)
    cs = (np.log(ps / (1 - ps)) - np.log(pi / (1 - pi)) + 0.5 * d * d) / d
    f = (1 - pi) * norm.pdf(cs) + pi * norm.pdf(cs - d)
    return lgd * ps * f
