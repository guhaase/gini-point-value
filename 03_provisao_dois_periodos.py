"""Canal C4 em dois períodos: estágio 2 por piora relativa da PD (IFRS 9 / CMN 4.966).

Modelo de fator latente (Merton/Vasicek de um tomador):
  X0 ~ N(0,1) qualidade na concessão; X1 = rho*X0 + sqrt(1-rho^2)*eta na data-base.
  Default nos 12 meses após a data-base: D = 1{X1 < Phi^-1(pi)}.
  Escore do modelo em t: s_t = sqrt(r)*X_t + sqrt(1-r)*e_t, com erro persistente
  e1 = kappa*e0 + sqrt(1-kappa^2)*u (kappa = persistência do erro do modelo). O poder cresce com r.
  PD calibrada: p(s) = Phi((c - sqrt(r) s) / sqrt(1-r)), c = Phi^-1(pi).
  Reduzir r é somar ruído independente ao escore (garbling), como na Proposição 2.

Estágio 2 se p(s1)/p(s0) > MULT (piora relativa) ou p(s1) > ABS_FLOOR (opcional).
ECL = LGD * [p1 no estágio 1; PD de vida toda 1-(1-p1)^T no estágio 2].

Saídas: latex/tables/Table_3.csv, latex/figures/Figure_3.pdf
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm, rankdata

import config as C

N = 1_000_000
R_GRID = np.linspace(0.05, 0.80, 31)
RHO_LIST = [0.80, 0.90, 0.95]
MULT_LIST = [2.0, 3.0]
KAPPA_LIST = [0.0, 0.5, 0.9]
BASE = {"rho": 0.90, "mult": 3.0, "kappa": 0.9}


def auc(score, d):
    rk = rankdata(score)
    n1 = d.sum(); n0 = len(d) - n1
    return (rk[d == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def run(r, rho, mult, kappa, pi=C.PI, lgd=C.LGD, T=C.HORIZON_YEARS, seed=C.RANDOM_SEED):
    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal(N)
    x1 = rho * x0 + np.sqrt(1 - rho**2) * rng.standard_normal(N)
    c = norm.ppf(pi)
    d = (x1 < c).astype(int)
    e0 = rng.standard_normal(N)
    e1 = kappa * e0 + np.sqrt(1 - kappa**2) * rng.standard_normal(N)
    s0 = np.sqrt(r) * x0 + np.sqrt(1 - r) * e0
    s1 = np.sqrt(r) * x1 + np.sqrt(1 - r) * e1
    pd_ = lambda s: norm.cdf((c - np.sqrt(r) * s) / np.sqrt(1 - r))
    p0, p1 = pd_(s0), pd_(s1)
    st2 = p1 / p0 > mult
    life = 1 - (1 - p1) ** T
    ecl = lgd * np.where(st2, life, p1)
    # comparação: limiar absoluto (versão estática)
    st2_abs = p1 > 2 * pi
    ecl_abs = lgd * np.where(st2_abs, life, p1)
    return {
        "r": r, "rho": rho, "mult": mult, "kappa": kappa, "auc": auc(-s1, d),  # s alto = melhor
        "el12": lgd * p1.mean(), "ecl": ecl.mean(), "stage2": st2.mean(),
        "ecl_abs": ecl_abs.mean(), "stage2_abs": st2_abs.mean(),
        "realized_dr": d.mean(),
    }


def main():
    combos = {(BASE["rho"], BASE["mult"], BASE["kappa"])}
    combos |= {(rho, BASE["mult"], BASE["kappa"]) for rho in RHO_LIST}
    combos |= {(BASE["rho"], m, BASE["kappa"]) for m in MULT_LIST}
    combos |= {(BASE["rho"], BASE["mult"], k) for k in KAPPA_LIST}
    rows = [run(r, *cb) for cb in sorted(combos) for r in R_GRID]
    df = pd.DataFrame(rows)
    df.to_csv(C.DATA_DIR / "c4_dois_periodos.csv", index=False)

    # valor do ponto: inclinação local de ECL em AUC (bps por 0,01)
    out = []
    for (rho, m, k), g in df.groupby(["rho", "mult", "kappa"]):
        g = g.sort_values("auc")
        slope = np.gradient(g.ecl.values, g.auc.values) * 0.01 * 1e4
        slope_abs = np.gradient(g.ecl_abs.values, g.auc.values) * 0.01 * 1e4
        a = g.auc.values
        for a_target in C.AUC_LEVELS_TABLE:
            if not a.min() <= a_target <= a.max():
                continue
            ip = lambda y: float(np.interp(a_target, a, y))
            out.append({"rho": rho, "mult": m, "kappa": k, "AUC": a_target,
                        "ECL (bps)": round(ip(g.ecl.values) * 1e4, 1),
                        "Estágio 2 (%)": round(100 * ip(g.stage2.values), 1),
                        "V ECL relativo (bps/ponto)": round(ip(slope), 2),
                        "V ECL absoluto (bps/ponto)": round(ip(slope_abs), 2)})
    t3 = pd.DataFrame(out)
    t3.to_csv(C.TABLES_DIR / "Table_3.csv", index=False)
    print(t3.to_string(index=False))

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ls = {0.0: ":", 0.5: "--", 0.9: "-"}
    for k in KAPPA_LIST:
        g = df[(df.rho == BASE["rho"]) & (df.mult == BASE["mult"]) & (df.kappa == k)].sort_values("auc")
        ax[0].plot(g.auc, g.ecl * 1e4, "k" + ls[k], label=f"kappa = {k}")
        ax[1].plot(g.auc, 100 * g.stage2, "k" + ls[k], label=f"kappa = {k}")
    ax[0].plot(g.auc, g.el12 * 1e4, color="grey", lw=0.8, label="12-month EL")
    ax[0].set_title("(a) IFRS 9 ECL, relative SICR (3x)"); ax[0].set_ylabel("bps of EAD")
    ax[1].set_title("(b) Share in stage 2"); ax[1].set_ylabel("%")
    for a_ in ax:
        a_.set_xlabel("AUC"); a_.grid(alpha=0.3); a_.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(C.FIGURES_DIR / "Figure_3.pdf"); plt.close(fig)


if __name__ == "__main__":
    main()
