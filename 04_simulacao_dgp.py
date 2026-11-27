"""Simulação com o DGP do V01: o valor do ponto sobrevive fora do binormal?

Em cada réplica:
  1. Gera treino e teste a partir do DGP logístico do V01 (sinal amplificado).
  2. Família A: escore = logit verdadeiro + sigma * erro do tomador; recalibração
     de Platt (logística de D no escore) no treino. Família B: logística com
     subconjuntos de variáveis (poder perdido por variável omitida).
  3. No teste: AUC, KS e calibração; custos por canal com a PD do modelo
     para decidir e a PD verdadeira para medir a perda esperada.
  4. Valor do ponto: inclinação local do custo em AUC (família A, ajuste
     quadrático), comparada à forma fechada binormal (Proposição 3).

Saídas: experimentos/data/sim_modelos.csv, latex/tables/Table_4.csv, Table_5.csv (Tabelas 4 e 5),
        Table_S2_suficiencia_auc.csv (apoio ao texto),
        latex/figures/Figure_4.pdf
"""
import sys
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq

import config as C
from utils import (irb_k, lifetime_pd, d_from_auc, dbr_ddelta, dprofit_ddelta,
                   jacobians, channel_capital, channel_provision, channel_selection)

FEATS = list(C.FEATURE_SPEC)


# ------------------------------------------------------------------ dados
def gen_features(rng, n):
    X = {}
    for f, (dist, par) in C.FEATURE_SPEC.items():
        x = rng.normal(par[0], par[1], n) if dist == "normal" else rng.exponential(par[0], n)
        X[f] = np.clip(x, *C.CLIP[f])
    return X


def linear_signal(X):
    return C.SIGNAL_SCALE * sum(C.BETA_V01[f] * X[f] for f in FEATS)


def calibrate_intercept(seed=C.RANDOM_SEED, n=1_000_000):
    eta = linear_signal(gen_features(np.random.default_rng(seed), n))
    return brentq(lambda b: (1 / (1 + np.exp(-(eta + b)))).mean() - C.PI, -30, 30)


def gen_sample(rng, n, b0):
    X = gen_features(rng, n)
    logit = b0 + linear_signal(X)
    p = 1 / (1 + np.exp(-logit))
    d = rng.binomial(1, p)
    return X, logit, p, d


# ------------------------------------------------------- logística (IRLS)
def fit_logit(Z, y, iters=25):
    Z = np.asarray(Z, dtype=float).reshape(len(y), -1)
    mu, sd = Z.mean(0), Z.std(0)
    Zs = np.column_stack([np.ones(len(y)), (Z - mu) / sd])
    b = np.zeros(Zs.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(Zs @ b)))
        W = p * (1 - p)
        H = Zs.T @ (Zs * W[:, None]) + 1e-8 * np.eye(len(b))
        step = np.linalg.solve(H, Zs.T @ (y - p))
        b += step
        if np.abs(step).max() < 1e-8:
            break

    def predict(Znew):
        Zn = np.asarray(Znew, dtype=float).reshape(len(Znew), -1)
        Zn = np.column_stack([np.ones(len(Zn)), (Zn - mu) / sd])
        return 1 / (1 + np.exp(-(Zn @ b)))
    return predict


# ----------------------------------------------------------------- métricas
def auc_ks(score, d):
    order = np.argsort(score, kind="mergesort")
    y = d[order]
    n1 = y.sum(); n0 = len(y) - n1
    ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    auc = (ranks[d == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    ks = np.max(np.abs(np.cumsum(y) / n1 - np.cumsum(1 - y) / n0))
    return auc, ks


def channel_costs(pd_model, pd_true):
    n = len(pd_model)
    acc = np.argsort(pd_model, kind="mergesort")[: int(C.ACCEPT * n)]
    p_star = C.MARGIN / (C.MARGIN + C.LGD)
    ok = pd_model < p_star
    st2 = pd_model > C.STAGE2_MULT * C.PI
    ecl = np.where(st2, lifetime_pd(pd_model, C.HORIZON_YEARS), pd_model) * C.LGD
    return {
        "el_accepted": C.LGD * pd_true[acc].mean(),
        "profit_optimal": (((1 - pd_true) * C.MARGIN - pd_true * C.LGD) * ok).sum() / n,
        "capital_other": irb_k(pd_model, C.LGD, "other_retail").mean(),
        "capital_mortgage": irb_k(pd_model, C.LGD, "mortgage").mean(),
        "ecl_static": ecl.mean(),
        "mean_pd_model": pd_model.mean(),
        "mean_pd_true": pd_true.mean(),
    }


# ----------------------------------------------------------------- réplica
def replication(rep, b0):
    rng = np.random.default_rng(C.RANDOM_SEED + 1000 + rep)
    Xtr, ltr, ptr, dtr = gen_sample(rng, C.N_TRAIN, b0)
    Xte, lte, pte, dte = gen_sample(rng, C.N_TEST, b0)
    utr, ute = rng.standard_normal(C.N_TRAIN), rng.standard_normal(C.N_TEST)
    rows = []
    for sig in C.NOISE_GRID:                       # família A
        pd_te = fit_logit(ltr + sig * utr, dtr)(lte + sig * ute)
        auc, ks = auc_ks(pd_te, dte)
        rows.append({"rep": rep, "family": "A_noise", "level": sig, "auc": auc, "ks": ks,
                     **channel_costs(pd_te, pte)})
    for sub in C.SUBSETS:                          # família B
        Ztr = np.column_stack([Xtr[f] for f in sub])
        Zte = np.column_stack([Xte[f] for f in sub])
        pd_te = fit_logit(Ztr, dtr)(Zte)
        auc, ks = auc_ks(pd_te, dte)
        rows.append({"rep": rep, "family": "B_subset", "level": len(sub), "auc": auc, "ks": ks,
                     **channel_costs(pd_te, pte)})
    return rows


# ------------------------------------------------------ valor do ponto
COST_COLS = {"el_accepted": "C1 EL accepted", "profit_optimal": "C1' profit",
             "capital_other": "C3 K other retail", "capital_mortgage": "C3 K mortgage",
             "ecl_static": "C4 ECL (static)"}


def local_slopes(g):
    """Ajuste quadrático do custo em AUC (família A) e derivada nos AUC_EVAL, em bps/ponto."""
    out = {}
    a = g.auc.values
    for col in COST_COLS:
        coef = np.polyfit(a, g[col].values, 2)
        for t in C.AUC_EVAL:
            out[(col, t)] = (2 * coef[0] * t + coef[1]) * 0.01 * 1e4
    return out


def binormal_prediction(t, pi):
    d = d_from_auc(t)
    J = jacobians(d)["AUC"]
    h = 0.005
    s_br, _ = dbr_ddelta(d, pi, C.ACCEPT)

    def num(fn):
        return (fn(d_from_auc(t + h)) - fn(d_from_auc(t - h))) / (2 * h) * 0.01 * 1e4
    return {
        "el_accepted": 0.01 * C.LGD * s_br * J * 1e4,
        "profit_optimal": 0.01 * dprofit_ddelta(d, pi, C.LGD, C.MARGIN) * J * 1e4,
        "capital_other": num(lambda dd: channel_capital(dd, pi, C.LGD, "other_retail")),
        "capital_mortgage": num(lambda dd: channel_capital(dd, pi, C.LGD, "mortgage")),
        "ecl_static": num(lambda dd: channel_provision(dd, pi, C.LGD, C.STAGE2_MULT * pi,
                                                       C.HORIZON_YEARS)["ecl"]),
    }


def main(n_rep=C.N_REPLICATIONS):
    t0 = time.time()
    b0 = calibrate_intercept()
    rows = []
    for rep in range(n_rep):
        rows += replication(rep, b0)
        if (rep + 1) % 50 == 0:
            print(f"  réplica {rep + 1}/{n_rep}  ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(C.DATA_DIR / "sim_modelos.csv", index=False)

    # Tabela 5: modelos (média e IC 95% entre réplicas)
    q = lambda p: (lambda x: x.quantile(p))
    summ = (df.groupby(["family", "level"])
              .agg(auc=("auc", "mean"), auc_lo=("auc", q(.025)), auc_hi=("auc", q(.975)),
                   ks=("ks", "mean"), pd_model=("mean_pd_model", "mean"),
                   pd_true=("mean_pd_true", "mean"), el_acc=("el_accepted", "mean"),
                   cap_m=("capital_mortgage", "mean"), ecl=("ecl_static", "mean"))
              .reset_index())
    summ[["el_acc", "ecl"]] *= 1e4
    summ["cap_m"] *= 100
    summ.round(4).to_csv(C.TABLES_DIR / "Table_4.csv", index=False)
    print(summ.round(4).to_string(index=False))

    # Tabela 6: valor do ponto, simulação vs. forma fechada binormal
    slopes = [local_slopes(g) for _, g in df[df.family == "A_noise"].groupby("rep")]
    pi_hat = df.mean_pd_true.mean()
    out = []
    for col, lab in COST_COLS.items():
        for t in C.AUC_EVAL:
            v = np.array([s[(col, t)] for s in slopes])
            out.append({"channel": lab, "AUC": t, "sim_mean": v.mean(),
                        "sim_lo": np.quantile(v, .025), "sim_hi": np.quantile(v, .975),
                        "binormal": binormal_prediction(t, pi_hat)[col]})
    t6 = pd.DataFrame(out).round(2)
    t6.to_csv(C.TABLES_DIR / "Table_5.csv", index=False)
    print(t6.to_string(index=False))

    # Tabela S2 (apoio): o AUC basta? Desvio da família B em relação à curva da família A
    # (mesma réplica, interpolação linear em AUC), em bps de EAD
    dev = []
    for rep, g in df.groupby("rep"):
        a_ = g[g.family == "A_noise"].sort_values("auc")
        for _, b in g[g.family == "B_subset"].iterrows():
            if not a_.auc.min() <= b.auc <= a_.auc.max():
                continue
            row = {"rep": rep, "level": b.level, "auc": b.auc}
            for col in COST_COLS:
                row[col] = (b[col] - np.interp(b.auc, a_.auc, a_[col])) * 1e4
            dev.append(row)
    dev = pd.DataFrame(dev)
    t7 = dev.groupby("level").agg(
        auc=("auc", "mean"),
        **{f"{c}_mean": (c, "mean") for c in COST_COLS},
        **{f"{c}_sd": (c, "std") for c in COST_COLS}).reset_index().round(2)
    t7.to_csv(C.TABLES_DIR / "Table_S2_suficiencia_auc.csv", index=False)
    print(t7.to_string(index=False))

    # Figura 4: custo x AUC por modelo (100 réplicas) + curva binormal
    grid = np.linspace(0.60, 0.86, 40)
    bn = [
        [channel_selection(d_from_auc(a), pi_hat, C.LGD, C.ACCEPT)["el"] * 1e4 for a in grid],
        [channel_capital(d_from_auc(a), pi_hat, C.LGD, "mortgage") * 100 for a in grid],
        [channel_provision(d_from_auc(a), pi_hat, C.LGD, C.STAGE2_MULT * pi_hat,
                           C.HORIZON_YEARS)["ecl"] * 1e4 for a in grid],
    ]
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    for fam, mk, col, lab in [("A_noise", ".", "tab:blue", "noise (A)"),
                              ("B_subset", "x", "tab:orange", "omitted variables (B)")]:
        g = df[(df.family == fam) & (df.rep < 100)]
        ax[0].scatter(g.auc, g.el_accepted * 1e4, s=3, marker=mk, c=col, alpha=.3, label=lab)
        ax[1].scatter(g.auc, g.capital_mortgage * 100, s=3, marker=mk, c=col, alpha=.3)
        ax[2].scatter(g.auc, g.ecl_static * 1e4, s=3, marker=mk, c=col, alpha=.3)
    for a_, y in zip(ax, bn):
        a_.plot(grid, y, "k-", lw=1.2, label="binormal")
        a_.set_xlabel("AUC (test sample)"); a_.grid(alpha=.3)
    ax[0].set_title("(a) EL, accepted 80%"); ax[0].set_ylabel("bps of EAD")
    ax[1].set_title("(b) Mortgage IRB capital"); ax[1].set_ylabel("% of EAD")
    ax[2].set_title("(c) IFRS 9 ECL (static)"); ax[2].set_ylabel("bps of EAD")
    ax[0].legend(fontsize=7, markerscale=3)
    fig.tight_layout(); fig.savefig(C.FIGURES_DIR / "Figure_4.pdf"); plt.close(fig)
    print(f"tempo total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else C.N_REPLICATIONS)
