"""Extensão LGD (§8): valor de um ponto de poder discriminante de um modelo de LGD.

Modelo: LGD realizada l = Phi(mu + sig X), X ~ N(0,1) (probit-normal, bimodal para sig > 1).
Sinal do modelo de LGD: s_L = sqrt(r) X + sqrt(1-r) e. LGD calibrada:
    lhat(s_L) = E[l | s_L] = Phi((mu + sig sqrt(r) s_L) / sqrt(1 + sig^2 (1-r))).
Poder do modelo de LGD medido pela correlação de Spearman entre lhat e l:
    rho_S = (6/pi) arcsin(sqrt(r)/2).
PD: modelo binormal calibrado (utils), independente da LGD.

Canais:
  - seleção por perda esperada (aprova os ACCEPT com menor PD x lhat): EL dos aprovados;
  - capital IRB com piso de LGD (Basileia III: 30% outros varejo sem garantia, 50% QRRE);
  - capital sem piso e ECL IFRS 9: lineares em lhat, logo invariantes (verificado).

Saídas:
  latex/tables/Table_6.csv     (Tabela 6) valor de um ponto de Spearman da LGD por canal
  latex/figures/Figure_5.pdf   custo x poder do modelo de LGD
  data/lgd_log.txt
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq

import config as C
from utils import d_from_auc, pd_of_s, irb_k

AUC_PD = 0.75
LGD_MEAN = C.LGD
LGD_SIG = 1.5                 # dispersão latente (LGD bimodal, varejo sem garantia)
FLOORS = {"other_retail": 0.30, "revolving": 0.50}
RHO_LEVELS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
H = 0.005                     # meio passo da diferença central (meio ponto)

SP = np.linspace(-8.0, 11.0, 3801)          # escore de PD
WP_RAW = None
SL = np.linspace(-8.0, 8.0, 2401)           # sinal de LGD
WL = norm.pdf(SL); WL = WL / WL.sum()

MU = norm.ppf(LGD_MEAN) * np.sqrt(1 + LGD_SIG ** 2)


def pd_grid(auc):
    d = d_from_auc(auc)
    w = (1 - C.PI) * norm.pdf(SP) + C.PI * norm.pdf(SP - d)
    return pd_of_s(d, C.PI, SP), w / w.sum()


def r_from_rho(rho):
    return (2 * np.sin(np.pi * rho / 6)) ** 2


def lhat(r):
    return norm.cdf((MU + LGD_SIG * np.sqrt(r) * SL) / np.sqrt(1 + LGD_SIG ** 2 * (1 - r)))


def lgd_moments():
    x = np.linspace(-9, 9, 20001); w = norm.pdf(x); w /= w.sum()
    l = norm.cdf(MU + LGD_SIG * x)
    m = (l * w).sum(); v = ((l - m) ** 2 * w).sum()
    return m, np.sqrt(v), ((l < 0.1) * w).sum(), ((l > 0.9) * w).sum()


def r2_of(r):
    m, sd, *_ = lgd_moments()
    lh = lhat(r)
    return ((lh - m) ** 2 * WL).sum() / sd ** 2


def accepted_el(p, wp, lh, accept=C.ACCEPT):
    """EL por unidade de exposição aceita, aprovando os de menor p*lh (fração exata na fronteira)."""
    el = np.outer(p, lh).ravel()
    w = np.outer(wp, WL).ravel()
    o = np.argsort(el, kind="stable")
    el, w = el[o], w[o]
    cw = np.cumsum(w)
    k = np.searchsorted(cw, accept)
    part = accept - (cw[k - 1] if k > 0 else 0.0)
    tot = (el[:k] * w[:k]).sum() + el[k] * part
    return tot / accept


def costs(rho, auc=AUC_PD):
    p, wp = pd_grid(auc)
    r = r_from_rho(rho)
    lh = lhat(r)
    out = {"rho_S": rho, "r": r, "el_accepted": accepted_el(p, wp, lh)}
    for seg, f in FLOORS.items():
        g = irb_k(p, 1.0, seg)                 # K por unidade de LGD
        out[f"cap_{seg}_floor"] = (g * wp).sum() * (np.maximum(lh, f) * WL).sum()
        out[f"cap_{seg}_nofloor"] = (g * wp).sum() * (lh * WL).sum()
    out["el_total"] = (p * wp).sum() * (lh * WL).sum()
    return out


def value_rho(rho, key, auc=AUC_PD):
    return 0.01 * (costs(rho + H, auc)[key] - costs(rho - H, auc)[key]) / (2 * H)


def value_auc_pd(rho, key="el_accepted"):
    return 0.01 * (costs(rho, AUC_PD + H)[key] - costs(rho, AUC_PD - H)[key]) / (2 * H)


def main():
    m, sd, p_lo, p_hi = lgd_moments()
    log = [f"LGD probit-normal: mu={MU:.4f}, sig={LGD_SIG}; mean={m:.4f}, sd={sd:.4f}, "
           f"P(l<0.1)={p_lo:.3f}, P(l>0.9)={p_hi:.3f}",
           f"PD: binormal calibrado, AUC={AUC_PD}, pi={C.PI}, aprovação={C.ACCEPT}"]

    rows = []
    for rho in RHO_LEVELS:
        v_el = value_rho(rho, "el_accepted")
        v_pd = value_auc_pd(rho)
        rows.append({
            "Spearman": rho,
            "R2": r2_of(r_from_rho(rho)),
            "EL accepted (bps)": costs(rho)["el_accepted"] * 1e4,
            "V EL accepted": v_el * 1e4,
            "V K other retail floor 30%": value_rho(rho, "cap_other_retail_floor") * 1e4,
            "V K revolving floor 50%": value_rho(rho, "cap_revolving_floor") * 1e4,
            "V K no floor": value_rho(rho, "cap_other_retail_nofloor") * 1e4,
            "V PD AUC point (EL accepted)": v_pd * 1e4,
            "LGD points per PD AUC point": v_pd / v_el,
        })
    t = pd.DataFrame(rows)
    t.round(4).to_csv(C.TABLES_DIR / "Table_6.csv", index=False)
    log.append(t.round(3).to_string(index=False))

    c0 = costs(0.0)
    log.append(f"rho=0: EL accepted={c0['el_accepted']*1e4:.2f} bps; "
               f"K other floor={c0['cap_other_retail_floor']*100:.3f}%, no floor={c0['cap_other_retail_nofloor']*100:.3f}%; "
               f"K revolving floor={c0['cap_revolving_floor']*100:.3f}%")
    c6 = costs(0.6)
    log.append(f"rho=0.6: EL accepted={c6['el_accepted']*1e4:.2f} bps; "
               f"K other floor={c6['cap_other_retail_floor']*100:.3f}%, no floor={c6['cap_other_retail_nofloor']*100:.3f}%; "
               f"K revolving floor={c6['cap_revolving_floor']*100:.3f}%; EL total={c6['el_total']*1e4:.2f} vs rho0 {c0['el_total']*1e4:.2f}")

    # ---- Figura 5 ----
    grid = np.linspace(0.0, 0.8, 33)
    cs = pd.DataFrame([costs(x) for x in grid])
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    ax[0].plot(grid, cs.el_accepted * 1e4, "k-", label=f"EL, accepted ({C.ACCEPT:.0%}) on PD x LGD")
    ax[0].plot(grid, cs.el_total * 1e4, "k--", label="EL, whole population")
    ax[0].set_title("(a) Selection on expected loss"); ax[0].set_ylabel("bps of EAD")
    b0o, b0r = cs.cap_other_retail_nofloor.iloc[0], cs.cap_revolving_nofloor.iloc[0]
    ax[1].plot(grid, cs.cap_other_retail_floor / b0o * 100 - 100, "k-", label="Other retail, 30% floor")
    ax[1].plot(grid, cs.cap_revolving_floor / b0r * 100 - 100, "k:", label="Revolving, 50% floor")
    ax[1].plot(grid, cs.cap_other_retail_nofloor / b0o * 100 - 100, "k--", label="No floor")
    ax[1].set_title("(b) IRB capital with LGD floors"); ax[1].set_ylabel("% change vs. no-floor capital")
    for a_ in ax:
        a_.set_xlabel("Spearman correlation of LGD model"); a_.grid(alpha=0.3); a_.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(C.FIGURES_DIR / "Figure_5.pdf"); plt.close(fig)

    txt = "\n".join(log)
    (C.DATA_DIR / "lgd_log.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
