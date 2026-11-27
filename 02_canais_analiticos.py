"""Curvas de custo por canal e valor do ponto V(AUC) no modelo binormal calibrado.

Saídas:
  latex/tables/Table_1.csv  conversão ponto de AUC / Gini / KS            (Tabela 1)
  latex/tables/Table_2.csv  valor do ponto de AUC por canal e nível de AUC (Tabela 2)
  latex/tables/Table_7.csv  limiar de materialidade, carteira ilustrativa  (Tabela 7)
  latex/tables/Table_S1_decomposicao.csv  jacobiano x sensibilidade (Prop. 3; apoio ao texto)
  latex/figures/Figure_1.pdf  custo por canal x AUC
  latex/figures/Figure_2.pdf  valor do ponto x AUC, por PD média
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

import config as C
from utils import (d_from_auc, densities, pd_of_s, DS, channel_selection,
                   channel_optimal_cutoff, channel_capital, channel_provision,
                   jacobians, dbr_ddelta, dprofit_ddelta)


def costs(auc, pi=C.PI):
    d = d_from_auc(auc)
    _, _, mix = densities(d, pi)
    el_total = (pd_of_s(d, pi) * mix).sum() * DS * C.LGD
    sel = channel_selection(d, pi, C.LGD, C.ACCEPT)
    opt = channel_optimal_cutoff(d, pi, C.LGD, C.MARGIN)
    k = channel_capital(d, pi, C.LGD, C.SEGMENT)
    k_mort = channel_capital(d, pi, C.LGD, "mortgage")
    k_rev = channel_capital(d, pi, C.LGD, "revolving")
    prov = channel_provision(d, pi, C.LGD, C.STAGE2_MULT * pi, C.HORIZON_YEARS)
    return {
        "auc": auc, "el_total": el_total, "el_selected": sel["el"],
        "profit_optimal": opt["profit"], "capital": k,
        "capital_cost": k * C.COST_OF_EQUITY,
        "capital_mortgage": k_mort, "capital_revolving": k_rev, "ecl": prov["ecl"],
        "stage2": prov["share_stage2"],
    }


def value_of_point(auc, pi=C.PI, h=0.005):
    """Derivada central por 0,01 de AUC (um ponto)."""
    lo, hi = costs(auc - h, pi), costs(auc + h, pi)
    return {k: (hi[k] - lo[k]) / (2 * h) * 0.01 for k in lo if k != "auc"}


def main():
    # ---- Tabela 2 do paper: valor do ponto (bps de EAD) ----
    rows = []
    for a in C.AUC_LEVELS_TABLE:
        v = value_of_point(a)
        rows.append({
            "AUC": a, "Gini": 2 * a - 1,
            "C1 EL aprovados (bps)": v["el_selected"] * 1e4,
            "C1' lucro regra ótima (bps)": v["profit_optimal"] * 1e4,
            "C3 capital K outros varejo (bps)": v["capital"] * 1e4,
            "C3 capital K hipotecário (bps)": v["capital_mortgage"] * 1e4,
            "C3 capital K rotativo (bps)": v["capital_revolving"] * 1e4,
            "C3 custo do capital (bps/ano)": v["capital_cost"] * 1e4,
            "C4 ECL IFRS 9 (bps)": v["ecl"] * 1e4,
            "EL total (bps)": v["el_total"] * 1e4,
        })
    t1 = pd.DataFrame(rows).round(2)
    t1.to_csv(C.TABLES_DIR / "Table_2.csv", index=False)
    print(t1.to_string(index=False))

    # ---- Tabela 1 do paper: conversão entre métricas ----
    conv = []
    for a in C.AUC_LEVELS_TABLE:
        d = d_from_auc(a)
        dauc_dks = norm.pdf(d / np.sqrt(2)) / (np.sqrt(2) * norm.pdf(d / 2))
        conv.append({"AUC": a, "KS": 2 * norm.cdf(d / 2) - 1, "d": d,
                     "AUC por ponto de Gini": 0.5,
                     "AUC por ponto de KS": dauc_dks})
    t2 = pd.DataFrame(conv).round(3)
    t2.to_csv(C.TABLES_DIR / "Table_1.csv", index=False)
    print(t2.to_string(index=False))

    # ---- Tabela S1 (apoio): decomposição jacobiano x sensibilidade (Proposição 3) ----
    dec = []
    for a in C.AUC_LEVELS_TABLE:
        d = d_from_auc(a)
        J = jacobians(d)
        S, p_c = dbr_ddelta(d, C.PI, C.ACCEPT)
        Sp = dprofit_ddelta(d, C.PI, C.LGD, C.MARGIN)
        dec.append({"AUC": a, "delta": d, "J_AUC": J["AUC"], "J_KS": J["KS"],
                    "p_c": p_c, "dBR/ddelta": S,
                    "V C1 AUC fechado (bps)": 0.01 * C.LGD * S * J["AUC"] * 1e4,
                    "V C1 KS fechado (bps)": 0.01 * C.LGD * S * J["KS"] * 1e4,
                    "V C1' AUC fechado (bps)": 0.01 * Sp * J["AUC"] * 1e4})
    t4 = pd.DataFrame(dec).round(4)
    t4.to_csv(C.TABLES_DIR / "Table_S1_decomposicao.csv", index=False)
    print(t4.to_string(index=False))

    # ---- Tabela 7 do paper: limiar de materialidade (Seção de limiares) ----
    # valor por ano = |C1| + custo anual do capital (outros varejo); provisão à parte
    thr = []
    for a in C.AUC_LEVELS_THRESHOLD:
        v = value_of_point(a)  # sem arredondar
        v_bps = (abs(v["el_selected"]) + abs(v["capital_cost"])) * 1e4
        d_auc = C.MATERIALITY_LIMIT / (C.EXPOSURE_EXAMPLE * v_bps * 1e-4)  # pontos de AUC
        d = d_from_auc(a)
        dauc_dks = norm.pdf(d / np.sqrt(2)) / (np.sqrt(2) * norm.pdf(d / 2))
        thr.append({"AUC": a, "Valor do ponto (bps/ano)": v_bps,
                    "Limiar AUC (pontos)": d_auc, "Limiar Gini (pontos)": 2 * d_auc,
                    "Limiar KS (pontos)": d_auc / dauc_dks,
                    "Limiar probit delta": 0.01 * d_auc * jacobians(d)["AUC"]})
    t7 = pd.DataFrame(thr).round(3)
    t7.to_csv(C.TABLES_DIR / "Table_7.csv", index=False)
    print(t7.to_string(index=False))

    # ---- Figura 1: custo por canal ----
    grid = np.array(C.AUC_GRID)
    cs = pd.DataFrame([costs(a) for a in grid])
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    ax[0].plot(grid, cs.el_total * 1e4, "k--", label="EL, whole population")
    ax[0].plot(grid, cs.el_selected * 1e4, "k-", label=f"EL, accepted ({C.ACCEPT:.0%})")
    ax[0].set_title("(a) Selection"); ax[0].set_ylabel("bps of EAD")
    ax[1].plot(grid, cs.capital * 100, "k-", label="Other retail")
    ax[1].plot(grid, cs.capital_mortgage * 100, "k--", label="Mortgage")
    ax[1].plot(grid, cs.capital_revolving * 100, "k:", label="Revolving")
    ax[1].legend(fontsize=8)
    ax[1].set_title("(b) IRB capital"); ax[1].set_ylabel("% of EAD")
    ax[2].plot(grid, cs.ecl * 1e4, "k-", label="ECL (stage 1 + 2)")
    ax[2].plot(grid, cs.el_total * 1e4, "k--", label="12-month EL")
    ax[2].set_title("(c) IFRS 9 provision"); ax[2].set_ylabel("bps of EAD")
    for a_ in ax:
        a_.set_xlabel("AUC"); a_.grid(alpha=0.3)
    ax[0].legend(fontsize=8); ax[2].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(C.FIGURES_DIR / "Figure_1.pdf"); plt.close(fig)

    # ---- Figura 2: valor do ponto por PD média ----
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    styles = {0.02: ":", 0.05: "-", 0.10: "--"}
    g2 = grid[1:-1]
    for pi in C.PI_SENS:
        vs = pd.DataFrame([value_of_point(a, pi) for a in g2])
        lab = f"PD = {pi:.0%}"
        ax[0].plot(g2, -vs.el_selected * 1e4, "k" + styles[pi], label=lab)
        ax[1].plot(g2, -vs.capital * 1e4, "k" + styles[pi], label=lab)
        ax[2].plot(g2, vs.ecl * 1e4, "k" + styles[pi], label=lab)
    ax[0].set_title("(a) EL saved per AUC point")
    ax[1].set_title("(b) Capital released per AUC point")
    ax[2].set_title("(c) Extra ECL per AUC point")
    for a_ in ax:
        a_.set_xlabel("AUC level"); a_.set_ylabel("bps of EAD"); a_.grid(alpha=0.3)
        a_.axhline(0, color="grey", lw=0.6)
    ax[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(C.FIGURES_DIR / "Figure_2.pdf"); plt.close(fig)


if __name__ == "__main__":
    main()
