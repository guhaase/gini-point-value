"""Parâmetros centrais do V06 (mesmo padrão do V01)."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
# No repositório do paper, as saídas vão para ../latex; no repositório público
# de código (só esta pasta), vão para ./output.
_LATEX = BASE_DIR.parent / "latex"
OUT_DIR = _LATEX if (_LATEX / "main.tex").exists() else BASE_DIR / "output"
FIGURES_DIR = OUT_DIR / "figures"
TABLES_DIR = OUT_DIR / "tables"
for _d in (DATA_DIR, FIGURES_DIR, TABLES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42

# Carteira-base (varejo)
PI = 0.05            # PD média de 12 meses
LGD = 0.45           # LGD (Basileia, varejo sem garantia: referência)
ACCEPT = 0.80        # taxa de aprovação fixa (canal C1)
MARGIN = 0.04        # margem líquida anual sobre empréstimo bom (regra ótima de Stein)
SEGMENT = "other_retail"
COST_OF_EQUITY = 0.12    # para converter capital em custo anual
STAGE2_MULT = 2.0        # estágio 2 se PD12 > STAGE2_MULT * PI (aproximação estática)
HORIZON_YEARS = 3        # prazo para a ECL de vida toda
PORTFOLIO = 1_000_000_000  # R$ 1 bi, para conversão monetária

AUC_GRID = [0.60 + 0.01 * i for i in range(31)]   # 0,60 a 0,90
AUC_LEVELS_TABLE = [0.65, 0.70, 0.75, 0.80, 0.85]
PI_SENS = [0.02, 0.05, 0.10]

# Exemplo de limiar de materialidade (Tabela 7)
EXPOSURE_EXAMPLE = 2_000_000_000   # exposição da carteira ilustrativa
MATERIALITY_LIMIT = 1_000_000      # limite de materialidade por ano
AUC_LEVELS_THRESHOLD = [0.65, 0.75, 0.85]

# ---------------------------------------------------------------------------
# Simulação com o DGP do V01 (04_simulacao_dgp.py)
# ---------------------------------------------------------------------------
# Variáveis e pesos relativos do V01 (regime estável); sinal amplificado por
# SIGNAL_SCALE para cobrir AUC 0,62-0,85 (o DGP original tem AUC máximo ~0,69);
# intercepto recalibrado para PD média = PI.
FEATURE_SPEC = {
    "credit_score":     ("normal", (650, 80)),
    "debt_to_income":   ("normal", (0.35, 0.12)),
    "loan_to_value":    ("normal", (0.70, 0.15)),
    "employment_years": ("exp",    (5.0,)),
    "log_income":       ("normal", (10.5, 0.5)),
}
BETA_V01 = {"credit_score": -0.005, "debt_to_income": 2.00, "loan_to_value": 1.50,
            "employment_years": -0.10, "log_income": -0.50}
CLIP = {"credit_score": (300, 850), "debt_to_income": (0.01, 1.0),
        "loan_to_value": (0.10, 1.20), "employment_years": (0, 40),
        "log_income": (8.0, 13.0)}
SIGNAL_SCALE = 2.5
N_TRAIN = 50_000
N_TEST = 50_000
N_REPLICATIONS = 500
NOISE_GRID = [0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.6, 3.4, 4.5]   # família A: ruído no logit
SUBSETS = [["credit_score"],                                  # família B: variáveis omitidas
           ["credit_score", "debt_to_income"],
           ["credit_score", "debt_to_income", "loan_to_value"],
           ["credit_score", "debt_to_income", "loan_to_value", "employment_years"],
           list(FEATURE_SPEC)]
AUC_EVAL = [0.70, 0.75, 0.80]
