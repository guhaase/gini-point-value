"""Roda todos os experimentos do V06 em ordem."""
import runpy
from pathlib import Path

for script in ["02_canais_analiticos.py", "03_provisao_dois_periodos.py", "04_simulacao_dgp.py", "05_lgd.py"]:
    print(f"\n===== {script} =====")
    runpy.run_path(str(Path(__file__).with_name(script)), run_name="__main__")
