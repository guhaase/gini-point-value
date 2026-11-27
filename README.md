# What Is a Gini Point Worth? — replication code

Python code that reproduces every table and figure of

> *What Is a Gini Point Worth? Translating Discriminatory Power into Credit Risk Cost*
> (manuscript submitted to *The Journal of Risk Model Validation*).

## Requirements

Python 3.10+ and the packages in `requirements.txt`:

```
pip install -r requirements.txt
```

## Running

```
python run_all.py
```

or each script on its own. Outputs are written to `output/tables` and
`output/figures` (or to `../latex/` when run inside the paper's own repository).
All random draws use the fixed seed in `config.py` (`RANDOM_SEED = 42`).

| Script | Paper output |
|---|---|
| `02_canais_analiticos.py` | Tables 1, 2 and 7; Figures 1 and 2; supporting Table S1 (Jacobian × sensitivity decomposition) |
| `03_provisao_dois_periodos.py` | Table 3; Figure 3 (two-period IFRS 9 provisioning, one million loans per configuration) |
| `04_simulacao_dgp.py` | Tables 4 and 5; Figure 4; supporting Table S2 (is the AUC sufficient?). 500 replications; this is the slow step |
| `05_lgd.py` | Table 6; Figure 5 (LGD model power and Basel III floors) |
| `01_prototipo_binormal.py` | Early prototype of the binormal model; not used in the paper |

`config.py` holds every parameter reported in the text; `utils.py` holds the
binormal model, the cost channels and the closed-form derivatives.

Code comments are in Portuguese.

## License

MIT — see `LICENSE`.
