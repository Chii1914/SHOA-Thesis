# SHOA-COMBINED / TMLAP

Implementacion minima ejecutable de SHOA-COMBINED para instancias TMLAP.

## Ejecutar

python run_tmlap_combined.py --instances 1.instancia_simple.txt --pop 20 --max-iter 50 --runs 1

Opcional para activar LIME:

python run_tmlap_combined.py --instances 1.instancia_simple.txt --lime-enabled 1 --lime-min-samples 200

## Salidas

En outputs/run-YYYY-MM-DD-HH-MM-SS/ se generan:

- config_used.json
- runs_raw.csv
- full_output.csv
- lime_contributions.csv
- global_feature_explanations.csv
- stagnation_history.csv
- stagnation_events.csv
- summary_by_function.csv

## Nota

Esta variante tmlap prioriza ejecucion y trazabilidad base dentro de la nueva estructura. La variante cec2022 mantiene la implementacion completa de referencia.
