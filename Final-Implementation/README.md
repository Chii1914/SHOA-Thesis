# Final-Implementation

Estructura por algoritmo y problema:

- SHOA/
  - cec2022/
  - tmlap/
- PSO/
  - cec2022/
  - tmlap/
- SHOA-COMBINED/
  - cec2022/
  - tmlap/

## Comandos rapidos

SHOA + CEC2022:
python SHOA/cec2022/run_shoa_cec2022.py --functions 1 --dim 10 --pop 20 --max-iter 50 --runs 1

SHOA + TMLAP:
python SHOA/tmlap/run_shoa_tmlap.py --instances 1.instancia_simple.txt --pop 20 --max-iter 50 --runs 1

PSO + CEC2022:
python PSO/cec2022/run_pso_cec2022.py --functions 1 --dim 10 --particles 30 --max-iter 50 --runs 1

PSO + TMLAP:
python PSO/tmlap/run_pso_tmlap.py

SHOA-COMBINED + CEC2022:
python SHOA-COMBINED/cec2022/run_cec2022_combined.py --functions 1 --dim 10 --pop 20 --max-iter 50 --runs 1

SHOA-COMBINED + TMLAP:
python SHOA-COMBINED/tmlap/run_tmlap_combined.py --instances 1.instancia_simple.txt --pop 20 --max-iter 50 --runs 1

## Orquestador fail-safe CEC2022

Para correr el protocolo completo (PSO, SHOA y SHOA-COMBINED) con resume, reintentos,
tablas estadisticas y graficos:

python run_cec2022_failsafe.py --dims 10,20 --functions 1-12 --runs 30 --seed 42

Salida principal:

- experiments/cec2022_failsafe/state.json
- experiments/cec2022_failsafe/reports/tables/
- experiments/cec2022_failsafe/reports/plots/

Notas:

- El error reportado es |f(x)-f*| usando f* oficial de CEC2022 (opfunu).
- Las comparaciones estadisticas se hacen solo entre SHOA y PSO.
- Los graficos de contribucion se generan solo para SHOA-COMBINED.
