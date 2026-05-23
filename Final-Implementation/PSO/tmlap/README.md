# PSO / TMLAP

Implementacion PSO para TMLAP con dos modos:

- `light` (default): PSO continuo minimo sobre el adaptador `tmlap_problem.py`.
- `legacy`: ejecuta el script original `PSO-TMLAP.py`.

## Ejecutar

python run_pso_tmlap.py

Smoke test rapido (modo light):

python run_pso_tmlap.py --instances 1.instancia_simple.txt --particles 20 --max-iter 20 --runs 1

Modo legacy:

python run_pso_tmlap.py --mode legacy --particles 20 --max-iter 20

Tambien puedes ejecutar directo el script legado:

python PSO-TMLAP.py

## Archivos de instancia disponibles

- 1.instancia_simple.txt
- 2.instancia_mediana.txt
- 3.instancia_dura.txt
- 4.instancia.txt
- 5.instancia.txt
