# SHOA-TMLAP

Implementacion de TMLAP resuelta con Sea-Horse Optimizer (SHOA), reutilizando el core de `python-code/SHO.py`.

## Contenido

- `SHOA-TMLAP.py`: Solver SHOA para el mismo problema definido en `PSO-TMLAP/PSO-TMLAP.py`.
- `compare_pso_vs_shoa.py`: Comparacion automatica PSO vs SHOA con mismas semillas y estadisticas.

## Idea principal

- El problema TMLAP es discreto (asignacion cliente->hub).
- SHOA trabaja en continuo, asi que se usa un vector latente continuo y una etapa de reparacion para obtener una asignacion factible.
- La factibilidad respeta:
  - distancia maxima cliente-hub (`D_max`)
  - capacidad por hub (`capacidad`)

## Ejecucion

Desde la raiz del repositorio:

```bash
cd SHOA-TMLAP
python SHOA-TMLAP.py
```

## Salida

- Mejor fitness por iteracion (`g_best fitness`).
- Resultado final:
  - asignacion cliente->hub
  - hubs abiertos
  - fitness final

## Comparacion PSO vs SHOA

Desde la raiz del repositorio:

```bash
cd SHOA-TMLAP
python compare_pso_vs_shoa.py --runs 30 --seed-start 1 --max-iter 25 --population 10 --show-each
```

Archivos generados:

- `SHOA-TMLAP/results/<run_tag>/per_seed_results.csv`
- `SHOA-TMLAP/results/<run_tag>/summary.txt`

El resumen incluye metricas (min, max, media, mediana, desviacion estandar) y conteo de victorias por solver.
