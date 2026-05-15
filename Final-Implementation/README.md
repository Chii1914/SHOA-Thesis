# Final-Implementation

Esta carpeta contiene implementaciones base autonomas para CEC2022 y TMLAP.

## Estructura

- `base-codes/cec2022_shoa`
- `base-codes/cec2022_pso`
- `base-codes/cec2022_sholime`
- `base-codes/tmlap_shoa`
- `base-codes/tmlap_pso`
- `base-codes/tmlap_sholime`

Cada implementacion incluye:

- `run_benchmark.py` como entrypoint local.
- benchmark script(s) y configuraciones necesarias.
- codigo esencial local (sin dependencias de rutas fuera de su carpeta).
- README detallado con pruebas smoke/full/profiles y artefactos.

## Quick start

Ejemplo (cualquier implementacion):

```bash
cd base-codes/cec2022_sholime
python run_benchmark.py --mode smoke
```

## Start Tests (Servidor, Secuencial)

Para ejecutar toda la suite final con un solo comando y sin paralelismo:

```bash
cd Final-Implementation
python start-tests.py --suite all
```

Esto ejecuta:

- CEC2022 SHOA (full)
- CEC2022 PSO (full)
- CEC2022 SHOA+LIME en perfiles soft/medium/hard y modos leader_repulsion/levy_teleport
- TMLAP SHOA (full)
- TMLAP PSO (full)
- TMLAP SHOA+LIME (full)

Ademas, para TMLAP genera automaticamente instancias muy grandes y factibles de 500 y 1000 hubs dentro de `benchmarks/`.

### Bloque estadistico al final

Al terminar los benchmarks, `start-tests.py` ejecuta analisis estadistico SHOA vs SHOA+LIME:

- Normalidad con Shapiro-Wilk
- Wilcoxon pareado (si hay pares por `run_id`/`seed`)
- Mann-Whitney U (si no hay pares)

### Salidas

Cada corrida de `start-tests.py` crea una carpeta:

`final-test-runs/start_tests_<timestamp>/`

Con:

- `start_tests.log`
- `job_manifest.csv`
- `job_manifest.json`
- `start-tests-summary.md`
- `statistics/normality_shapiro_shoa_vs_sholime.csv`
- `statistics/comparisons_shoa_vs_sholime.csv`
- `statistics/reporte_estadistico_shoa_vs_sholime.md`

### Flags utiles

```bash
# Solo CEC2022
python start-tests.py --suite cec2022

# Solo TMLAP
python start-tests.py --suite tmlap

# Continuar lo maximo posible (default). Usar fail-fast si quieres cortar en primer error.
python start-tests.py --suite all --fail-fast

# Omitir bloque estadistico
python start-tests.py --suite all --skip-stats
```

## TMLAP Very Large Matrix (2000 Iter)

Para ejecutar solo TMLAP en instancias grandes con 2000 iteraciones y 30 runs:

```bash
cd Final-Implementation
python run_tmlap_very_large_matrix.py
```

Este runner ejecuta en secuencia:

- PSO TMLAP sobre `4.instancia_very_large_500_hubs.txt` y `5.instancia_very_large_1000_hubs.txt`
- SHOA TMLAP sobre las mismas dos instancias
- SHOA+LIME TMLAP en 6 combinaciones:
	- soft/medium/hard x leader_repulsion/levy_teleport

Y luego genera:

- Analisis descriptivo por algoritmo (CSV + graficos)
- Analisis estadistico pairwise:
	- PSO vs SHOA
	- SHOA vs SHOA+LIME (por profile_mode)
	- PSO vs SHOA+LIME (por profile_mode)

### Salidas

Cada corrida crea una carpeta:

`final-test-runs/tmlap_very_large_matrix_<timestamp>/`

Con:

- `run_tmlap_very_large_matrix.log`
- `job_manifest.csv`
- `job_manifest.json`
- `analysis_manifest.csv` (si no se omite analisis descriptivo)
- `analysis_manifest.json` (si no se omite analisis descriptivo)
- `analysis/descriptive/*`
- `analysis/statistics/normality_shapiro_pairwise.csv`
- `analysis/statistics/comparisons_pairwise.csv`
- `analysis/statistics/reporte_estadistico_pairwise.md`
- `run_tmlap_very_large_summary.md`

### Flags utiles

```bash
# Cambiar iteraciones y runs
python run_tmlap_very_large_matrix.py --max-iter 2000 --runs 30

# Cambiar instancias objetivo
python run_tmlap_very_large_matrix.py --instances 4.instancia_very_large_500_hubs.txt,5.instancia_very_large_1000_hubs.txt

# Cambiar perfiles y modos SHOLIME
python run_tmlap_very_large_matrix.py --profiles soft,medium,hard --modes leader_repulsion,levy_teleport

# Omitir analisis descriptivo
python run_tmlap_very_large_matrix.py --skip-descriptive-analysis

# Omitir analisis estadistico
python run_tmlap_very_large_matrix.py --skip-stats

# Probar sin ejecutar benchmarks
python run_tmlap_very_large_matrix.py --dry-run
```
