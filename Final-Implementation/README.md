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
