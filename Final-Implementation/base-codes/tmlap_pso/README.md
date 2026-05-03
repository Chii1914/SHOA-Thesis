# TMLAP - PSO (Base Implementation)

Implementacion base autonoma de PSO para resolver TMLAP.

## 1) Objetivo

Ejecutar benchmark TMLAP con PSO y generar artefactos compatibles para comparacion con SHOA y SHOA+LIME.

## 2) Estructura local

- `run_benchmark.py`: entrypoint local (smoke/full/profiles).
- `benchmarks/run_tmlap_pso_benchmark.py`: benchmark principal.
- `benchmarks/run_tmlap_pso_profiles_matrix.py`: matriz de perfiles.
- `benchmarks/pso_tmlap_profiles_config.json`: perfiles soft/medium/hard.
- `benchmarks/1.instancia_simple.txt`, `2.instancia_mediana.txt`, `3.instancia_dura.txt`: instancias.
- `benchmarks/analyze_tmlap_benchmark_logs.py`: analisis por corrida/global.
- `benchmarks/organize_and_analyze_tmlap_outputs.py`: organizacion y analisis por grupos.

No hay imports ni rutas requeridas fuera de esta carpeta.

## 3) Dependencias

```bash
pip install numpy pandas matplotlib
```

## 4) Ejecucion de pruebas

### Smoke test

```bash
python run_benchmark.py --mode smoke
```

### Benchmark completo

```bash
python run_benchmark.py --mode full
```

### Matriz de perfiles

```bash
python run_benchmark.py --mode profiles
```

### Argumentos extra

```bash
python run_benchmark.py --mode full -- --instances 1.instancia_simple.txt --runs 10
```

## 5) Artefactos esperados

Cada corrida crea carpeta en `results/benchmark_logs/tmlap_pso_*`.

Archivos obligatorios:

- `full_output.csv`: trazabilidad por iteracion.
- `runs_raw.csv`: resumen por run.
- `summary_by_function.csv`: agregados por instancia.
- `ranking_by_dimension.csv`: ranking.
- `config_used.json`: configuracion utilizada.
- `lime_contributions.csv`: header-only (compatibilidad no-LIME).
- `skipped_cases.csv`: solo si hubo fallas.

## 6) Artefactos extra (analisis/visualizacion)

### Analisis de corridas

```bash
python benchmarks/analyze_tmlap_benchmark_logs.py \
  --benchmark-root results/benchmark_logs \
  --runs all
```

### Organizacion + analisis por grupos

```bash
python benchmarks/organize_and_analyze_tmlap_outputs.py \
  --benchmark-root results/benchmark_logs
```

## 7) Verificacion rapida

```bash
ls -lah results/benchmark_logs/*/runs_raw.csv
ls -lah results/benchmark_logs/*/full_output.csv
```

## 8) Nota de aislamiento

Esta implementacion esta empaquetada para ejecutarse de forma local sin depender del resto del repositorio.
