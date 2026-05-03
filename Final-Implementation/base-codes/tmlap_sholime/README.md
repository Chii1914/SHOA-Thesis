# TMLAP - SHOA+LIME (Base Implementation)

Implementacion base autonoma de SHOA+LIME para resolver TMLAP.

## 1) Objetivo

Ejecutar benchmark TMLAP con diagnostico LIME, rescues y trazabilidad completa por iteracion.

## 2) Estructura local

- `run_benchmark.py`: entrypoint local (smoke/full/profiles).
- `benchmarks/run_tmlap_sholime_benchmark.py`: benchmark principal.
- `benchmarks/run_tmlap_sholime_profiles_matrix.py`: matriz de perfiles.
- `benchmarks/sholime_tmlap_profiles_config.json`: perfiles soft/medium/hard.
- `base/`: controlador SHOA+LIME y dependencias.
- `benchmarks/1.instancia_simple.txt`, `2.instancia_mediana.txt`, `3.instancia_dura.txt`: instancias.
- `benchmarks/analyze_tmlap_benchmark_logs.py`: analisis por corrida/global.
- `benchmarks/organize_and_analyze_tmlap_outputs.py`: organizacion y analisis por grupos.

No hay imports ni rutas requeridas fuera de esta carpeta.

## 3) Dependencias

```bash
pip install numpy pandas matplotlib scikit-learn lime
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
python run_benchmark.py --mode full -- --rescue-mode levy_teleport --tag full_levy
```

## 5) Artefactos esperados

Cada corrida crea carpeta en `results/benchmark_logs/tmlap_sholime_*`.

Archivos obligatorios:

- `full_output.csv`: trazabilidad por iteracion con diagnostico/rescue/pesos.
- `runs_raw.csv`: resumen por run.
- `summary_by_function.csv`: agregados por instancia.
- `ranking_by_dimension.csv`: ranking.
- `config_used.json`: configuracion utilizada.
- `lime_contributions.csv`: contribuciones LIME por diagnostico.
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
ls -lah results/benchmark_logs/*/lime_contributions.csv
```

## 8) Nota de aislamiento

Esta implementacion esta empaquetada para ejecutarse de forma local sin depender del resto del repositorio.
