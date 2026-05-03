# CEC2022 - SHOA+LIME (Base Implementation)

Implementacion base autonoma de SHOA+LIME para CEC2022.

## 1) Objetivo

Ejecutar benchmark CEC2022 con diagnostico de estancamiento via LIME, rescues y trazabilidad completa por iteracion.

## 2) Estructura local

- `run_benchmark.py`: entrypoint local (smoke/full/profiles).
- `benchmarks/run_cec2022_benchmark.py`: benchmark principal.
- `benchmarks/run_sholime_profiles_matrix.py`: matriz de perfiles.
- `benchmarks/sholime_profiles_config.json`: perfiles soft/medium/hard.
- `benchmarks/plot_lime_contributions.py`: plots de contribuciones LIME.
- `benchmarks/benchmark_sholime_gui.py`: visor interactivo de full output + LIME.
- `base/`: controlador SHOA+LIME y dependencias.

No hay imports ni rutas requeridas fuera de esta carpeta.

## 3) Dependencias

```bash
pip install numpy pandas matplotlib scikit-learn lime opfunu
pip install setuptools==75.8.0
```

## 4) Ejecucion de pruebas

### Smoke test

```bash
python run_benchmark.py --mode smoke
```

### Benchmark completo (protocolo tesis)

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

Cada corrida crea carpeta en `benchmark_logs/cec2022_sholime_*`.

Archivos obligatorios:

- `full_output.csv`: trazabilidad por iteracion, incluyendo diagnostico/rescue/pesos.
- `runs_raw.csv`: resumen por run.
- `summary_by_function.csv`: agregados por funcion/dimension.
- `ranking_by_dimension.csv`: ranking por rendimiento.
- `config_used.json`: configuracion utilizada.
- `lime_contributions.csv`: contribuciones LIME por diagnostico.
- `skipped_cases.csv`: solo si hubo fallas.

## 6) Artefactos extra (analisis/visualizacion)

### Plot de contribuciones LIME

```bash
python benchmarks/plot_lime_contributions.py \
  --csv-path benchmark_logs/<run_folder>/lime_contributions.csv \
  --all-matches --iteration 10
```

### GUI interactiva

```bash
python benchmarks/benchmark_sholime_gui.py --run-dir benchmark_logs/<run_folder>
```

## 7) Verificacion rapida

```bash
ls -lah benchmark_logs/*/runs_raw.csv
ls -lah benchmark_logs/*/lime_contributions.csv
```

## 8) Nota de aislamiento

Esta implementacion esta empaquetada para ejecutarse de forma local sin depender del resto del repositorio.
