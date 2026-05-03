# CEC2022 - PSO (Base Implementation)

Implementacion base autonoma de PSO para CEC2022.

## 1) Objetivo

Ejecutar PSO sobre CEC2022 con el mismo contrato de salida que SHOA y SHOA+LIME para comparacion directa.

## 2) Estructura local

- `run_benchmark.py`: entrypoint local (smoke/full/profiles).
- `benchmarks/run_cec2022_pso_benchmark.py`: benchmark principal PSO.
- `benchmarks/run_cec2022_pso_profiles_matrix.py`: matriz de perfiles.
- `benchmarks/pso_cec2022_profiles_config.json`: perfiles soft/medium/hard.

No hay imports ni rutas requeridas fuera de esta carpeta.

## 3) Dependencias

```bash
pip install numpy opfunu
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
python run_benchmark.py --mode full -- --dims 10,20 --tag pso_d10_d20
```

## 5) Artefactos esperados

Cada corrida crea carpeta en `benchmark_logs/cec2022_pso_*`.

Archivos obligatorios:

- `full_output.csv`: trazabilidad por iteracion.
- `runs_raw.csv`: resumen por run.
- `summary_by_function.csv`: agregados por funcion/dimension.
- `ranking_by_dimension.csv`: ranking por rendimiento.
- `config_used.json`: configuracion utilizada.
- `lime_contributions.csv`: header-only (compatibilidad no-LIME).
- `skipped_cases.csv`: solo si hubo fallas.

## 6) Verificacion rapida

```bash
ls -lah benchmark_logs/*/runs_raw.csv
ls -lah benchmark_logs/*/full_output.csv
```

## 7) Nota de aislamiento

Esta implementacion esta empaquetada para ejecutarse de forma local sin depender del resto del repositorio.
