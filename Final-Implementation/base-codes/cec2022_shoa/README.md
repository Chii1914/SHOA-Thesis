# CEC2022 - SHOA (Base Implementation)

Implementacion base autonoma de SHOA para CEC2022.

## 1) Objetivo

Ejecutar SHOA puro sobre funciones CEC2022 (F1..F12) y generar artefactos estandarizados para comparacion directa con PSO y SHOA+LIME.

## 2) Estructura local

- `run_benchmark.py`: entrypoint local (smoke/full).
- `benchmarks/run_cec2022_benchmark_shoa_puro.py`: benchmark principal.
- `base/`: utilidades para construir objetivo CEC2022.
- `python-code/`: nucleo SHOA (`SHO.py`, `initialization.py`, `levy.py`).

No hay imports ni rutas requeridas fuera de esta carpeta.

## 3) Dependencias

Instalar en tu entorno activo:

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

### Pasar argumentos extra al benchmark

```bash
python run_benchmark.py --mode full -- --dims 10,20 --runs 10 --tag custom_full
```

## 5) Artefactos esperados

Cada corrida crea carpeta en `benchmark_logs/cec2022_shoa_puro_*`.

Archivos obligatorios:

- `full_output.csv`: trazabilidad por iteracion.
- `runs_raw.csv`: resumen por run (este es el resumen por corridas).
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
