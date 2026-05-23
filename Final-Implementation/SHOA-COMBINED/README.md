# SHOA-COMBINED

Implementacion combinada online de:

- SHOA-LIME (diagnostico explicable por contribuciones de features)
- SHOA-STAGNATION (deteccion global de estancamiento por SFEs)

## Ejecucion

Desde esta carpeta:

```bash
python run_cec2022_combined.py --functions 1 --dim 10 --pop 20 --max-iter 50 --runs 1
```

## Parametros clave

- Trigger LIME actual: solo en `stagnation_start` (event-driven).
- Acumulacion de dataset: solo fuera de estancamiento; durante estancamiento se pausa.
- `--stagnation-lime-selection-mode {medoid,selected_agents}`:
  - `medoid` (default): explica un representante.
  - `selected_agents`: explica todo el grupo seleccionado.
- `--lime-selection-mode` sigue disponible como alias legacy del modo anterior.
- `--lime-min-samples`: minimo historico de samples antes de habilitar LIME.
- `--lime-every`: parametro legacy de cadencia. Se conserva por compatibilidad, pero no activa trigger periodico en el modo actual.
- `--min-sfes-ratio`: ratio para MinSFEs del detector de estancamiento.
- `--max-fes`: presupuesto global de FEs (0 = auto estimado).
- Warm-up fijo: 5% de MaxFEs. Durante warm-up solo corre SHOA (sin acumulacion LIME, sin trigger LIME y sin reinicios).
- `--restart-enabled {0,1}`: habilita reinicio parcial al detectar `stagnation_start`.
- `--restart-percent`: porcentaje fijo de reinicio en rango [5,10], preservando elite y reiniciando peores fitness.
- `--restart-cooldown-fes-ratio`: cooldown separado del detector, medido en FEs (si se omite usa `--min-sfes-ratio`).
- `--restart-dominance-threshold`: umbral de dominancia LIME fusionada para activar mutador por feature (default 0.90).
- Si no hay feature LIME elegible por umbral, se aplica fallback valido en el subconjunto de reinicio:
  - `x = LB + (UB - LB) * rand[0,1]`

## Artefactos de salida

En `outputs/run-YYYY-MM-DD-HH-MM-SS/` se generan:

- `config_used.json`
- `runs_raw.csv`
- `full_output.csv`
- `lime_contributions.csv`
- `global_feature_explanations.csv`
- `stagnation_history.csv`
- `stagnation_events.csv`
- `summary_by_function.csv`

La idea es mantener compatibilidad con scripts existentes de plots para LIME y estancamiento.

## Script unificado de graficos

Para generar en una sola pasada:

- contribuciones LIME (signed/absolute/temporal + global If/Sf)
- convergencia con tramos de estancamiento
- marcas de eventos (`stagnation_start`, `recovered`) y disparos LIME

usar:

```bash
python plot_combined_run.py --run-dir outputs/run-YYYY-MM-DD-HH-MM-SS
```

Opcionales:

- `--top-k-temporal 8`
- `--target-fitness 0`
- `--full-log-file /ruta/salida.txt`
- `--log-y`
- `--show`

El script ahora genera adicionalmente:

- curva de convergencia con linea horizontal de fitness objetivo (`target_fitness`)
- marcadores de reinicio de poblacion:
  - `restart_lime_mutator` (usando feature)
  - `restart_random_fallback` (sin feature elegible)
- graficos de contribucion LIME en momentos de reinicio:
  - `reset_all_*`
  - `reset_with_feature_*`
  - `reset_without_feature_*`
- `restart_outcomes.png` con conteo por tipo de resultado de reinicio
- `full_log_report.txt` con log completo por iteracion y por evento
