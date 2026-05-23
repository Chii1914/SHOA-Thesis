# SHOA-LIME (CEC2022)

Implementacion de SHOA + LIME por agente, con salidas por ejecucion en carpeta `run-timestamp`.

## Archivos principales

- `SHO_LIME_Controller.py`: nucleo SHOA con extraccion causal de features por agente.
- `run_cec2022_benchmark.py`: runner CEC2022 (opfunu) y export de artefactos.
- `lime_diagnostic.py`: seleccion estratificada 4%-3%-2%-1% y explicaciones LIME.
- `global_explanations.py`: agregaciones globales (If, Sf, ventanas temporales).
- `plot_feature_contributions_run.py`: script separado para graficos por run.

## Ejecucion benchmark

Desde la raiz del repo:

```bash
/Users/maximilianoaguirre/Desktop/SHOA-Thesis/.venv/bin/python "Initial Implementations/original-code/SHOA-LIME/run_cec2022_benchmark.py" --functions all --dimension 10 --pop-size 30 --max-iter 50 --runs 1 --progress-every 1 --log-level INFO
```

Flag de modo LIME:

- `--lime-selection-mode medoid` (default): explica solo el representante medoid del grupo seleccionado (mas rapido).
- `--lime-selection-mode selected_agents`: explica todos los agentes seleccionados en el diagnostico.

Ejemplo con medoid (default):

```bash
/Users/maximilianoaguirre/Desktop/SHOA-Thesis/.venv/bin/python "Initial Implementations/original-code/SHOA-LIME/run_cec2022_benchmark.py" --functions 1 --dimension 10 --pop-size 30 --max-iter 50 --runs 1 --lime-selection-mode medoid
```

Ejemplo con agentes seleccionados:

```bash
/Users/maximilianoaguirre/Desktop/SHOA-Thesis/.venv/bin/python "Initial Implementations/original-code/SHOA-LIME/run_cec2022_benchmark.py" --functions 1 --dimension 10 --pop-size 30 --max-iter 50 --runs 1 --lime-selection-mode selected_agents
```

Opciones de logging:

- `--progress-every N`: muestra progreso cada N iteraciones dentro de SHOA_LIME.
- `--log-level LEVEL`: `DEBUG`, `INFO`, `WARNING`, `ERROR`.
- `--quiet`: desactiva log iteracion-a-iteracion del nucleo (mantiene logs del runner).

## Graficos de contribucion por feature (codigo aparte)

```bash
/Users/maximilianoaguirre/Desktop/SHOA-Thesis/.venv/bin/python "Initial Implementations/original-code/SHOA-LIME/plot_feature_contributions_run.py" --run-dir "Initial Implementations/original-code/SHOA-LIME/outputs/run-YYYY-MM-DD-HH-MM-SS"
```

Se generan PNG en `run-timestamp/plots/`.
