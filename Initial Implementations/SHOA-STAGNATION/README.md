# SHOA-STAGNATION (CEC2022)

Implementacion de SHOA + detector de estancamiento paper-style (MinSFEs + MaxFEs), sin LIME.

## Archivos principales

- `SHO_STAGNATION_Controller.py`: nucleo SHOA con detector de estancamiento.
- `stagnation_detector.py`: logica MinSFEs/MaxFEs (SFEs = FE - lastImprovementFE).
- `run_cec2022_stagnation.py`: runner CEC2022 (opfunu) y export run-timestamp.
- `opfunu_wrapper.py`: wrapper de funciones CEC2022 con contador nfev.
- `utils_logging.py`: utilidades de escritura CSV/JSON.

## Regla de estancamiento

Se marca estancamiento cuando:

- `SFEs = FE - lastImprovementFE`
- `stagnated = (SFEs >= MinSFEs)`

`MinSFEs` se calcula como ratio de `MaxFEs`.
Ratios recomendados para analisis: 0.02, 0.04, 0.10.

## Ejecucion benchmark

Desde la raiz del repo:

```bash
/Users/maximilianoaguirre/Desktop/SHOA-Thesis/.venv/bin/python "Initial Implementations/original-code/SHOA-STAGNATION/run_cec2022_stagnation.py" --functions all --dimension 10 --pop-size 30 --max-iter 50 --min-sfes-ratio 0.04 --progress-every 1 --log-level INFO
```

## Logging

- Log de progreso por iteracion (best, avg, FE, SFEs, estado de estancamiento).
- Log explicito cuando se detecta estancamiento (`stagnation_start`).
- Log explicito cuando se recupera (`recovered`).

Opciones:

- `--min-sfes-ratio`: ratio para MinSFEs respecto a MaxFEs.
- `--max-fes`: presupuesto global de evaluaciones; `0` usa estimacion automatica.
- `--progress-every N`: log cada N iteraciones.
- `--quiet`: desactiva log iteracion-a-iteracion del nucleo.

## Artifacts por run

- `config_used.json`
- `runs_raw.csv`
- `full_output.csv`
- `stagnation_history.csv`
- `stagnation_events.csv`
- `summary_by_function.csv`

## Graficos de convergencia con estancamiento

Script separado:

- `plot_convergence_stagnation_run.py`

Uso:

```bash
/Users/maximilianoaguirre/Desktop/SHOA-Thesis/.venv/bin/python "Initial Implementations/original-code/SHOA-STAGNATION/plot_convergence_stagnation_run.py" --run-dir "Initial Implementations/original-code/SHOA-STAGNATION/outputs/run-YYYY-MM-DD-HH-MM-SS"
```

Opcional:

- `--log-y` para escala logaritmica en el eje Y.
- `--show` para mostrar la figura ademas de guardarla.

El script guarda PNG en `run-timestamp/plots/` y marca:

- Intervalos en estancamiento (`stagnated=1`) con sombreado rojo suave.
- Eventos `stagnation_start` con marcador rojo.
- Eventos `recovered` con marcador verde.
