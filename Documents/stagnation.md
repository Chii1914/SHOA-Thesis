# Guía Técnica de Implementación

# Detección de Estancamiento Basada en MinSFEs + MaxFEs (alineada al paper MsMA)

## Metaheurísticas Poblacionales Continuas (CEC / Continuous Optimization)

---

# 1. Objetivo

Detectar estancamiento usando exactamente la idea central del paper:

- MaxFEs como criterio global de parada.
- MinSFEs como criterio interno de estancamiento.

Interpretación:

> Si se consumen MinSFEs evaluaciones consecutivas sin mejorar el mejor fitness global, se considera estancamiento.

Este documento se centra solo en detección.

---

# 2. Variables del detector

- MaxFEs: presupuesto total de evaluaciones de la corrida.
- FE: evaluaciones acumuladas usadas hasta el instante actual.
- bestFitness: mejor fitness global observado.
- lastImprovementFE: FE en la que ocurrió la última mejora de bestFitness.
- SFEs: evaluaciones sin mejora global consecutiva.
- MinSFEs: umbral interno de estancamiento.

Relación clave:

$$
SFEs = FE - lastImprovementFE
$$

---

# 3. Definición formal de estancamiento (paper-style)

Se declara estancamiento cuando:

$$
SFEs \ge MinSFEs
$$

y se mantiene MaxFEs como tope global:

$$
FE \le MaxFEs
$$

Por tanto, el algoritmo puede seguir en ejecución global hasta MaxFEs, pero internamente detecta ciclos de estancamiento al llegar a MinSFEs.

---

# 4. Inicialización del detector

En la primera mejora válida (o primera evaluación de referencia):

- fijar bestFitness
- fijar lastImprovementFE = FE actual
- fijar SFEs = 0

---

# 5. Actualización en cada paso

En cada actualización (evaluación individual o bloque/generación):

1. Evaluar si hubo mejora global real de bestFitness.
2. Si hubo mejora:
- actualizar bestFitness
- actualizar lastImprovementFE = FE actual
- SFEs = 0
3. Si no hubo mejora:
- SFEs = FE - lastImprovementFE
4. Detectar:
- stagnated = (SFEs >= MinSFEs)

---

# 6. Cómo fijar MinSFEs según el paper

El paper evalúa MinSFEs como porcentaje de MaxFEs. Configuración recomendada para análisis comparativo:

- MinSFEs = 0.02 * MaxFEs
- MinSFEs = 0.04 * MaxFEs
- MinSFEs = 0.10 * MaxFEs

Lectura práctica:

- 2%: detección más temprana.
- 4%: compromiso intermedio.
- 10%: detección más conservadora.

No hay valor universalmente óptimo para todos los algoritmos/problemas.

---

# 7. Pseudocódigo mínimo

```text
Entrada:
  MaxFEs
  MinSFEs

Estado:
  FE = 0
  bestFitness = +inf
  lastImprovementFE = 0

Mientras FE < MaxFEs:

  evaluar candidato/población
  FE = FE + evaluaciones_consumidas

  currentBest = mejor fitness global observado en este paso

  si currentBest < bestFitness:
      bestFitness = currentBest
      lastImprovementFE = FE

  SFEs = FE - lastImprovementFE

  si SFEs >= MinSFEs:
      stagnated = True
      reportar detección
  sino:
      stagnated = False
```

---

# 8. Telemetría mínima para tesis

Registrar en cada actualización:

- FE
- bestFitness
- lastImprovementFE
- SFEs
- MinSFEs
- stagnated

Ejemplo CSV:

```text
fe,bestFitness,lastImprovementFE,sfes,minsfes,stagnated
100,95.3,100,0,12000,False
500,95.3,100,400,12000,False
12100,95.3,100,12000,12000,True
```

---

# 9. Nota sobre δ-stagnation radius

El paper menciona δ-stagnation radius como marco conceptual para definir mejora relevante, pero el mecanismo operativo de MsMA para disparar estancamiento se implementa con MinSFEs (evaluaciones sin mejora) bajo el límite global MaxFEs.

---

# 10. Checklist de implementación

- [ ] Usar MaxFEs como criterio global
- [ ] Definir MinSFEs como porcentaje de MaxFEs
- [ ] Mantener lastImprovementFE
- [ ] Calcular SFEs = FE - lastImprovementFE
- [ ] Disparar estancamiento cuando SFEs >= MinSFEs
- [ ] Loggear métricas por actualización

---

# 11. Estado implementado actual (SHOA-STAGNATION)

Esta seccion documenta la implementacion actualmente activa en el codigo (post-reversion del modo por agente).

## 11.1 Alcance activo

- Detector global unico (no existe detector por agente).
- Criterio operacional: `stagnated = (SFEs >= MinSFEs)`.
- Tope global de corrida: `FE <= MaxFEs`.

## 11.2 Implementacion efectiva

- `StagnationDetector` mantiene estado global: `best_fitness`, `last_improvement_fe`, `sfes`, `stagnated`.
- Inicializacion con primer mejor fitness global disponible.
- Actualizacion por iteracion usando FE real acumulado del wrapper (`nfev`).
- Eventos emitidos:
  - `stagnation_start`
  - `recovered`
  - `none`

## 11.3 Logging operativo actual

El flujo reporta:

- Progreso iterativo con `best`, `avg`, `fe`, `sfes`, `stagnated`.
- Mensaje explicito al detectar estancamiento (iteracion y FE).
- Mensaje explicito al recuperarse tras mejora global.

## 11.4 Artefactos producidos actualmente

El runner SHOA-STAGNATION escribe:

- `config_used.json`
- `runs_raw.csv`
- `full_output.csv`
- `stagnation_history.csv`
- `stagnation_events.csv`
- `summary_by_function.csv`

## 11.5 Parametros de configuracion vigentes

- `--min-sfes-ratio` (default `0.04`)
- `--max-fes` (0 usa estimacion automatica)
- `--progress-every`, `--log-level`, `--quiet`

No hay parametros activos para estancamiento por agente porque ese modo fue removido intencionalmente.

## 11.6 Checklist de estado actual

- [x] MaxFEs usado como criterio global.
- [x] MinSFEs definido como porcentaje configurable de MaxFEs.
- [x] SFEs calculado con `FE - lastImprovementFE`.
- [x] Deteccion cuando `SFEs >= MinSFEs`.
- [x] Logging por iteracion y por evento.

---

# 12. Estado combinado online (SHOA-COMBINED)

Esta seccion resume como se integra el detector de estancamiento con LIME en un unico loop online.

## 12.1 Alcance

- LIME y estancamiento activos desde el inicio de la corrida.
- Detector global unico (no detector por agente).
- Trigger de explicabilidad LIME solo en `stagnation_start`.
- Durante estancamiento no se acumulan nuevas muestras para LIME.
- Al salir de estancamiento (`recovered`) se reanuda la acumulacion.
- Sin `rescue_mode` en esta etapa.

## 12.2 Integracion tecnica

- Controlador: `Initial Implementations/SHOA-COMBINED/SHO_HYBRID_Controller.py`.
- Runner: `Initial Implementations/SHOA-COMBINED/run_cec2022_combined.py`.
- El detector se actualiza por iteracion usando FE real (`nfev`) tras evaluar poblacion/offspring.
- La deteccion sigue siendo:

$$
stagnated = (SFEs \ge MinSFEs)
$$

con:

$$
SFEs = FE - lastImprovementFE
$$

## 12.3 Logging combinado

En cada iteracion se registra en linea:

- `best`, `avg`, `improved`
- estado LIME (`lime_triggered`, `diagnosis_id`, `lime_selection_mode`)
- fuente de trigger LIME (`lime_trigger_source`)
- estado de dataset LIME (`lime_dataset_updated`, `lime_buffer_size`)
- estado estancamiento (`fe`, `sfes`, `min_sfes`, `stagnated`, `event`)

Eventos `stagnation_start` y `recovered` se conservan como filas dedicadas.

## 12.4 Artefactos del runner combinado

`run_cec2022_combined.py` produce:

- `config_used.json`
- `runs_raw.csv`
- `full_output.csv`
- `lime_contributions.csv`
- `global_feature_explanations.csv`
- `stagnation_history.csv`
- `stagnation_events.csv`
- `summary_by_function.csv`

Este contrato mantiene compatibilidad con scripts de plots de LIME y estancamiento ya existentes.
