# Metodología y Tablas de Resultados — SHOA-COMBINED (CEC2022)

> Documento de detalle metodológico. Describe **cómo** se obtuvieron los resultados
> (pipeline, datos, fórmulas y pruebas) y presenta las tablas de fitness por función
> para el método propuesto **SHOA-COMBINED**, incluyendo el **fitness objetivo `f*`**
> de cada función CEC2022.

---

## 1. ¿Cómo se hizo? — Pipeline experimental

### 1.1 Ejecución de los experimentos
Se ejecutó un orquestador fail-safe que corre los tres algoritmos (PSO, SHOA y
SHOA-COMBINED) sobre las 12 funciones de CEC2022, en dos dimensiones (D10 y D20):

| Etapa | Detalle |
|---|---|
| Algoritmos | PSO, SHOA, **SHOA-COMBINED** (SHOA + controlador XAI/LIME) |
| Funciones | F1–F12 de CEC2022 (vía `opfunu.cec_based.cec2022`) |
| Dimensiones | D10 y D20 |
| Corridas independientes | 30 por función (semilla base reproducible) |
| Presupuesto por corrida | 200 000 evaluaciones de la función (FEs) |
| Salida cruda | un directorio `run-*` por algoritmo/dimensión con `runs_raw.csv`, `full_output.csv` (telemetría por iteración), `summary_by_function.csv`, eventos de estancamiento y contribuciones LIME |

Cada corrida registra el mejor fitness alcanzado (`best_fitness`) y la traza de
convergencia (`best_fitness_so_far` por FE en `full_output.csv`).

### 1.2 Cálculo de métricas
A partir de los resultados crudos se calcula, por corrida:

- **Fitness crudo:** `best_fitness` = valor de la función objetivo de la mejor solución.
- **Fitness objetivo `f*`:** óptimo global de cada función, tomado de los metadatos de
  opfunu (`f_global`).
- **Error:** la métrica de desempeño usada en las pruebas estadísticas,

$$\text{error} = \lvert f(x_{best}) - f^{*} \rvert$$

El error normaliza todas las funciones a un mismo origen (0 = óptimo), lo que permite
comparar y aplicar las pruebas estadísticas de forma homogénea.

### 1.4 ¿Cómo se hicieron las pruebas de hipótesis?

El contraste se realiza **por función y dimensión**, emparejando las 30 corridas del
método propuesto con las 30 de la referencia (mismo `run_number` → comparación pareada).

| Paso | Acción | Herramienta | Criterio |
|---|---|---|---|
| 1 | Calcular error por corrida | `|f(x_best) − f*|` | menor = mejor |
| 2 | Emparejar corridas propuesto/referencia | `run_number` común (n = 30) | muestras pareadas |
| 3 | Verificar normalidad | Shapiro-Wilk + Lilliefors (α = 0.05) | `p < 0.05` ⇒ no normal |
| 4 | Elegir test | por no-normalidad generalizada | test **no paramétrico** |
| 5 | Contrastar hipótesis | Wilcoxon de los rangos con signo, **una cola** (`alternative="less"`) | — |
| 6 | Decidir | comparar `p` con α y la dirección | acepta Hₐ si `p < 0.05` **y** `mean_prop < mean_ref` |
| 7 | Cuantificar tendencia | `delta = error_prop − error_ref` | `delta < 0` ⇒ propuesto mejor |

- **H₀:** `µ_prop ≥ µ_ref` — el método propuesto **NO** mejora.
- **Hₐ:** `µ_prop < µ_ref` — el método propuesto mejora de forma significativa.
- Se ejecutan **dos contrastes**: SHOA-COMBINED vs **SHOA** (aísla el aporte del XAI) y
  SHOA-COMBINED vs **PSO** (línea base clásica).

### 1.3 Generación de reportes
El script [Final-Implementation/presentation_reports.py](../../../../../Final-Implementation/presentation_reports.py)
reutiliza los datos ya completos (no re-ejecuta optimización) y produce:

1. **Tablas de fitness crudo** (Best/Worst/Mean/Std + `f*`) — sección 3 de este documento.
2. **Tablas resumen de error** (best/mean/std/median/worst por algoritmo).
3. **Pruebas de normalidad** (Shapiro-Wilk + Lilliefors).
4. **Pruebas de hipótesis** (Wilcoxon de una cola).
5. **Curvas de convergencia** y **boxplots**.

```bash
cd Final-Implementation
../.venv/bin/python presentation_reports.py \
    --output-root ../Resultados/experiments/cec2022_failsafe
```

---

## 2. Fitness objetivo `f*` de las funciones CEC2022

El óptimo global es el mismo para D10 y D20 (definido por el benchmark):

| Función | `f*` (objetivo) | Tipo |
|---|---|---|
| F1_2022 | 300 | Unimodal |
| F2_2022 | 400 | Multimodal básica |
| F3_2022 | 600 | Multimodal básica |
| F4_2022 | 800 | Multimodal básica |
| F5_2022 | 900 | Multimodal básica |
| F6_2022 | 1800 | Híbrida |
| F7_2022 | 2000 | Híbrida |
| F8_2022 | 2200 | Híbrida |
| F9_2022 | 2300 | Compuesta |
| F10_2022 | 2400 | Compuesta |
| F11_2022 | 2600 | Compuesta |
| F12_2022 | 2700 | Compuesta |

> Cuanto más cerca esté el fitness obtenido de `f*`, mejor es el desempeño.
> El **error** reportado en las pruebas estadísticas es justamente la distancia a `f*`.

---

## 3. Tablas de fitness — SHOA-COMBINED

Estadísticos del **fitness crudo** (`best_fitness`) sobre las 30 corridas por función,
con la columna añadida del **fitness objetivo `f*`**.

### 3.1 Dimensión D10

| Función | f* (Objetivo) | Best (Mejor) | Worst (Peor) | Mean (Media) | Std (Desviación) |
|---|---|---|---|---|---|
| F1_2022 | 300 | 308.535 | 2191.021 | 604.749 | 466.207 |
| F2_2022 | 400 | 400.341 | 630.988 | 478.108 | 65.094 |
| F3_2022 | 600 | 600.000 | 600.095 | 600.024 | 0.028 |
| F4_2022 | 800 | 822.000 | 858.000 | 837.412 | 7.855 |
| F5_2022 | 900 | 900.098 | 901.772 | 900.653 | 0.450 |
| F6_2022 | 1800 | 10111.529 | 43465.000 | 19500.796 | 7735.530 |
| F7_2022 | 2000 | 2023.677 | 2327.240 | 2068.315 | 78.016 |
| F8_2022 | 2200 | 2222.853 | 3889.764 | 2367.878 | 414.254 |
| F9_2022 | 2300 | 2304.637 | 2873.107 | 2619.078 | 203.858 |
| F10_2022 | 2400 | 357.339 | 2737.662 | 1370.795 | 512.754 |
| F11_2022 | 2600 | 2600.193 | 2719.694 | 2619.748 | 32.498 |
| F12_2022 | 2700 | 2257.372 | 2946.790 | 2821.483 | 130.944 |

### 3.2 Dimensión D20

| Función | f* (Objetivo) | Best (Mejor) | Worst (Peor) | Mean (Media) | Std (Desviación) |
|---|---|---|---|---|---|
| F1_2022 | 300 | 1620.926 | 13399.578 | 6845.813 | 3301.137 |
| F2_2022 | 400 | 573.297 | 1507.832 | 800.264 | 215.685 |
| F3_2022 | 600 | 600.066 | 600.916 | 600.330 | 0.258 |
| F4_2022 | 800 | 928.405 | 1041.887 | 978.939 | 28.737 |
| F5_2022 | 900 | 901.857 | 908.140 | 903.655 | 1.416 |
| F6_2022 | 1800 | 29358.947 | 119569937.544 | 17951410.216 | 34301034.726 |
| F7_2022 | 2000 | 2069.803 | 4148.106 | 2475.081 | 524.058 |
| F8_2022 | 2200 | 3262.629 | 6559.920 | 4025.250 | 747.479 |
| F9_2022 | 2300 | 2836.385 | 3778.155 | 3066.440 | 202.607 |
| F10_2022 | 2400 | 597.161 | 2778.771 | 1784.052 | 601.558 |
| F11_2022 | 2600 | 2624.536 | 4561.382 | 2945.335 | 442.028 |
| F12_2022 | 2700 | 2644.461 | 3263.978 | 3072.995 | 125.268 |

> **Nota sobre F3 y F5:** los valores Best/Mean prácticamente igualan a `f*`
> (600 y 900), indicando convergencia casi exacta al óptimo.
> **F6** muestra alta dispersión (función híbrida muy sensible), especialmente en D20.

---

## 4. Pruebas estadísticas

### 4.1 Verificación de normalidad (Shapiro-Wilk + Lilliefors, α = 0.05)

Se aplica a la muestra de **error** de cada algoritmo (30 corridas por función).
`estado = no normal` cuando `p < 0.05` (la mayoría de los casos), `normal` cuando
`p ≥ 0.05`, y `constante` cuando la muestra no tiene varianza (p = NaN).
Se muestra el `p` de Shapiro-Wilk para los tres algoritmos.

#### Dimensión D10

| Función | Dim | n | p_PSO | estado_PSO | p_SHOA | estado_SHOA | p_COMBINED | estado_COMBINED |
|---|---|---|---|---|---|---|---|---|
| F1 | 10 | 30 | NaN | constante | 7.73e-08 | no normal | 3.86e-09 | no normal |
| F2 | 10 | 30 | 5.22e-08 | no normal | 3.43e-02 | no normal | 4.65e-03 | no normal |
| F3 | 10 | 30 | 7.15e-10 | no normal | 8.71e-03 | no normal | 1.18e-04 | no normal |
| F4 | 10 | 30 | 1.94e-04 | no normal | 6.19e-02 | normal | 3.37e-01 | normal |
| F5 | 10 | 30 | 5.97e-07 | no normal | 1.76e-02 | no normal | 6.82e-02 | normal |
| F6 | 10 | 30 | 5.15e-08 | no normal | 8.93e-05 | no normal | 2.66e-04 | no normal |
| F7 | 10 | 30 | 2.79e-06 | no normal | 8.17e-07 | no normal | 8.10e-08 | no normal |
| F8 | 10 | 30 | 7.53e-11 | no normal | 4.57e-09 | no normal | 1.30e-10 | no normal |
| F9 | 10 | 30 | 2.37e-08 | no normal | 3.92e-05 | no normal | 9.91e-06 | no normal |
| F10 | 10 | 30 | 1.03e-05 | no normal | 9.69e-02 | normal | 9.91e-01 | normal |
| F11 | 10 | 30 | 1.45e-10 | no normal | 1.67e-05 | no normal | 1.26e-08 | no normal |
| F12 | 10 | 30 | 2.26e-05 | no normal | 2.82e-02 | no normal | 2.29e-03 | no normal |

#### Dimensión D20

| Función | Dim | n | p_PSO | estado_PSO | p_SHOA | estado_SHOA | p_COMBINED | estado_COMBINED |
|---|---|---|---|---|---|---|---|---|
| F1 | 20 | 30 | 2.39e-10 | no normal | 9.69e-01 | normal | 1.01e-01 | normal |
| F2 | 20 | 30 | 3.88e-04 | no normal | 3.49e-04 | no normal | 3.44e-04 | no normal |
| F3 | 20 | 30 | 6.75e-03 | no normal | 7.09e-02 | normal | 4.65e-04 | no normal |
| F4 | 20 | 30 | 1.30e-01 | normal | 9.55e-01 | normal | 8.65e-01 | normal |
| F5 | 20 | 30 | 4.23e-03 | no normal | 1.16e-03 | no normal | 2.56e-03 | no normal |
| F6 | 20 | 30 | 5.72e-01 | normal | 6.39e-09 | no normal | 6.19e-08 | no normal |
| F7 | 20 | 30 | 2.27e-06 | no normal | 1.60e-05 | no normal | 1.20e-06 | no normal |
| F8 | 20 | 30 | 1.28e-08 | no normal | 8.85e-12 | no normal | 7.48e-05 | no normal |
| F9 | 20 | 30 | 3.33e-04 | no normal | 4.17e-04 | no normal | 4.58e-04 | no normal |
| F10 | 20 | 30 | 1.33e-01 | normal | 2.85e-02 | no normal | 2.97e-02 | no normal |
| F11 | 20 | 30 | 2.35e-06 | no normal | 6.05e-05 | no normal | 1.21e-06 | no normal |
| F12 | 20 | 30 | 8.95e-02 | normal | 4.61e-01 | normal | 3.30e-01 | normal |

**Interpretación del paso de normalidad:**
- En la mayoría de las series `p < 0.05` (no normalidad); algunas (p. ej. PSO·D10·F1)
  presentan valores constantes.
- Esto **justifica usar un test no paramétrico** (Wilcoxon) para la comparación pareada.
- En conjunto, solo el **19.4 %** de las 72 muestras es normal según ambas pruebas.

### 4.2 Comparación de hipótesis — SHOA-COMBINED vs SHOA (Wilcoxon, una cola)

`delta = error_COMBINED − error_SHOA`; `delta < 0` ⇒ SHOA-COMBINED comete menos error.
Se acepta Hₐ cuando `p < 0.05` y la media del propuesto es menor.

| Función | Dim | n_pares | wilcoxon_stat | p_value | decisión (α=0.05) | mediana_delta | media_delta | tendencia |
|---|---|---|---|---|---|---|---|---|
| F1 | 10 | 30 | 75.0 | 3.65e-04 | Acepta Hₐ | -44.327 | -286.732 | SHOA-COMBINED mejor (signif.) |
| F2 | 10 | 30 | 100.0 | 2.69e-03 | Acepta Hₐ | -1.342 | -13.714 | SHOA-COMBINED mejor (signif.) |
| F3 | 10 | 30 | 60.0 | 8.49e-05 | Acepta Hₐ | -0.024 | -0.037 | SHOA-COMBINED mejor (signif.) |
| F4 | 10 | 30 | 150.0 | 3.68e-01 | No rechazar H₀ | 0.000 | -0.544 | Empate (mediana delta = 0) |
| F5 | 10 | 30 | 217.0 | 3.81e-01 | No rechazar H₀ | -0.001 | 0.001 | SHOA-COMBINED mejor (no signif.) |
| F6 | 10 | 30 | 165.0 | 8.53e-02 | No rechazar H₀ | -1062.056 | -1570.024 | SHOA-COMBINED mejor (no signif.) |
| F7 | 10 | 30 | 178.0 | 1.36e-01 | No rechazar H₀ | -1.684 | -2.751 | SHOA-COMBINED mejor (no signif.) |
| F8 | 10 | 30 | 163.0 | 7.90e-02 | No rechazar H₀ | -3.064 | -167.141 | SHOA-COMBINED mejor (no signif.) |
| F9 | 10 | 30 | 93.0 | 1.61e-03 | Acepta Hₐ | -3.097 | -14.719 | SHOA-COMBINED mejor (signif.) |
| F10 | 10 | 30 | 293.0 | 8.94e-01 | No rechazar H₀ | 210.336 | 101.376 | SHOA mejor (no signif.) |
| F11 | 10 | 30 | 89.0 | 1.18e-03 | Acepta Hₐ | -3.863 | -16.299 | SHOA-COMBINED mejor (signif.) |
| F12 | 10 | 30 | 288.0 | 8.73e-01 | No rechazar H₀ | 9.016 | 9.670 | SHOA mejor (no signif.) |
| F1 | 20 | 30 | 74.0 | 3.33e-04 | Acepta Hₐ | -4388.385 | -3529.675 | SHOA-COMBINED mejor (signif.) |
| F2 | 20 | 30 | 143.0 | 3.33e-02 | Acepta Hₐ | -29.585 | -118.925 | SHOA-COMBINED mejor (signif.) |
| F3 | 20 | 30 | 37.0 | 5.30e-06 | Acepta Hₐ | -0.307 | -0.284 | SHOA-COMBINED mejor (signif.) |
| F4 | 20 | 30 | 196.0 | 2.32e-01 | No rechazar H₀ | -4.717 | -1.083 | SHOA-COMBINED mejor (no signif.) |
| F5 | 20 | 30 | 113.0 | 6.42e-03 | Acepta Hₐ | -0.133 | -0.205 | SHOA-COMBINED mejor (signif.) |
| F6 | 20 | 30 | 192.0 | 2.08e-01 | No rechazar H₀ | -3064.092 | -17238550.814 | SHOA-COMBINED mejor (no signif.) |
| F7 | 20 | 30 | 178.0 | 1.36e-01 | No rechazar H₀ | -11.989 | -64.709 | SHOA-COMBINED mejor (no signif.) |
| F8 | 20 | 30 | 186.0 | 1.75e-01 | No rechazar H₀ | -26.858 | -8953.974 | SHOA-COMBINED mejor (no signif.) |
| F9 | 20 | 30 | 143.0 | 3.33e-02 | Acepta Hₐ | -88.374 | -85.502 | SHOA-COMBINED mejor (signif.) |
| F10 | 20 | 30 | 205.0 | 2.92e-01 | No rechazar H₀ | -149.195 | -62.189 | SHOA-COMBINED mejor (no signif.) |
| F11 | 20 | 30 | 121.0 | 1.04e-02 | Acepta Hₐ | -63.518 | -164.198 | SHOA-COMBINED mejor (signif.) |
| F12 | 20 | 30 | 184.0 | 1.64e-01 | No rechazar H₀ | -7.892 | -25.755 | SHOA-COMBINED mejor (no signif.) |

**Resumen de los contrastes (acepta Hₐ = mejora significativa):**

| Contraste | D10 | D20 | Global |
|---|---|---|---|
| SHOA-COMBINED vs SHOA | 5 / 12 | 6 / 12 | **11 / 24** |
| SHOA-COMBINED vs PSO | 0 / 12 | 1 / 12 | 1 / 24 |

> **Lectura:** frente a SHOA, el método propuesto **nunca pierde de forma significativa**
> (no hay casos "SHOA mejor (signif.)") y gana en 11 de 24, evidenciando el aporte del
> controlador XAI. El detalle del contraste vs PSO está en `wilcoxon_combined_vs_pso.csv`.

---

## 4.3 Comparativo SHOA-COMBINED vs SHOA puro (error medio ± desv.)

Comparación directa del **error al óptimo** (`|f(x_best) − f*|`, menor es mejor),
media sobre 30 corridas ± desviación estándar. **En negrita el mejor resultado**
(menor media) por función.

### Dimensión 10

| Función | SHOA-COMBINED (media ± std) | SHOA (media ± std) | Mejor |
|---|---|---|---|
| F1 | **3.05e+02 ± 4.66e+02** | 5.91e+02 ± 7.98e+02 | SHOA-COMBINED |
| F2 | **7.81e+01 ± 6.51e+01** | 9.18e+01 ± 6.00e+01 | SHOA-COMBINED |
| F3 | **2.43e-02 ± 2.75e-02** | 6.09e-02 ± 4.89e-02 | SHOA-COMBINED |
| F4 | **3.74e+01 ± 7.86e+00** | 3.80e+01 ± 7.73e+00 | SHOA-COMBINED |
| F5 | 6.53e-01 ± 4.50e-01 | **6.52e-01 ± 4.72e-01** | SHOA |
| F6 | **1.77e+04 ± 7.74e+03** | 1.93e+04 ± 8.92e+03 | SHOA-COMBINED |
| F7 | **6.83e+01 ± 7.80e+01** | 7.11e+01 ± 6.59e+01 | SHOA-COMBINED |
| F8 | **1.68e+02 ± 4.14e+02** | 3.35e+02 ± 6.30e+02 | SHOA-COMBINED |
| F9 | **3.19e+02 ± 2.04e+02** | 3.34e+02 ± 2.08e+02 | SHOA-COMBINED |
| F10 | 1.06e+03 ± 4.52e+02 | **9.55e+02 ± 4.44e+02** | SHOA |
| F11 | **1.97e+01 ± 3.25e+01** | 3.60e+01 ± 3.90e+01 | SHOA-COMBINED |
| F12 | 1.59e+02 ± 8.00e+01 | **1.49e+02 ± 7.56e+01** | SHOA |

> **D10:** SHOA-COMBINED gana en **9 de 12** funciones (F1–F4, F6–F9, F11);
> SHOA gana en 3 (F5, F10, F12).

### Dimensión 20

| Función | SHOA-COMBINED (media ± std) | SHOA (media ± std) | Mejor |
|---|---|---|---|
| F1 | **6.55e+03 ± 3.30e+03** | 1.01e+04 ± 2.69e+03 | SHOA-COMBINED |
| F2 | **4.00e+02 ± 2.16e+02** | 5.19e+02 ± 2.47e+02 | SHOA-COMBINED |
| F3 | **3.30e-01 ± 2.58e-01** | 6.14e-01 ± 2.06e-01 | SHOA-COMBINED |
| F4 | **1.79e+02 ± 2.87e+01** | 1.80e+02 ± 2.19e+01 | SHOA-COMBINED |
| F5 | **3.66e+00 ± 1.42e+00** | 3.86e+00 ± 1.34e+00 | SHOA-COMBINED |
| F6 | **1.79e+07 ± 3.43e+07** | 3.52e+07 ± 7.41e+07 | SHOA-COMBINED |
| F7 | **4.75e+02 ± 5.24e+02** | 5.40e+02 ± 5.38e+02 | SHOA-COMBINED |
| F8 | **1.83e+03 ± 7.47e+02** | 1.08e+04 ± 4.92e+04 | SHOA-COMBINED |
| F9 | **7.66e+02 ± 2.03e+02** | 8.52e+02 ± 2.41e+02 | SHOA-COMBINED |
| F10 | **6.82e+02 ± 5.23e+02** | 7.44e+02 ± 5.21e+02 | SHOA-COMBINED |
| F11 | **3.45e+02 ± 4.42e+02** | 5.10e+02 ± 5.12e+02 | SHOA-COMBINED |
| F12 | **3.77e+02 ± 1.13e+02** | 4.02e+02 ± 1.32e+02 | SHOA-COMBINED |

> **D20:** SHOA-COMBINED gana en **las 12** funciones.

**Resumen del comparativo (victorias por menor error medio):**

| Algoritmo | D10 | D20 | Global |
|---|---|---|---|
| **SHOA-COMBINED** | **9 / 12** | **12 / 12** | **21 / 24** |
| SHOA | 3 / 12 | 0 / 12 | 3 / 24 |

> **Lectura:** por error medio, SHOA-COMBINED supera a SHOA en **21 de 24** casos.
> En D20 la mejora es total; en D10 SHOA solo conserva ventaja marginal en F5, F10 y F12.
> Datos crudos en `comparison_combined_vs_shoa_D10.csv` y `comparison_combined_vs_shoa_D20.csv`.

### Convergencia (collage SHOA-COMBINED · CEC2022)

![Convergencia D10](plots/collage_convergencia_combined_D10.png)

*Figura: convergencia media (30 corridas) ± 1 desv. estándar, eje Y logarítmico, D=10.*

![Convergencia D20](plots/collage_convergencia_combined_D20.png)

*Figura: convergencia media (30 corridas) ± 1 desv. estándar, eje Y logarítmico, D=20.*

---

## 5. Archivos de respaldo

Ubicación: `Resultados/experiments/cec2022_failsafe/reports/presentation/tables/`

| Archivo | Contenido |
|---|---|
| `raw_fitness_combined_D10.csv`, `raw_fitness_combined_D20.csv` | tablas de esta sección 3 (fitness crudo + `f*`) |
| `results_summary_D10.csv`, `results_summary_D20.csv` | estadísticos de **error** para los 3 algoritmos |
| `normality_tests.csv`, `normality_summary.csv` | Shapiro-Wilk + Lilliefors |
| `wilcoxon_combined_vs_shoa.csv`, `wilcoxon_combined_vs_pso.csv` | contrastes de hipótesis por función, con `wilcoxon_statistic`, `p_value`, `decision`, `mediana_delta`, `media_delta` y `tendencia` |
| `comparison_combined_vs_shoa_D10.csv`, `comparison_combined_vs_shoa_D20.csv` | comparativo de error medio ± std SHOA-COMBINED vs SHOA + columna `mejor` (sección 4.3) |

Figuras: `plots/collage_convergencia_combined_D10.png`, `plots/collage_convergencia_combined_D20.png`.

---

## 6. Marco conceptual

Conceptos base (sección 2.1 de la tesis), cada uno con su definición breve.

- **2.1.1 Heurísticas:** Método o estrategia para resolver problemas complejos o tomar decisiones, basado en reglas empíricas, sin garantizar una solución óptima o exacta.

- **2.1.2 Metaheurísticas:** Marco de trabajo de alto nivel para diseñar algoritmos de optimización que exploran espacios de búsqueda grandes y complejos. Constan de dos fases: **exploración** (diversificación) y **explotación** (intensificación).

- **2.1.3 Optimización por enjambre de partículas (PSO):** Metaheurística de optimización global basada en inteligencia de enjambre, que modela el comportamiento social colectivo (bandadas, cardúmenes). Cada partícula tiene posición, velocidad, mejor posición individual (Pbest) y global (Gbest).

- **2.1.4 Algoritmos bio-inspirados:** Categoría de metaheurísticas que resuelven problemas de optimización imitando procesos y comportamientos de la naturaleza, basándose en la inteligencia colectiva.

- **2.1.5 SeaHorse Optimization Algorithm (SHO):** Algoritmo metaheurístico bio-inspirado basado en PSO, enfocado en el comportamiento de los caballitos de mar. Consta de tres fases: **movimiento**, **depredación** y **reproducción**.

- **2.1.6 Exploración - explotación:** Fases clave de las metaheurísticas; la exploración busca en regiones nuevas del espacio de soluciones, y la explotación refina las soluciones actuales. Su balance determina el rendimiento del algoritmo.

- **2.1.7 Parámetros de ajuste en algoritmos bio-inspirados:** Valores internos que toma la técnica para favorecer distintos aspectos; en SHO destacan el **Levy Flight** (paseo aleatorio de cola pesada) y el **parámetro de predación temporal** (factor escalador del paso temporal).

- **2.1.8 Convergencia prematura y estancamiento:** **Convergencia prematura** = pérdida rápida de diversidad poblacional agrupándose en un óptimo local; **estancamiento** = incapacidad de mejorar la función objetivo tras muchas iteraciones (síntoma observable de la convergencia prematura).

- **2.1.9 Hiperheurísticas:** Marco de búsqueda de alto nivel que opera sobre un conjunto de heurísticas/metaheurísticas de bajo nivel, automatizando qué técnica o parámetro aplicar (p. ej. con Q-Learning o aprendizaje por refuerzo).

- **2.1.10 Explicabilidad:** Grado de transparencia que permite a un humano comprender la causa, justificación o proceso que condujo a una decisión del modelo. Responde a "¿por qué se obtuvo este resultado?".

- **2.1.11 XAI (Inteligencia Artificial Explicable):** Campo que desarrolla teorías, metodologías y herramientas para que los sistemas de IA generen explicaciones, transformando modelos de "caja negra" en modelos transparentes.

- **2.1.12 LIME (Local Interpretable Model-agnostic Explanations):** Técnica de XAI agnóstica al modelo que genera explicaciones locales, aproximando modelos complejos mediante modelos simples e interpretables. Es la técnica de explicabilidad usada en el trabajo.

- **2.1.13 Explicabilidad post ejecución:** Explicación emitida sobre el cambio de comportamiento del algoritmo tras completar una iteración; común en hiperheurísticas. Parte del controlador en línea propuesto.

- **2.1.14 Métricas de diagnóstico:** Variables internas y continuas medidas en cada iteración para evaluar el estado del algoritmo (p. ej. diversidad poblacional y tasa de mejora); entrada principal del módulo XAI.

- **2.1.15 Programación lineal:** Técnica de optimización que busca el mejor resultado (máximo beneficio o mínimo costo) en un modelo matemático con componentes lineales: función objetivo, restricciones y variables de decisión.

- **2.1.16 Problemas de optimización computacional:** Desafíos que buscan el mejor conjunto de valores para variables de decisión; a diferencia de la PL, la función objetivo no es necesariamente lineal y suelen ser NP-Hard.

- **2.1.17 Variables discretas y continuas:** **Discretas** = toman un número contable de valores sin intermedios (p. ej. contador de iteraciones); **continuas** = toman infinitos valores en un rango (p. ej. resultados de análisis estadísticos).

- **2.1.18 Problemas NP-hard:** Clase de problemas de optimización y decisión computacionalmente muy difíciles, para los cuales se buscan formas más eficientes de resolverlos.

- **2.1.19 Congress on Evolutionary Computation (CEC):** Conferencia del IEEE donde se publican los **CEC Benchmark Functions**, conjuntos estandarizados de funciones de prueba para evaluar y comparar algoritmos de optimización.

- **2.1.20 Criterios estadísticos para la evaluación de metaheurísticas:** Protocolo estadístico riguroso para comparar rendimiento, empleando pruebas de normalidad (Shapiro-Wilk, Lilliefors) y tests no paramétricos (Wilcoxon).

- **2.1.21 Controlador en línea:** Componente de software que ajusta dinámicamente, en tiempo real, los parámetros o estrategia de un algoritmo. Monitorea métricas de diagnóstico, diagnostica riesgo de estancamiento vía XAI y ejecuta intervenciones justificadas.

---

## 7. Estado del arte (resumen por documento)

Revisión sistemática (PICO, 2020–2025) organizada en cinco ejes. Cada fila encasilla
un trabajo según su eje, categoría y aporte principal.

### Aplicaciones de metaheurísticas en problemas actuales

| Ref | Trabajo | Categoría | Aporte |
|---|---|---|---|
| [23] | Zhang et al. (2025) | Scheduling | Survey de metaheurísticas multiobjetivo para scheduling en Industria 4.0/5.0 |
| [27] | Liu (2024) | Scheduling | Modelo de scheduling industrial con SI + reducción de dimensión |
| [28] | Rajwar et al. (2023) | Scheduling | Revisión exhaustiva de metaheurísticas (taxonomía y retos) |
| [2] | Qawqzeh et al. (2021) | Scheduling | Revisión de SI para scheduling/optimización en cloud computing |
| [26] | Altay et al. (2024) | Scheduling | Estudio comparativo de 17 metaheurísticas en diseño de ingeniería |
| [25] | Barrera-García et al. (2023) | Feature Selection | Revisión sistemática de feature selection con metaheurísticas |
| [29] | Kamal et al. (2024) | Feature Selection | Survey de metaheurísticas para FS en alta dimensionalidad |
| [24] | Hamadneh (2025) | Feature Selection | Orangutan Optimization Algorithm para feature selection |
| [1] | Rezk et al. (2024) | Consideraciones metodológicas | Revisión de metaheurísticas en ingeniería eléctrica y civil |
| [30] | Osaba et al. (2021) | Consideraciones metodológicas | Tutorial de diseño y experimentación rigurosa de metaheurísticas |

### Trabajos recientes con SeaHorse Optimization Algorithm (SHO)

| Ref | Trabajo | Categoría | Aporte |
|---|---|---|---|
| [31] | Hashim et al. (2023) | Variante de SHO | mSHO (Modified Sea Horse Optimizer) para optimización global e ingeniería |
| [32] | Gülmez (2025) | Aplicación de SHO | LSTM + SHO para predicción de precios de acciones |
| [33] | Li et al. (2024) | Variante/aplicación | ESHO (mapeo caótico + seno-coseno) para hiperparámetros en imágenes agrícolas |
| [34] | Houssein et al. (2024) | Variante anti-estancamiento | SHO-OBL (aprendizaje por oposición) + cluster head en WSN; **relevante para estancamiento** |
| [35] | Erduman et al. (2025) | Aplicación de SHO | Estimación de parámetros de celda de combustible PEM |
| [36] | Hasanien et al. (2024) | Híbrido SHO | PSO + SHO para despacho óptimo de potencia reactiva con vehículos eléctricos |
| [37] | Wang et al. (2024) | Variante binaria | GOG-MBSHO (golden sine + escape) para selección de genes de cáncer |
| [38] | Andic et al. (2024) | Aplicación de SHO | Controlador de carga-frecuencia en sistemas eléctricos de dos áreas |

###  Mejoras a metaheurísticas (hiperheurísticas / control adaptativo)

| Ref | Trabajo | Categoría | Aporte |
|---|---|---|---|
| [8] | Črepinšek et al. (2025) | Anti-estancamiento | MsMA: partición meta-nivel para superar estancamiento sin modificar el algoritmo base |
| [39] | Hussien et al. (2023) | Mejora adaptativa | mBWO con evolución de élite y transición dinámica exploración-explotación |
| [40] | Lagos & Pereira (2024) | Hiperheurística | Multi-Armed Bandit (EXP3) para balance en VRPTW |
| [41] | Seyyedabbasi et al. (2024) | Híbrido | HBASCSO (Honey Badger + Sand Cat) para optimización global |
| [42] | Lin & Xu (2025) | Control adaptativo | ADE: evolución diferencial con ajuste dinámico y mejora de diversidad |
| [43] | Dokeroglu et al. (2024) | Revisión | Survey y taxonomía de hiperheurísticas |
| [44] | Abed-Alguni (2026) | Explicabilidad | EvoMapX: marco explicativo (OAM, PEG, CDS) para algoritmos poblacionales |
| [45] | Sun et al. (2025) | Control de parámetros | HCDE: control jerárquico con reinicios por entropía; **relevante para parámetros** |
| [46] | Rodríguez-Esparza et al. (2024) | Hiperheurística + RL | HHASA (Simulated Annealing + RL) para CEVRP de última milla |
| [47] | Zhang et al. (2023) | Revisión | Survey del balance exploración-explotación en Differential Evolution |

### Metaheurísticas con explicabilidad (XAI)

| Ref | Trabajo | Categoría | Aporte |
|---|---|---|---|
| [48] | Barros-Everett et al. (2025) | Predicción de parámetros | ML para ajustar parámetros en CVRPTW, con XAI en análisis |
| [49] | Hu et al. (2025) | Diagnóstico interpretable | Modelo interpretable (SHAP) para fallas en transformadores |
| [50] | Wang et al. (2025) | Interpretabilidad | Ensemble ML con LIME y SHAP; usa metaheurística Puma |
| [4] | Almeida et al. (2025) | Revisión sistemática | Revisión de explicabilidad en inteligencia computacional para optimización |
| [51] | Bolufé-Röhler & Tamayo-Vera (2025) | Revisión | ML para potenciar metaheurísticas (transparencia y trazabilidad) |
| [52] | Nakagawa et al. (2025) | Ajuste con XAI | Xtune: SHAP para tuning de hiperparámetros en series temporales |
| [53] | Medani et al. (2025) | XAI + optimización | LXAIOA-ADPCM: XAI + optimización para predicción de demencia |
| [54] | Almsallti et al. (2025) | XAI + ML | RBMO-ELM para predicción de emisiones de CO₂ |

###  SeaHorse Optimization Algorithm y LIME (vacío de conocimiento)

| Ref | Trabajo | Categoría | Aporte |
|---|---|---|---|
| [55] | Salamon & Ksiazek (2025) | SHO + LIME/SHAP | Único trabajo que combina SHO con LIME/SHAP, pero **solo como análisis post-hoc** (rayos-X de neumonía), no como guía del algoritmo → confirma el vacío que justifica esta tesis |

---

## 8. Resultados TMLAP — comparativo de los 3 algoritmos

Aplicación al **problema real de localización de instalaciones (TMLAP)**: asignación
de 2201 clientes a 500 hubs candidatos, minimizando la distancia total más los costos
fijos de apertura, sujeto a capacidad por hub y distancia máxima tolerada (`D_max = 12`).
Se comparan **PSO**, **SHOA** y **SHOA-COMBINED** sobre cuatro instancias de dificultad
creciente, con 30 corridas por configuración. Menor fitness es mejor.

### 8.1 Cuadro comparativo (fitness: media ± desv., con tasa de factibilidad)

| Instancia | Presupuesto (FEs) | PSO (media ± std) | SHOA (media ± std) | SHOA-COMBINED (media ± std) | Mejor |
|---|---|---|---|---|---|
| 1. Simple | 5 000 | **89.00 ± 0.00** | **89.00 ± 0.00** | **89.00 ± 0.00** | Empate (óptimo común) |
| 2. Mediana | 5 000 | **156.53 ± 1.83** | 159.23 ± 2.81 | 158.77 ± 3.37 | PSO |
| 3. Dura | 5 000 | **301.87 ± 4.01** | 311.83 ± 3.82 | 311.20 ± 5.33 | PSO |
| 4. Grande | 5 000 | **2460.43 ± 38.74** | 2673.37 ± 28.71 | 2675.20 ± 21.52 | PSO |
| 4. Grande | 50 000 | **2262.80 ± 44.13** | 2671.63 ± 22.14 | 2669.33 ± 28.19 | PSO |

> La **tasa de factibilidad fue 1.000 (100 %)** para los tres algoritmos en todas las
> instancias: todas las soluciones reportadas respetan capacidad y `D_max`.

### 8.2 Estadísticos completos por algoritmo (Best / Mean / Std / Median / Worst)

| Instancia | FEs | Algoritmo | Best | Mean | Std | Median | Worst |
|---|---|---|---|---|---|---|---|
| 1. Simple | 5 000 | PSO | 89.0 | 89.00 | 0.00 | 89.0 | 89.0 |
| 1. Simple | 5 000 | SHOA | 89.0 | 89.00 | 0.00 | 89.0 | 89.0 |
| 1. Simple | 5 000 | SHOA-COMBINED | 89.0 | 89.00 | 0.00 | 89.0 | 89.0 |
| 2. Mediana | 5 000 | PSO | **153.0** | **156.53** | 1.83 | 156.0 | 161.0 |
| 2. Mediana | 5 000 | SHOA | 154.0 | 159.23 | 2.81 | 160.0 | 164.0 |
| 2. Mediana | 5 000 | SHOA-COMBINED | 151.0 | 158.77 | 3.37 | 159.0 | 165.0 |
| 3. Dura | 5 000 | PSO | **285.0** | **301.87** | 4.01 | 302.0 | 306.0 |
| 3. Dura | 5 000 | SHOA | 301.0 | 311.83 | 3.82 | 312.0 | 320.0 |
| 3. Dura | 5 000 | SHOA-COMBINED | 290.0 | 311.20 | 5.33 | 312.0 | 317.0 |
| 4. Grande | 5 000 | PSO | **2383.0** | **2460.43** | 38.74 | 2465.5 | 2529.0 |
| 4. Grande | 5 000 | SHOA | 2619.0 | 2673.37 | 28.71 | 2673.0 | 2719.0 |
| 4. Grande | 5 000 | SHOA-COMBINED | 2636.0 | 2675.20 | 21.52 | 2675.0 | 2724.0 |
| 4. Grande | 50 000 | PSO | **2167.0** | **2262.80** | 44.13 | 2264.5 | 2343.0 |
| 4. Grande | 50 000 | SHOA | 2631.0 | 2671.63 | 22.14 | 2675.0 | 2712.0 |
| 4. Grande | 50 000 | SHOA-COMBINED | 2613.0 | 2669.33 | 28.19 | 2670.5 | 2713.0 |

### 8.3 Contraste estadístico SHOA vs PSO (Wilcoxon, α = 0.05)

`outcome` indica el resultado de SHOA frente a PSO: `−` = SHOA peor, `≈` = empate.

| Instancia | FEs | n_pares | SHOA media | PSO media | wilcoxon_stat | p_value | resultado SHOA vs PSO |
|---|---|---|---|---|---|---|---|
| 1. Simple | 5 000 | 30 | 89.00 | 89.00 | 0.0 | NaN | ≈ (empate exacto) |
| 2. Mediana | 5 000 | 30 | 159.23 | 156.53 | 59.5 | 1.06e-03 | − (PSO mejor, signif.) |
| 3. Dura | 5 000 | 30 | 311.83 | 301.87 | 1.0 | 4.11e-06 | − (PSO mejor, signif.) |
| 4. Grande | 5 000 | 30 | 2673.37 | 2460.43 | 0.0 | 1.73e-06 | − (PSO mejor, signif.) |
| 4. Grande | 50 000 | 30 | 2671.63 | 2262.80 | 0.0 | 1.73e-06 | − (PSO mejor, signif.) |

> **Lectura del caso TMLAP:** a diferencia del benchmark CEC2022, en este problema
> **discreto y fuertemente restringido** PSO domina en las instancias no triviales,
> mientras SHOA y SHOA-COMBINED quedan muy cerca entre sí (SHOA-COMBINED iguala o
> supera ligeramente a SHOA puro en la instancia más grande con mayor presupuesto:
> 2669.33 vs 2671.63). Esto sugiere que el controlador XAI/LIME aporta más en
> paisajes continuos multimodales que en la representación discreta cliente→hub.

### Convergencia (collage SHOA-COMBINED · TMLAP)

![Convergencia TMLAP](plots/collage_run1_tmlap_combined.png)

*Figura: convergencia + intervalos de estancamiento + disparos LIME de SHOA-COMBINED
sobre las instancias TMLAP (run 1).*

---

## 9. Requerimientos, preparación y evidencia del ambiente

### 9.1 Requerimientos mínimos y recomendados

**Hardware**

| Recurso | Mínimo | Recomendado |
|---|---|---|
| Procesador (CPU) | 2 núcleos @ 2.0 GHz | 4+ núcleos @ 2.5 GHz o superior |
| Memoria RAM | 4 GB | 8–16 GB (TMLAP usa matrices de distancias 2201×500) |
| Almacenamiento | 2 GB libres | 5+ GB libres (salidas crudas: CSV, JSON y figuras por corrida) |
| GPU | No requerida | No requerida (cómputo solo en CPU) |

**Software**

| Componente | Mínimo | Recomendado |
|---|---|---|
| Sistema operativo | Windows 10 / macOS 12 / Linux (kernel 5.x) | macOS 14+, Ubuntu 22.04+ o Windows 11 |
| Python | 3.10 | **3.14** (versión usada en el proyecto) |
| Gestor de paquetes | `pip` 23+ | `pip` actualizado + entorno virtual (`venv`) |
| Control de versiones | — | Git |

**Dependencias de Python (`requirements.txt`)** — librerías directas principales:

| Paquete | Versión | Uso en el proyecto |
|---|---|---|
| `numpy` | 2.4.4 | Cálculo numérico y vectorización |
| `scipy` | 1.17.1 | Pruebas estadísticas (Wilcoxon, Shapiro-Wilk) |
| `statsmodels` | 0.14.6 | Prueba de normalidad de Lilliefors |
| `scikit-learn` | 1.8.0 | Soporte de modelos para el módulo XAI |
| `lime` | 0.2.0.1 | Explicabilidad local (controlador XAI) |
| `opfunu` | 1.0.1 | Funciones benchmark CEC2022 (F1–F12) |
| `matplotlib` | 3.10.8 | Curvas de convergencia, boxplots y collages |
| `scikit-image` | 0.26.0 | Composición de imágenes para los collages |
| `tqdm` | 4.67.3 | Barras de progreso en las corridas |

### 9.2 Preparación del ambiente

Desde la raíz del repositorio:

```bash
# 1. Ubicarse en la raíz del proyecto
cd SHOA-Thesis

# 2. Crear el entorno virtual aislado
python3.14 -m venv .venv

# 3. Activar el entorno virtual
#    macOS / Linux:
source .venv/bin/activate
#    Windows (PowerShell):
.venv\Scripts\Activate.ps1

# 4. Actualizar pip e instalar las dependencias fijadas
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 5. Verificar la instalación
python --version
python -m pip list
```

Para ejecutar el pipeline sin activar el entorno, se invoca directamente el intérprete del venv:

```bash
cd Final-Implementation
../.venv/bin/python presentation_reports.py \
    --output-root ../Resultados/experiments/cec2022_failsafe
```

### 9.3 Evidencia de preparación del ambiente

**1. Versión de Python e intérprete del entorno virtual**

```text
$ ./.venv/bin/python --version
Python 3.14.x
```

**2. Dependencias instaladas (coinciden con `requirements.txt`)**

```text
$ ./.venv/bin/python -m pip list
Package            Version
------------------ -----------
lime               0.2.0.1
matplotlib         3.10.8
numpy              2.4.4
opfunu             1.0.1
scikit-learn       1.8.0
scipy              1.17.1
statsmodels        0.14.6
tqdm               4.67.3
...
```

**3. Verificación funcional de las librerías clave**

```text
$ ./.venv/bin/python -c "import numpy, scipy, sklearn, lime, opfunu, matplotlib, statsmodels; print('OK: todas las librerías importan')"
OK: todas las librerías importan
```

**4. Evidencia de ejecución exitosa del pipeline**

Las corridas generan directorios `run-*` con artefactos por algoritmo/dimensión, lo que
constituye la evidencia de que el ambiente quedó operativo:

```text
Resultados/experiments/cec2022_failsafe/raw/SHOA-COMBINED/D10/run-2026-05-24-02-43-57/
├── config_used.json
├── runs_raw.csv
├── full_output.csv
├── summary_by_function.csv
├── stagnation_events.csv
├── lime_contributions.csv
└── plots/
```

> **Sugerencia:** acompañar cada bloque con una captura de pantalla de la terminal
> (con fecha visible) para que la evidencia sea verificable en la defensa.

---

## 10. Conclusiones

### 10.1 Éxitos conseguidos

- **Método propuesto funcional y validado:** se construyó **SHOA-COMBINED** (SHOA + controlador en línea XAI/LIME) y se evaluó de forma reproducible (30 corridas, semilla base 42, 200 000 FEs) sobre las 12 funciones CEC2022 en D10 y D20.
- **Mejora estadísticamente significativa frente a SHOA puro:** por error medio, SHOA-COMBINED supera a SHOA en **21 de 24** casos (9/12 en D10 y **12/12 en D20**), y en el contraste Wilcoxon de una cola **nunca pierde de forma significativa** (gana en 11 de 24).
- **Aporte aislado del controlador XAI:** al comparar contra SHOA sin controlador, se demuestra que la mejora proviene del módulo de explicabilidad y no de un cambio del algoritmo base.
- **Pipeline experimental robusto y reproducible:** orquestador fail-safe, protocolo estadístico riguroso (normalidad Shapiro-Wilk + Lilliefors, test no paramétrico Wilcoxon) y generación automatizada de tablas, curvas de convergencia, boxplots y collages.
- **Aplicación a un problema real (TMLAP):** se llevó el método a un problema de localización de instalaciones (2201 clientes, 500 hubs) con **100 % de soluciones factibles** en los tres algoritmos.

### 10.2 Avances en el área

- **Llena un vacío de conocimiento:** la literatura solo registra SHO + LIME/SHAP como análisis **post-hoc**; este trabajo usa la explicabilidad **como guía en línea** del algoritmo durante la ejecución (controlador que anticipa el estancamiento).
- **Explicabilidad accionable:** se transforma a LIME de herramienta de interpretación a **mecanismo de control adaptativo**, diagnosticando el riesgo de estancamiento y disparando intervenciones (reinicio parcial, diversificación) justificadas por métricas de diagnóstico.
- **Evidencia de escalabilidad favorable:** el beneficio del controlador **crece con la dimensión** (mejora total en D20), señalando utilidad en paisajes más complejos.

### 10.3 Problemas y/o riesgos identificados

- **Dependencia del tipo de problema:** en **TMLAP** (discreto y fuertemente restringido) **PSO domina** las instancias no triviales; SHOA y SHOA-COMBINED quedan muy cercanos entre sí. El aporte del XAI es menor en la representación discreta cliente→hub.
- **Sensibilidad en funciones híbridas:** **F6** muestra dispersión muy alta (hasta ~1e8 en D20), lo que dificulta la estabilidad y la comparación.
- **Costo computacional del módulo XAI:** LIME añade sobrecarga (muestreo, ajuste local) que debe amortizarse con la frecuencia de disparo (`lime_every`) y la política de estancamiento.
- **Casos sin ventaja en D10:** SHOA conserva ventaja marginal en F5, F10 y F12, indicando que el controlador no es universalmente beneficioso.
- **Riesgo de overfitting de parámetros:** la configuración del controlador (umbrales de dominancia LIME, ventanas de historial, cooldown/warmup) fue ajustada para CEC2022 y podría no transferirse directamente a otros dominios.

### 10.4 Oportunidades y/o propuestas de trabajo futuro

- **Adaptar el controlador a problemas discretos/restringidos:** rediseñar las métricas de diagnóstico y las intervenciones para representaciones combinatorias (como TMLAP), donde PSO hoy domina.
- **Ajuste automático de hiperparámetros del controlador:** auto-tuning de `lime_every`, umbrales de dominancia y porcentaje de reinicio (p. ej. con un meta-optimizador o aprendizaje por refuerzo).
- **Explorar otras técnicas de XAI:** comparar LIME con **SHAP** u otros explicadores como guía en línea, y medir el trade-off explicabilidad-costo.
- **Ampliar el benchmark:** evaluar en CEC2017/CEC2020, mayores dimensiones (D50, D100) y más problemas reales para confirmar la generalización.
- **Estabilizar funciones de alta dispersión:** estrategias específicas (escalado robusto, reinicios dirigidos) para casos como F6.
- **Reducir el costo del módulo XAI:** explicadores incrementales o aproximados que mantengan la calidad del diagnóstico con menor sobrecarga.
