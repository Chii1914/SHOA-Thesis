# Resumen de Reportes — SHOA-COMBINED en CEC2022 (D10 / D20)

> Documento de apoyo para la presentación. Sintetiza el protocolo experimental, las
> pruebas estadísticas y los principales hallazgos sobre el método propuesto
> **SHOA-COMBINED** (SHOA + controlador XAI basado en LIME) frente a las referencias
> **SHOA** (misma metaheurística sin XAI) y **PSO** (línea base clásica).

---

## 1. Objetivo

Validar empíricamente, mediante el benchmark CEC2022, si el método propuesto
SHOA-COMBINED mejora de forma estadísticamente significativa a sus referencias, y
generar artefactos listos para la presentación (curvas de convergencia, tablas de
resultados, pruebas de normalidad y de hipótesis).

---

## 2. Hipótesis

Para cada función y dimensión se contrasta el método propuesto contra cada referencia:

- **H₀:** `µ_prop ≥ µ_ref` — el método propuesto **NO** mejora (media de error igual o mayor).
- **Hₐ:** `µ_prop < µ_ref` — el método propuesto mejora de forma significativa (media de error menor).

Se **rechaza H₀ / se acepta Hₐ** cuando `p < α` **y** `mean_combined < mean_ref`.

---

## 3. Protocolo experimental

| Parámetro | Valor |
|---|---|
| Benchmark | CEC2022 (F1–F12) |
| Dimensiones | D10 y D20 |
| Corridas independientes | 30 por función |
| Presupuesto | 200 000 evaluaciones de la función (FEs) |
| Métrica de desempeño | error `= |f(x_best) − f*|` |
| f* | metadatos de opfunu CEC2022 |
| Algoritmos | PSO, SHOA, **SHOA-COMBINED** |
| Nivel de significancia | α = 0.05 |

**Protocolo estadístico (dos pasos):**
1. **Normalidad** — Shapiro-Wilk + Lilliefors sobre las muestras de error por run.
2. **Contraste de hipótesis** — Wilcoxon de una cola (`alternative="less"`), por ser
   no paramétrico y apropiado dada la no-normalidad observada.

---

## 4. Pruebas de normalidad

Resultados agregados sobre las **72 muestras** (3 algoritmos × 2 dimensiones × 12 funciones):

| Métrica | Valor |
|---|---|
| Muestras totales | 72 |
| Normales según Shapiro-Wilk | 22.2 % |
| Normales según Lilliefors | 27.8 % |
| Normales según ambos | 19.4 % |
| **No normales** | **80.6 %** |

**Conclusión:** la mayoría de las distribuciones de error no son normales, lo que
**justifica el uso del test no paramétrico de Wilcoxon** en lugar de una prueba t.

---

## 5. Contraste de hipótesis (Wilcoxon, α = 0.05)

### 5.1 SHOA-COMBINED vs SHOA (aporte del controlador XAI)

| Dimensión | Acepta Hₐ (mejora signif.) | No rechaza H₀ |
|---|---|---|
| D10 | 5 / 12 | 7 / 12 |
| D20 | 6 / 12 | 6 / 12 |
| **Global** | **11 / 24** | 13 / 24 |

Funciones donde SHOA-COMBINED mejora significativamente a SHOA:
- **D10:** F1, F2, F3, F9, F11
- **D20:** F1, F2, F3, F5, F9, F11

> **Interpretación:** el controlador XAI aporta una mejora estadísticamente
> significativa sobre la metaheurística base en casi la mitad de los casos, sin
> degradarla significativamente en el resto.

### 5.2 SHOA-COMBINED vs PSO (línea base clásica)

| Dimensión | Acepta Hₐ (mejora signif.) | No rechaza H₀ |
|---|---|---|
| D10 | 0 / 12 | 12 / 12 |
| D20 | 1 / 12 (F10) | 11 / 12 |
| **Global** | **1 / 24** | 23 / 24 |

> **Interpretación:** PSO es una línea base muy fuerte en CEC2022; SHOA-COMBINED solo
> lo supera significativamente en D20·F10. Este resultado contextualiza el alcance de
> la propuesta: el aporte del XAI se mide mejor contra SHOA (misma base algorítmica).

---

## 6. Hallazgos clave

1. **El controlador XAI funciona como mejora sobre SHOA:** 11 de 24 contrastes aceptan
   Hₐ frente a SHOA, principalmente en funciones unimodales/básicas (F1–F3) y algunas
   multimodales (F9, F11).
2. **PSO domina en CEC2022:** la mejora frente a PSO es marginal (solo D20·F10), lo que
   sugiere que la comparación más informativa para aislar el efecto del XAI es contra
   SHOA.
3. **No-normalidad generalizada (80.6 %):** valida metodológicamente el uso de Wilcoxon.
4. **Spot-check de coherencia:** en D10·F1, PSO alcanza error ≈ 0, por lo que
   SHOA-COMBINED (≈ 305) correctamente **no** lo supera (No rechaza H₀).

---

## 7. Artefactos generados

Ubicación: `Resultados/experiments/cec2022_failsafe/reports/presentation/`

### Tablas (`tables/`)
| Archivo | Contenido |
|---|---|
| `results_summary_D10.csv`, `results_summary_D20.csv` | best / mean / std / median / worst por función para los 3 algoritmos + mejor algoritmo |
| `normality_tests.csv` | Shapiro-Wilk (W, p) + Lilliefors (D, p) por (algoritmo, dim, función) — 72 filas |
| `normality_summary.csv` | porcentajes agregados de normalidad |
| `wilcoxon_combined_vs_shoa.csv` | Wilcoxon por función vs SHOA — 24 filas |
| `wilcoxon_combined_vs_pso.csv` | Wilcoxon por función vs PSO — 24 filas |
| `wilcoxon_combined_vs_*_summary.csv` | conteo Acepta Hₐ / No rechaza H₀ por dimensión |
| `convergence_combined_manifest.csv` | índice de curvas generadas |
| `boxplots_manifest.csv` | índice de boxplots generados |

### Gráficos (`plots/`, DPI 300)
- **24 curvas de convergencia** de SHOA-COMBINED — `convergencia_combined_D{dim}_F{fid}.png`
  (media sobre 30 runs + banda ±1 desv. estándar, eje Y logarítmico).
- **24 boxplots** comparativos PSO vs SHOA vs SHOA-COMBINED — `boxplot_D{dim}_F{fid}.png`.

---

## 8. Reproducibilidad

```bash
cd Final-Implementation
../.venv/bin/python presentation_reports.py \
    --output-root ../Resultados/experiments/cec2022_failsafe
```

- **No re-ejecuta** la optimización: reutiliza los datos ya completos (6 jobs).
- Lee `reports/tables/per_run_errors.csv` y los `full_output.csv` de cada corrida.
- Dependencia añadida: `statsmodels` (para Lilliefors).
- Nota: los `run_dir` se descubren por patrón bajo `--output-root` (la columna
  `run_dir` del CSV apunta a rutas de otra máquina y no se usa).
