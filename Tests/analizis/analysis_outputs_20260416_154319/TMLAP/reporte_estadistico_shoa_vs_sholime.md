# Reporte Estadistico: SHOA vs SHOLIME (TMLAP)

Fecha: 2026-04-16

## 1) Objetivo

Evaluar si la diferencia de rendimiento entre SHOA y SHOLIME en TMLAP es estadisticamente significativa.

## 2) Hipotesis

- H0: No hay diferencia significativa entre SHOA y SHOLIME.
- H1: Si hay diferencia significativa entre SHOA y SHOLIME.

Nivel de significancia: alpha = 0.05

## 3) Metodologia

1. Se uso fitness final por corrida (ultima iteracion por run_id/seed).
2. Se emparejaron datos por: function, dimension, profile, run_id, seed.
3. Paso 1 (normalidad): Test de Shapiro-Wilk sobre SHOA, SHOLIME y delta (SHOLIME - SHOA).
4. Paso 2 (contraste): Test de Wilcoxon pareado, bilateral (two-sided).

Total de pares usados: 270

## 4) Verificacion de Normalidad (Shapiro-Wilk)

| function | dimension | profile | n_pairs | p_shoa | estado_shoa | p_sholime | estado_sholime | p_delta | estado_delta |
|---|---:|---|---:|---:|---|---:|---|---:|---|
| 1.instancia_simple | 6 | hard | 30 | NaN | constante | NaN | constante | NaN | constante |
| 1.instancia_simple | 6 | medium | 30 | NaN | constante | NaN | constante | NaN | constante |
| 1.instancia_simple | 6 | soft | 30 | NaN | constante | NaN | constante | NaN | constante |
| 2.instancia_mediana | 12 | hard | 30 | 0.033787545931755654 | ok | 0.023876792483194843 | ok | 0.10910587037069272 | ok |
| 2.instancia_mediana | 12 | medium | 30 | 0.016120080205728658 | ok | 0.49572819696747333 | ok | 0.06646269633091825 | ok |
| 2.instancia_mediana | 12 | soft | 30 | 0.3936663557925795 | ok | 0.10673265721932042 | ok | 0.13253844391236397 | ok |
| 3.instancia_dura | 24 | hard | 30 | 0.018014820616772308 | ok | 0.010743940071579439 | ok | 0.15006584605888681 | ok |
| 3.instancia_dura | 24 | medium | 30 | 0.00012180420802680065 | ok | 0.0009263685496742326 | ok | 0.008806458392262602 | ok |
| 3.instancia_dura | 24 | soft | 30 | 0.00023769905349342504 | ok | 0.0022014978183243408 | ok | 0.18014848901236383 | ok |

Interpretacion del paso 1:

- En varias series p < 0.05 (no normalidad) y en otras hay valores constantes.
- Esto justifica usar un test no parametrico para comparacion pareada.

## 5) Wilcoxon Pareado por Perfil

| function | dimension | profile | n_pairs | wilcoxon_stat | p_value | decision (alpha=0.05) | mediana_delta (SHOLIME-SHOA) | media_delta (SHOLIME-SHOA) | tendencia |
|---|---:|---|---:|---:|---:|---|---:|---:|---|
| 1.instancia_simple | 6 | hard | 30 | 232.5 | 1.0 | No rechazar H0 | 0.0 | 0.0 | Empate (mediana delta = 0) |
| 1.instancia_simple | 6 | medium | 30 | 232.5 | 1.0 | No rechazar H0 | 0.0 | 0.0 | Empate (mediana delta = 0) |
| 1.instancia_simple | 6 | soft | 30 | 232.5 | 1.0 | No rechazar H0 | 0.0 | 0.0 | Empate (mediana delta = 0) |
| 2.instancia_mediana | 12 | hard | 30 | 232.5 | 1.0 | No rechazar H0 | 0.0 | 0.03333333333333333 | Empate (mediana delta = 0) |
| 2.instancia_mediana | 12 | medium | 30 | 210.0 | 0.6398739285968992 | No rechazar H0 | 1.0 | 0.5 | SHOA mejor (menor fitness) |
| 2.instancia_mediana | 12 | soft | 30 | 218.0 | 0.7649575823624072 | No rechazar H0 | 0.0 | -0.5 | Empate (mediana delta = 0) |
| 3.instancia_dura | 24 | hard | 30 | 224.5 | 0.8689707534027493 | No rechazar H0 | 0.0 | 0.0 | Empate (mediana delta = 0) |
| 3.instancia_dura | 24 | medium | 30 | 200.5 | 0.5088339023713164 | No rechazar H0 | -1.0 | -0.2 | SHOLIME mejor (menor fitness) |
| 3.instancia_dura | 24 | soft | 30 | 213.5 | 0.6951640914861223 | No rechazar H0 | 0.0 | -0.26666666666666666 | Empate (mediana delta = 0) |

## 6) Wilcoxon Pareado Global por Funcion (agregando perfiles)

| function | dimension | n_pairs | wilcoxon_stat | p_value | decision (alpha=0.05) | mediana_delta (SHOLIME-SHOA) | media_delta (SHOLIME-SHOA) | tendencia |
|---|---:|---:|---:|---:|---|---:|---:|---|
| 1.instancia_simple | 6 | 90 | 2047.5 | 1.0 | No rechazar H0 | 0.0 | 0.0 | Empate (mediana delta = 0) |
| 2.instancia_mediana | 12 | 90 | 1991.5 | 0.8209163160031171 | No rechazar H0 | 0.0 | 0.011111111111111112 | Empate (mediana delta = 0) |
| 3.instancia_dura | 24 | 90 | 1931.0 | 0.6383818023565493 | No rechazar H0 | 0.0 | -0.15555555555555556 | Empate (mediana delta = 0) |

## 7) Conclusiones

1. No se obtuvo ningun caso con p < 0.05 en Wilcoxon.
2. Por lo tanto, no se rechaza H0 en ninguna comparacion (ni por perfil ni global por funcion).
3. Con estos resultados, no hay evidencia estadistica de mejora significativa entre SHOA y SHOLIME en TMLAP.
4. Las diferencias observadas en medias son pequenas y compatibles con variacion estocastica.

## 8) Archivos base usados

- normality_shapiro_shoa_vs_sholime.csv
- wilcoxon_shoa_vs_sholime_by_profile.csv
- wilcoxon_shoa_vs_sholime_by_function.csv
