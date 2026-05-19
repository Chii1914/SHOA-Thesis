# Ecuaciones adaptativas del controlador explicadas en palabras

Fecha: 2026-05-18
Contexto: SHOA+LIME adaptativo (sin perfiles fijos)

## 1) Ventana adaptativa (w_win)

La ventana no es fija.
Se calcula segun la dificultad del problema: dimension, tamano de poblacion y presupuesto iterativo.

En palabras:
- Si el problema es mas grande o complejo, el controlador mira mas historia.
- Si el problema es mas pequeno, usa una ventana mas corta.

## 2) Escala robusta del fitness (S_f(t))

Se toma la ventana reciente del mejor fitness y se calcula una escala robusta.
Esa escala sirve para normalizar y comparar en terminos relativos.

En palabras:
- Evita usar umbrales absolutos que se rompen cuando cambia la escala del problema.

## 3) Indicador relativo de estancamiento (z_stag(t))

Mide cuanta variacion real hay en la ventana reciente y la divide por la escala robusta del fitness.

En palabras:
- Si el valor es bajo, hay poca variacion relativa y posible estancamiento.
- Si el valor es alto, todavia hay dinamica de mejora/exploracion.

## 4) Umbral adaptativo de estancamiento (tau_stag(t))

No se usa un umbral fijo.
El umbral se aprende online con cuantiles del historial reciente del indicador de estancamiento.

En palabras:
- El sistema aprende que es normal para ese problema en ese momento.
- El trigger se ajusta al contexto, no a un numero universal.

## 5) Regla de trigger

Se activa candidato de estancamiento cuando el indicador actual cae por debajo del umbral adaptativo.

En palabras:
- Se dispara cuando la dinamica actual es anormalmente plana respecto al pasado reciente.

## 6) Mejora esperada relativa (Delta_rel(t))

La mejora esperada que predice LIME se divide por la escala robusta del fitness.

En palabras:
- No interesa solo mejorar en valor bruto.
- Importa cuanto mejora en proporcion al nivel del problema.

## 7) Importancia adaptativa de pesos LIME

Cada peso de feature se evalua con dos condiciones:
1. Debe ser grande respecto a su historial (cuantil adaptativo).
2. Debe ser estadisticamente significativo (intervalo de confianza que no cruce cero).

En palabras:
- No basta con peso alto aislado.
- Debe ser alto y confiable.

## 8) Severidad del estancamiento (S_sev(t))

Convierte el nivel de estancamiento en una escala normalizada entre 0 y 1.

En palabras:
- 0: estancamiento debil.
- 1: estancamiento severo.

## 9) Evidencia explicable agregada (E(t))

Resume la fuerza media de las features relevantes y significativas.

En palabras:
- Cuantifica cuanta evidencia explicable hay para justificar intervencion.

## 10) Ganancia global del controlador (G(t))

Combina severidad de estancamiento y evidencia explicable, luego la acota entre minimo y maximo.

En palabras:
- Decide cuanta fuerza debe aplicar el controlador en esa iteracion.

## 11) Fuerza efectiva de rescate

La intensidad de rescate se calcula multiplicando una base por la ganancia adaptativa.

En palabras:
- Si hay estancamiento fuerte y evidencia fuerte, el rescate sube.
- Si la evidencia es debil, el rescate baja para no sobrerreaccionar.

## Resumen corto

El controlador adaptativo reemplaza reglas fijas por reglas relativas, contextuales y estadisticamente justificadas.
Eso mejora robustez entre problemas de distinta dimension y distinta escala de fitness.
