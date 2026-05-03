# Explicacion detallada de la configuracion SHOXAIConfig

## Contexto de tu ejecucion actual

En tu archivo main_SHO.py, esta configuracion se usa para correr SHOA con controlador en linea basado en diagnostico LIME sobre la funcion F92022 con dimension 10.

Valores actuales:

- pop_size = popsize (en tu main, popsize = 30)
- max_iter = max_iter (en tu main, max_iter = 500)
- window_size = 15
- epsilon_stagnation = 2e-4
- cooldown_iters = 14
- lime_num_samples = 1600
- importance_threshold = 0.05
- delta_tolerance = 2e-5
- fidelity_threshold = 0.15
- rescue_mode = levy_teleport
- rescue_levy_scale = 0.14
- seed = 7

## Que hace cada parametro

## 1) pop_size

Controla cuantos agentes (sea-horses) hay en la poblacion.

- Mas alto:
  - Mas exploracion del espacio de busqueda.
  - Menor riesgo de convergencia prematura.
  - Mayor costo computacional por iteracion.
- Mas bajo:
  - Menor costo por iteracion.
  - Mayor riesgo de estancarse rapido.

## 2) max_iter

Controla cuantas iteraciones maximas ejecuta el algoritmo.

- Mas alto:
  - Da mas tiempo para mejorar y recuperarse tras rescates.
  - Aumenta tiempo total de corrida.
- Mas bajo:
  - Corridas mas rapidas.
  - Puede cortar el proceso antes de converger bien.

## 3) window_size

Tamano de la ventana deslizante de fitness usada para detectar estancamiento.

- La logica de gatillo se basa en la desviacion estandar de esta ventana.
- Mas alto:
  - Gatillo mas estable y menos sensible al ruido corto.
  - Diagnosticos menos frecuentes.
- Mas bajo:
  - Detecta estancamiento mas rapido.
  - Puede aumentar falsos positivos de estancamiento.

## 4) epsilon_stagnation

Umbral numerico para decidir si hay estancamiento.

- Se considera candidato a estancamiento cuando std(ventana) < epsilon_stagnation.
- Mas alto:
  - Mas sensible (diagnostica estancamiento con mayor facilidad).
- Mas bajo:
  - Mas estricto (requiere ventana mucho mas plana para disparar).

## 5) cooldown_iters

Numero de iteraciones de enfriamiento despues de aplicar un rescate.

- Durante cooldown, el controlador no vuelve a diagnosticar.
- Mas alto:
  - Evita sobre-diagnostico y rescates encadenados.
  - Puede demorar una reaccion necesaria.
- Mas bajo:
  - Reacciona mas seguido.
  - Puede generar rescates demasiado frecuentes.

## 6) lime_num_samples

Numero de perturbaciones que usa LIME para explicar la decision local.

- Mas alto:
  - Explicacion mas estable y generalmente mas fiable.
  - Mucho mayor costo computacional.
- Mas bajo:
  - Diagnostico mas rapido.
  - Explicacion mas ruidosa o inestable.

## 7) importance_threshold

Umbral de importancia para considerar dominancia estocastica en las variables de vuelo.

- En el controlador se evalua principalmente peso de mag_browniano y mag_predacion.
- Mas bajo:
  - Facilita marcar importancia estocastica fuerte.
  - Aumenta probabilidad de diagnostico positivo.
- Mas alto:
  - Exige evidencia explicativa mas fuerte.
  - Reduce positivos.

## 8) delta_tolerance

Umbral de mejora esperada minima (prediccion local de LIME).

- Si la mejora esperada predicha es menor que delta_tolerance, se interpreta como baja mejora.
- Mas alto:
  - Mas facil clasificar como baja mejora (mas positivos potenciales).
- Mas bajo:
  - Mas estricto para declarar falta de mejora.

## 9) fidelity_threshold

Umbral minimo de fidelidad local (score de explicacion LIME).

- Sirve para filtrar explicaciones poco confiables.
- Mas alto:
  - Pide explicaciones mas confiables para habilitar accion.
  - Menos positivos.
- Mas bajo:
  - Acepta explicaciones mas debiles.
  - Puede aumentar positivos pero con mas riesgo.

## 10) rescue_mode

Define la estrategia de rescate cuando hay diagnostico positivo.

- levy_teleport:
  - Aplica saltos tipo Levy alrededor de la zona del lider.
  - Bueno para escapar de pozos locales con cambios no lineales.
- leader_repulsion:
  - Empuja en direccion de repulsion del lider respecto al centroide.
  - Mas direccional.

## 11) rescue_levy_scale

Escala (magnitud) del salto Levy cuando rescue_mode = levy_teleport.

- Mas alto:
  - Saltos mas grandes (escape fuerte).
  - Riesgo de perder refinamiento local.
- Mas bajo:
  - Saltos mas suaves.
  - Puede no salir de estancamientos fuertes.

## 12) seed

Semilla aleatoria para reproducibilidad.

- Misma seed + mismos parametros => comportamiento repetible (salvo efectos de entorno o librerias).
- Cambiar seed permite estimar robustez estadistica entre corridas.

## Como se combinan para invocar diagnostico y rescate

En terminos operativos, el flujo es:

1. Se monitorea una ventana de fitness de tamano window_size.
2. Si la variacion de esa ventana cae bajo epsilon_stagnation, se invoca diagnostico LIME.
3. El diagnostico se considera positivo si simultaneamente:
   - hay importancia estocastica suficiente (importance_threshold),
   - la mejora esperada es baja (delta_tolerance),
   - la fidelidad explicativa cumple (fidelity_threshold).
4. Si es positivo, se aplica rescate segun rescue_mode y rescue_levy_scale.
5. Se bloquean nuevos diagnosticos por cooldown_iters.

## Lectura de tu configuracion actual

Tu set actual esta orientado a:

- sensibilidad media-alta para detectar estancamiento (window_size 15 con epsilon 2e-4),
- explicaciones relativamente densas (1600 muestras),
- criterio de activacion flexible (importance_threshold 0.05, fidelity_threshold 0.15),
- rescate con saltos Levy moderados (0.14),
- trazabilidad reproducible (seed 7).

En palabras simples: estas priorizando detectar y reaccionar al estancamiento de manera frecuente, manteniendo control para no rescatar en cada iteracion gracias al cooldown.
