# Guia Completa del Codigo SHOA-COMBINED

## 1. Objetivo del modulo

SHOA-COMBINED integra en una sola corrida:

- Optimizacion poblacional tipo Sea-Horse Optimizer (SHO).
- Deteccion de estancamiento con criterio paper-style basado en SFEs.
- Diagnostico explicable con LIME en linea.
- Politica de reinicio parcial de poblacion guiada por LIME, con fallback aleatorio valido.

El resultado es un pipeline unico que optimiza, detecta estancamiento, explica comportamiento y aplica reinicios controlados.

## 2. Estructura de archivos

- run_cec2022_combined.py
Runner principal. Parsea argumentos, configura experimento, ejecuta funciones/runs, y escribe artefactos.

- SHO_HYBRID_Controller.py
Nucleo del algoritmo hibrido. Contiene el loop de iteraciones, detector de estancamiento, logica LIME, warm-up, cooldown y reinicio.

- lime_diagnostic.py
Seleccion estratificada de agentes para explicabilidad y construccion de explicaciones LIME para dos targets.

- stagnation_detector.py
Implementa el detector MinSFEs + MaxFEs y emite eventos de estado (stagnation_start, recovered).

- global_explanations.py
Agrega contribuciones locales de LIME en metricas globales If y Sf (global y ventanas temporales).

- opfunu_wrapper.py
Adaptador para funciones CEC2022 de opfunu y parser de ids de funciones.

- initialization.py
Inicializacion de poblacion dentro de cotas.

- levy.py
Generador de saltos Levy usados en el movimiento SHO.

- utils_logging.py
Utilidades de escritura de CSV/JSON y resumen por funcion.

- plot_combined_run.py
Script de postproceso para graficar convergencia, estancamiento, LIME general, LIME en reinicios y full log.

- README.md
Guia corta de uso.

- __init__.py
Archivo de paquete (vacio).

## 3. Flujo de alto nivel

### 3.1 Fase de runner

1. Se parsean argumentos.
2. Se validan argumentos criticos del reinicio.
3. Se calculan parametros efectivos:
- MaxFEs efectivo.
- warm-up en FEs (5 por ciento de MaxFEs).
- cooldown de reinicio en FEs.
- cadence legacy de LIME (informativa, sin trigger periodico real).
4. Se ejecuta una malla de:
- funciones CEC seleccionadas
- numero de runs
5. Cada corrida llama al controlador hibrido.
6. Se agregan y escriben artefactos.

### 3.2 Fase de controlador hibrido

Por iteracion:

1. Movimiento SHO (rama helicoidal/Levy o Browniano).
2. Predacion SHO.
3. Reproduccion SHO.
4. Seleccion de nueva poblacion.
5. Actualizacion detector de estancamiento.
6. Compuertas por warm-up y estado stagnated.
7. Trigger LIME solo en stagnation_start cuando hay muestras suficientes.
8. Intento de reinicio parcial en stagnation_start, sujeto a cooldown.
9. Registro exhaustivo de full_output, history y events.

## 4. Detector de estancamiento

Implementado en stagnation_detector.py.

Estado principal:

- best_fitness
- last_improvement_fe
- sfes
- min_sfes
- stagnated

Regla:

- sfes = fe - last_improvement_fe
- hay estancamiento cuando sfes >= min_sfes

Eventos emitidos:

- initialized
- none
- stagnation_start
- recovered

## 5. Explicabilidad LIME

Implementado en lime_diagnostic.py + llamado desde SHO_HYBRID_Controller.py.

### 5.1 Seleccion de agentes

Se usa muestreo estratificado por categorias:

- elite_high_impact: 4 por ciento
- diverse: 3 por ciento
- outliers: 2 por ciento
- random: 1 por ciento

Con minimo 1 por categoria.

### 5.2 Targets explicados

Se generan explicaciones para dos canales:

- classification_improved
- regression_y_reg

### 5.3 Modo de explicacion

- selected_agents
Explica todos los agentes seleccionados.

- medoid
Explica solo el representante medoid del conjunto seleccionado.

## 6. Politica de warm-up, LIME y reinicio

### 6.1 Warm-up

Durante warm-up (5 por ciento de MaxFEs efectivos):

- Solo corre SHO.
- No se acumula dataset LIME.
- No se dispara LIME.
- No se intenta reinicio.

### 6.2 Trigger LIME

Fuera de warm-up:

- dataset se acumula cuando no hay estancamiento, o en el instante stagnation_start.
- LIME solo se dispara en stagnation_start.
- Si no hay min_samples, no dispara.

### 6.3 Reinicio parcial por estancamiento

Se intenta en stagnation_start cuando restart_enabled = 1.

1. Se verifica cooldown separado en FEs.
2. Si no se cumple cooldown, se registra bloqueo.
3. Si se cumple, se decide fuente de reinicio:

- lime_mutator
Si la dominancia fusionada de una feature >= umbral.

- random_fallback
Si no hay feature elegible.

4. Se seleccionan agentes peores por fitness preservando elite.
5. Se reinicia solo el subconjunto parcial.

### 6.4 Dominancia fusionada de LIME

Se fusionan importancias relativas por feature en ambos canales LIME.

- Se normaliza por target.
- Se promedia por cantidad de canales donde aparece la feature.
- Se elige feature dominante con score maximo.

### 6.5 Mutadores por feature

Mapeo principal:

- mag_levy* -> perturbacion Levy
- mag_browniano* -> ruido gaussiano escalado
- mag_predacion* o alpha -> mezcla hacia elite con ruido
- distance_to_elite, delta_position_norm, theta_mean, theta_active, r1, r2 -> paso direccional respecto a elite
- otros casos -> fallback aleatorio

### 6.6 Fallback valido de reinicio

Cuando no hay feature LIME elegible:

x = LB + (UB - LB) * rand[0,1]

Aplicado al subconjunto de reinicio (no a toda la poblacion).

## 7. Artefactos de salida

El runner genera en outputs/run-YYYY-MM-DD-HH-MM-SS:

- config_used.json
Configuracion usada, incluyendo bloque restart.

- runs_raw.csv
Resumen por run con fitness final, eventos y tiempos.

- full_output.csv
Telemetria por iteracion.
Incluye convergencia, estancamiento, warm-up, estado LIME y estado de reinicio.

- lime_contributions.csv
Contribuciones por diagnosis/feature/target.

- global_feature_explanations.csv
Agregados If/Sf globales y por ventanas.

- stagnation_history.csv
Historia por iteracion del detector.

- stagnation_events.csv
Eventos de estancamiento y eventos auxiliares (warmup_completed, restart_executed, restart_blocked_cooldown).

- summary_by_function.csv
Estadisticos agregados por funcion.

## 8. Parametros modificables

## 8.1 CLI del runner

Archivo: run_cec2022_combined.py

### Experimento base

- --functions (default: all)
Seleccion de funciones CEC2022. Ejemplos: all, 1, 1,3,5, 1-12.

- --dim (default: 10)
Dimension del problema.

- --pop (default: 50)
Tamano de poblacion.

- --max-iter (default: 500)
Iteraciones maximas por run.

- --runs (default: 30)
Numero de corridas por funcion.

- --seed (default: 42)
Semilla base.

### LIME

- --lime-every (default: None)
Parametro legacy de cadencia. Se conserva para metadata/compatibilidad.

- --lime-min-samples (default: 1000)
Minimo de muestras acumuladas antes de permitir trigger LIME.

- --stagnation-lime-selection-mode (default: medoid)
Modo actual recomendado para explicacion en trigger por estancamiento.

- --lime-selection-mode (default: None)
Alias legacy de selection mode.

### Estancamiento

- --min-sfes-ratio (default: 0.04)
Ratio para MinSFEs.

- --max-fes (default: 0)
Presupuesto global FE. Si 0 se autoestima.

### Reinicio

- --restart-enabled (default: 1)
Activa/desactiva reinicio parcial por estancamiento.

- --restart-percent (default: 7.0)
Porcentaje de poblacion a reiniciar. Debe estar en [5, 10].

- --restart-cooldown-fes-ratio (default: None)
Ratio para cooldown en FE. Si None, usa min-sfes-ratio.

- --restart-dominance-threshold (default: 0.90)
Umbral de dominancia fusionada para usar mutador por feature LIME.

### Logging y salida

- --progress-every (default: 10)
Frecuencia de log iterativo.

- --output-dir (default: outputs)
Directorio base de salida.

- --log-level (default: INFO)
Nivel de logging.

- --quiet
Si se activa, reduce logging iterativo del controlador.

## 8.2 Parametros programaticos del controlador

Archivo: SHO_HYBRID_Controller.py, funcion SHO_HYBRID.

Parametros mas relevantes:

- pop, max_iter, lower_bound, upper_bound, dim, fobj
- random_state
- min_samples_before_lime
- lime_selection_mode
- min_sfes_ratio, max_fes
- warmup_fes
- restart_enabled
- restart_percent
- restart_cooldown_fes
- restart_dominance_threshold
- enable_lime
- enable_stagnation
- progress_every
- verbose

## 8.3 Parametros del script de graficos

Archivo: plot_combined_run.py

- --run-dir
Directorio de corrida a graficar.

- --top-k-temporal (default: 8)
Numero de features para curva temporal.

- --target-fitness (default: 0.0)
Linea horizontal objetivo en convergencia.

- --full-log-file (default: vacio)
Ruta opcional para guardar el reporte full log.
Si no se define, se crea en plots/full_log_report.txt.

- --log-y
Escala logaritmica eje Y.

- --show
Muestra figuras en ventana interactiva.

## 9. Validaciones y restricciones actuales

- restart_percent debe estar en [5,10].
- restart_dominance_threshold debe estar en (0,1].
- restart_cooldown_fes_ratio, si se define, debe ser > 0.
- function_id para CEC debe estar en [1,12].
- lime_selection_mode debe ser selected_agents o medoid.

## 10. Notas operativas importantes

- El contador FE proviene del wrapper CEC y sube en cada evaluacion.
- Los reinicios consumen FE reales porque reevaluan agentes reiniciados.
- El evento warmup_completed se registra en stagnation_events.
- El conteo de stagnation_events en runs_raw contabiliza solo stagnation_start y recovered para mantener lectura historica estable.
- config_used.json guarda tanto inputs como efectivos para reproducibilidad.

## 11. Recomendaciones de uso rapido

- Analisis rapido conservador:
pop medio, restart_enabled=1, restart_percent=7, threshold 0.90.

- Forzar mas fallback aleatorio:
subir threshold cerca de 1.0.

- Forzar mas mutador por feature:
bajar threshold (por ejemplo 0.1 a 0.3, solo para pruebas).

- Reducir costo de LIME:
selection mode medoid y lime-min-samples alto.

## 12. Resumen corto

Este modulo implementa una estrategia online completa de optimizacion + diagnostico + control adaptativo. El comportamiento se regula desde CLI, con trazabilidad fuerte en CSV/JSON y visualizacion unificada en plot_combined_run.py.
