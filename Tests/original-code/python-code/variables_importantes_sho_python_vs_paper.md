# Variables importantes de SHO en Python y contraste con paper

Fecha: 2026-05-18

Este documento separa:
- SHO puro (equivalente al MATLAB original).
- SHO+LIME (extension con diagnostico y rescate).

Nota: el contraste del paper se hace con base en las ecuaciones ya reflejadas en el flujo del algoritmo (Eq.(4), Eq.(7), Eq.(10), Eq.(12), Eq.(13)).

## 1) SHO puro (mapeo principal)

| Variable (Python) | Rol | Evidencia | Mapeo con paper |
|---|---|---|---|
| pop, Max_iter, LB, UB, Dim, fobj | Parametros base del problema | SHO.py:18 | N, T, limites y funcion objetivo del algoritmo base |
| sea_horses | Poblacion actual | SHO.py:19 | Estado de seahorses en el espacio de busqueda |
| sea_horses_fitness | Fitness por agente | SHO.py:21 | Calidad individual para ranking |
| target_position | Mejor posicion global | SHO.py:34 | Elite global |
| target_fitness | Mejor fitness global | SHO.py:35 | Best-so-far |
| elite | Replica de la elite | SHO.py:48 | Referencia de atraccion en movimiento/depredacion |
| beta | Perturbacion gaussiana | SHO.py:47 | Parte estocastica (rama Eq.(7)) |
| step_length | Paso Levy | SHO.py:52, levy.py:10, levy.py:16 | Vuelo de Levy para exploracion |
| r1 | Selector rama de movimiento | SHO.py:57 | Activa Eq.(4) o Eq.(7) |
| theta,row,x,y,z | Intermedias de movimiento | SHO.py:59-63 | Geometria de la rama Eq.(4) |
| sea_horses_new1 | Estado tras motor behavior | SHO.py:53-71 | Salida de Eq.(4)/(7) |
| alpha | Factor temporal | SHO.py:78 | Balance exploracion-explotacion en Eq.(10) |
| r2 | Selector de depredacion | SHO.py:77-81 | Define variante de Eq.(10) |
| sea_horses_new2 | Estado tras predation | SHO.py:75-86 | Salida de Eq.(10) |
| sea_horses_father | Mitad superior (ranking) | SHO.py:92 | Eq.(12) |
| sea_horses_mother | Mitad complementaria | SHO.py:93 | Eq.(12) |
| r3 | Peso de recombinacion | SHO.py:97-98 | Eq.(13) |
| si / sea_horses_offspring | Descendencia | SHO.py:95-100 | Resultado de Eq.(13) |
| sea_horses_fitness_all | Fitness combinado | SHO.py:104 | Seleccion elitista |
| convergence_curve | Curva de convergencia | SHO.py:25, SHO.py:119 | Metrica de mejor fitness por iteracion |
| fitness_history | Historial fitness | SHO.py:22, SHO.py:112 | Trazabilidad poblacional |
| population_history | Historial de posiciones | SHO.py:23, SHO.py:113 | Trayectoria en el espacio |
| trajectories | Historial componente 1 | SHO.py:26, SHO.py:114 | Salida auxiliar para analisis |

## 2) Eq-paper vs bloques en Python

- Eq.(4) y Eq.(7): bloque de motor behavior en SHO.py:57-71.
- Eq.(10): bloque de predation behavior en SHO.py:81-85.
- Eq.(12): seleccion de fathers/mothers en SHO.py:92-93.
- Eq.(13): recombinacion en SHO.py:98.

## 3) Variables adicionales en SHO+LIME (no estaban en SHO puro)

### 3.1 Configuracion XAI y rescate

| Variable | Rol | Evidencia |
|---|---|---|
| window_size | Tamano de ventana de estancamiento | SHO_LIME_Controller.py:32 |
| epsilon_stagnation | Umbral de trigger por desviacion estandar | SHO_LIME_Controller.py:33 |
| cooldown_iters | Enfriamiento tras rescate | SHO_LIME_Controller.py:34 |
| lime_num_samples | Muestras para explicacion local | SHO_LIME_Controller.py:35 |
| importance_threshold | Umbral de importancia estocastica | SHO_LIME_Controller.py:36 |
| delta_tolerance | Umbral de mejora esperada baja | SHO_LIME_Controller.py:37 |
| fidelity_threshold | Umbral minimo de fidelidad local | SHO_LIME_Controller.py:38 |
| rescue_mode | Modo de rescate (levy_teleport o leader_repulsion) | SHO_LIME_Controller.py:40 |
| rescue_eta | Intensidad en leader_repulsion | SHO_LIME_Controller.py:41 |
| rescue_levy_scale | Escala de salto Levy en rescate | SHO_LIME_Controller.py:42 |
| rescue_patience_iters | Paciencia antes de evaluar rollback | SHO_LIME_Controller.py:43 |
| rescue_min_improvement | Mejora minima requerida | SHO_LIME_Controller.py:44 |
| enforce_elite_archive | Fuerza preservacion de mejor historico | SHO_LIME_Controller.py:45 |

### 3.2 Estado/log de diagnostico

| Variable | Rol | Evidencia |
|---|---|---|
| decision_history | Historial de variables de decision | SHO_LIME_Controller.py:413 |
| memory_window | Ventana de best fitness para trigger | SHO_LIME_Controller.py:414 |
| trigger_candidate | Candidato a diagnostico por estancamiento | SHO_LIME_Controller.py:467 |
| diagnostics_log | Bitacora de diagnosticos LIME | SHO_LIME_Controller.py:417 |
| diagnostics_invocation_iterations | Iteraciones donde se invoca LIME | SHO_LIME_Controller.py:418 |
| rescue_count | Numero de rescates aplicados | SHO_LIME_Controller.py:421 |
| rollback_count | Numero de rollbacks tras rescate no efectivo | SHO_LIME_Controller.py:422 |
| rescue_trial_state | Snapshot para validar mejora post-rescate | SHO_LIME_Controller.py:423 |

### 3.3 Features explicables (LIME)

| Variable | Rol | Evidencia |
|---|---|---|
| FEATURE_NAMES | Variables explicativas del salto | SHO_LIME_Controller.py:14 |
| r1 | Control rama motor | SHO_LIME_Controller.py:80 |
| mag_browniano | Magnitud estocastica browniana | SHO_LIME_Controller.py:80 |
| mag_levy | Magnitud estocastica Levy | SHO_LIME_Controller.py:80 |
| r2 | Selector depredacion | SHO_LIME_Controller.py:80 |
| mag_predacion | Intensidad en depredacion | SHO_LIME_Controller.py:80 |

## 4) Conclusion rapida

1. Si analizas SHO.py, el mapeo de variables es esencialmente el mismo que en MATLAB (nombres adaptados a snake_case).
2. Si analizas SHO_LIME_Controller.py, el mapeo base se mantiene, pero se agregan variables nuevas de diagnostico, interpretabilidad y rescate.
3. Para tesis: usa dos tablas separadas (SHO puro y SHO+LIME) para no mezclar variables del algoritmo original con variables del controlador XAI.