# Guia de Implementacion: Controlador de Estancamiento con SHO y LIME

Aqui tienes un pseudocodigo completo, alineado con tus 5 fases, antes de implementar.

## Pseudocodigo Maestro (SHO + Trigger + LIME + Rescate)

```text
ALGORITMO SHO_XAI_STAGNATION_CONTROLLER

ENTRADAS:
  cec_class_name
  ndim
  pop_size
  max_iter
  window_size
  epsilon_stagnation
  cooldown_iters
  lime_num_samples
  w_importance_threshold
  delta_tolerance
  rescue_mode   # "levy_teleport" o "leader_repulsion"

SALIDAS:
  best_position
  best_fitness
  convergence_curve
  diagnostics_log

FASE 1: PREPARACION
1) instalar/verificar librerias: numpy, opfunu, lime
2) objective = crear_funcion_cec_opfunu(cec_class_name, ndim)
3) bounds = [objective.lb, objective.ub]
4) definir estructura AGENTE:
     position
     fitness
     flight_log = [r1, mag_browniano, mag_levy, r2, mag_predacion]
5) inicializar poblacion factible y evaluar fitness
6) g_best = mejor agente inicial
7) memory_window = deque(maxlen=window_size)
8) decision_history = deque(maxlen=H)   # historial para entrenar LimeTabularExplainer
9) cooldown_counter = 0
10) convergence_curve = []
11) diagnostics_log = []

FASE 2: BUCLE SHO MODIFICADO
PARA iter EN 1..max_iter HACER:

  # 2.1 Movimiento + predacion con variables estocasticas desacopladas
  PARA cada agente i EN poblacion HACER:
    decision_i = sample_decision_variables(iter, i)
      # decision_i = [r1, mag_browniano, mag_levy, r2, mag_predacion]
    i.flight_log = decision_i
    decision_history.append(decision_i)

    x_motor = SHO_motor_step(i.position, g_best.position, decision_i, iter)
    x_motor = clip_to_bounds(x_motor, bounds)

    x_pred = SHO_predation_step(x_motor, g_best.position, decision_i, iter)
    x_pred = clip_to_bounds(x_pred, bounds)

    i.position_candidate = x_pred
    i.fitness_candidate = objective.evaluate(x_pred)

  # 2.2 Reproduccion y seleccion
  poblacion = SHO_reproduction_and_selection(poblacion)

  # actualizar fitness reales post-seleccion
  PARA cada agente i EN poblacion HACER:
    i.fitness = objective.evaluate(i.position)

  # actualizar lider
  g_best = argmin_fitness(poblacion)

  convergence_curve.append(g_best.fitness)
  memory_window.append(g_best.fitness)

  # 2.3 Trigger con ventana deslizante
  SI cooldown_counter > 0 ENTONCES
    cooldown_counter = cooldown_counter - 1
  SINO
    SI len(memory_window) == window_size Y std(memory_window) < epsilon_stagnation ENTONCES

      # FASE 4: Diagnostico XAI
      diagnosis = RUN_LIME_DIAGNOSIS(
                    g_best,
                    objective,
                    decision_history,
                    lime_num_samples,
                    w_importance_threshold,
                    delta_tolerance
                  )

      diagnostics_log.append(diagnosis)

      SI diagnosis.status == "POSITIVE_STAGNATION" ENTONCES

        # FASE 5: Protocolo de rescate
        poblacion = APPLY_RESCUE_MUTATION(
                     poblacion,
                     g_best,
                     bounds,
                     rescue_mode
                   )

        # reevaluar y refrescar lider
        PARA cada agente i EN poblacion HACER:
          i.fitness = objective.evaluate(i.position)
        g_best = argmin_fitness(poblacion)

        # cooldown
        clear(memory_window)
        cooldown_counter = cooldown_iters
      FIN SI
    FIN SI
  FIN SI

FIN PARA

RETORNAR g_best.position, g_best.fitness, convergence_curve, diagnostics_log
```

## Funciones clave (pseudocodigo detallado)

```text
FUNCION sample_decision_variables(iter, agent):
  r1 = random_normal(0, 1)
  mag_browniano = abs(random_normal(0, sigma_b))
  mag_levy = abs(sample_levy(alpha=1.5, scale=s_levy))
  r2 = random_uniform(0, 1)
  mag_predacion = random_uniform(0, 1) * alpha_schedule(iter)
  RETORNAR [r1, mag_browniano, mag_levy, r2, mag_predacion]
FIN FUNCION
```

```text
FUNCION SHO_motor_step(x, x_best, decision, iter):
  [r1, mag_browniano, mag_levy, r2, mag_predacion] = decision

  # usar tu ecuacion SHO real de movimiento
  # ejemplo conceptual: combinacion browniana + levy hacia elite
  step_brown = mag_browniano * random_direction(dim(x))
  step_levy = mag_levy * levy_direction(dim(x))
  x_new = x + phi1(r1, iter) * step_brown + phi2(r1, iter) * step_levy + phi3(iter) * (x_best - x)

  RETORNAR x_new
FIN FUNCION
```

```text
FUNCION SHO_predation_step(x_motor, x_best, decision, iter):
  [r1, mag_browniano, mag_levy, r2, mag_predacion] = decision

  # usar tu ecuacion SHO real de predacion
  x_new = x_motor + psi1(r2, iter) * mag_predacion * (x_best - x_motor)

  RETORNAR x_new
FIN FUNCION
```

```text
FUNCION RUN_LIME_DIAGNOSIS(g_best, objective, decision_history, lime_num_samples,
                           w_importance_threshold, delta_tolerance):

  # 1) construir datos de referencia para explainer
  # cada fila: [r1, mag_browniano, mag_levy, r2, mag_predacion]
  training_data = array_from(decision_history)
  SI training_data tiene pocas filas ENTONCES
    training_data = synthetic_samples_in_valid_ranges()

  # 2) capturar estado fijo del lider
  fixed_context.position = copy(g_best.position)
  fixed_context.old_fitness = g_best.fitness
  fixed_context.iter_context = contexto_actual

  # 3) wrapper para LIME
  FUNCION lime_wrapper(X_perturb_matrix):
    deltas = vector_vacio(len(X_perturb_matrix))
    PARA k EN 1..len(X_perturb_matrix) HACER:
      decision_k = X_perturb_matrix[k]

      x0 = copy(fixed_context.position)
      x1 = SHO_motor_step(x0, x0, decision_k, fixed_context.iter_context)
      x1 = clip_to_bounds(x1, [objective.lb, objective.ub])

      x2 = SHO_predation_step(x1, x0, decision_k, fixed_context.iter_context)
      x2 = clip_to_bounds(x2, [objective.lb, objective.ub])

      new_f = objective.evaluate(x2)
      delta = fixed_context.old_fitness - new_f   # mejora si delta > 0
      deltas[k] = delta
    FIN PARA
    RETORNAR deltas
  FIN FUNCION

  # 4) ejecutar LIME en la ultima decision del lider
  x_instance = g_best.flight_log
  explainer = LimeTabularExplainer(
                training_data,
                mode="regression",
                feature_names=["r1","mag_browniano","mag_levy","r2","mag_predacion"]
              )
  exp = explainer.explain_instance(x_instance, lime_wrapper,
                                   num_features=5, num_samples=lime_num_samples)

  # 5) extraer pesos
  weights = parse_feature_weights(exp)  # dict feature -> peso
  pred_delta = get_local_prediction(exp)  # delta esperado por modelo local

  # 6) regla de confirmacion
  strong_stochastic_importance =
    abs(weights["mag_browniano"]) > w_importance_threshold
    O
    abs(weights["mag_predacion"]) > w_importance_threshold

  low_expected_improvement = pred_delta < delta_tolerance

  SI strong_stochastic_importance Y low_expected_improvement ENTONCES
    status = "POSITIVE_STAGNATION"
  SINO
    status = "FALSE_ALARM"
  FIN SI

  RETORNAR {
    "status": status,
    "weights": weights,
    "pred_delta": pred_delta
  }
FIN FUNCION
```

```text
FUNCION APPLY_RESCUE_MUTATION(poblacion, g_best, bounds, rescue_mode):

  SI rescue_mode == "levy_teleport" ENTONCES
    k = floor(0.5 * len(poblacion))
    idx = seleccionar_aleatorio_indices(k)
    PARA cada j EN idx HACER:
      jump = levy_vector(dim(g_best.position), alpha=1.5, scale=s_rescue)
      poblacion[j].position = clip_to_bounds(g_best.position + jump, bounds)
    FIN PARA
  FIN SI

  SI rescue_mode == "leader_repulsion" ENTONCES
    centroid = promedio_posiciones(poblacion)
    dir = normalize(g_best.position - centroid)
    g_best.position = clip_to_bounds(g_best.position + eta_repulsion * dir, bounds)
  FIN SI

  RETORNAR poblacion
FIN FUNCION
```

## Condicion Trigger y Regla XAI en forma compacta

- Trigger: activar posible estancamiento si len(window)=N y sigma(window) < epsilon.
- Confirmacion XAI:
  - alta importancia en magnitudes estocasticas: |w_mag_browniano| o |w_mag_predacion| grande
  - y mejora local esperada baja: DeltaFitness_predicha < tau
  - entonces diagnostico positivo y rescate.

Si te parece bien este pseudocodigo, en el siguiente paso se implementa en Python modular, conectado a Opfunu CEC 2022 y LIME.

## Anexo: Revision de retroalimentacion e implementacion

### Fortalezas validadas

1. Desacoplamiento estocastico con `sample_decision_variables`: excelente para trazabilidad y explicabilidad con LIME.
2. `cooldown_counter`: evita re-disparo inmediato del trigger despues del rescate.
3. Uso de `x0, x0` en `lime_wrapper`: correcto para aislar el efecto de escalares estocasticos sobre el salto local del lider.

### Riesgos criticos y mitigaciones

#### 1) Riesgo de division por cero en `leader_repulsion`

Cuando el enjambre colapsa sobre el lider, `g_best.position - centroid` puede ser vector cero.

Mitigacion recomendada:

```text
vec = g_best.position - centroid
norm = ||vec||
SI norm < 1e-12 ENTONCES
  vec = random_normal_vector(dim)
  norm = ||vec||
dir = vec / (norm + 1e-12)
g_best.position = clip_to_bounds(g_best.position + eta_repulsion * dir, bounds)
```

#### 2) Cuello de botella en `lime_wrapper`

Con `num_samples` alto (ej. 5000), la evaluacion fila por fila puede ser costosa.

Mitigacion recomendada:

1. Preasignar `deltas` y evitar alocaciones dentro del bucle.
2. Mantener bucle simple optimizado en Python (no asumir que `np.apply_along_axis` sera mas rapido).
3. Usar estrategia progresiva para `lime_num_samples`:
   - arranque: 1000-2000
   - analisis final: 3000-5000

#### 3) `decision_history` pequeno en disparos tempranos

Si el trigger se activa en pocas iteraciones, LIME puede dar pesos inestables.

Mitigacion recomendada:

```text
SI len(training_data) < min_rows_lime ENTONCES
  training_data = concat(training_data, synthetic_samples)

donde min_rows_lime >= 100 (ideal 150-300)
```

La muestra sintetica debe respetar distribuciones por variable:

1. `r1`: normal
2. `mag_browniano`: magnitud positiva (normal truncada o abs(normal))
3. `mag_levy`: positiva con cola pesada (Levy)
4. `r2`: uniforme [0,1]
5. `mag_predacion`: uniforme/escalada por `alpha_schedule`

### Recomendaciones extra para robustez

#### A) Filtro de calidad de explicacion

Antes de confirmar estancamiento, verificar fidelidad local del modelo de LIME.

```text
SI local_fidelity < fidelity_threshold ENTONCES
  status = "FALSE_ALARM"
```

#### B) Trigger de doble confirmacion temporal

Evitar activaciones espurias por ruido usando dos ventanas consecutivas.

```text
stagnation_flag_t = std(window_t) < epsilon
stagnation_flag_t_1 = std(window_t_1) < epsilon

SI stagnation_flag_t Y stagnation_flag_t_1 ENTONCES
  invocar LIME
```

### Ajuste sugerido a la regla de confirmacion

```text
confirmar_estancamiento =
  (abs(w_mag_browniano) > w_thr O abs(w_mag_predacion) > w_thr)
  Y (pred_delta < delta_tolerance)
  Y (local_fidelity >= fidelity_threshold)
```

Con este ajuste, el controlador reduce falsos positivos y mantiene decisiones de rescate mas confiables.
