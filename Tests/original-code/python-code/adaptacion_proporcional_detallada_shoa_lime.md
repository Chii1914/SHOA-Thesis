# Adaptacion proporcional detallada para SHOA+LIME

Fecha: 2026-05-18
Estado: Especificacion cuantitativa (lista para implementar)

## 1) Objetivo

Definir reglas numericas concretas para adaptar online el controlador SHOA+LIME al problema, usando proporciones de:

1. Dimension D.
2. Poblacion N.
3. Presupuesto iterativo T.
4. Escala espacial del dominio (LB, UB).
5. Escala local del fitness.

## 2) Variables base y normalizaciones

Variables base:

1. D: dimension del problema.
2. N: pop_size.
3. T: max_iter.
4. LB, UB: limites por dimension.
5. f_best(t): mejor fitness en la iteracion t.

Referencias (baseline):

1. D_ref = 30
2. N_ref = 30
3. T_ref = 500

Normalizaciones:

$$
u_D = \frac{\log_2(D+1)}{\log_2(D_{ref}+1)}$$

$$
u_N = \sqrt{\frac{N}{N_{ref}}}$$

$$
u_T = \sqrt{\frac{T}{T_{ref}}}$$

Escala espacial:

$$S_x = \mathrm{mean}(UB-LB)$$

## 3) Proporciones de adaptacion (pesos de mezcla)

Para parametros temporales del controlador, usar mezcla proporcional:

1. 45% dependencia de dimension (nu_D).
2. 30% dependencia de poblacion (nu_N).
3. 25% dependencia de presupuesto iterativo (nu_T).

Factor compuesto:

$$
\Phi = 0.45\nu_D + 0.30\nu_N + 0.25\nu_T
$$

## 4) Ventana, cooldown y patience adaptativos

### 4.1 Tamano de ventana adaptativa

$$
w_{win}=\mathrm{clip}(\mathrm{round}(10\cdot\Phi),8,60)
$$

### 4.2 Cooldown adaptativo

$$
cooldown_{eff}=\mathrm{clip}(\mathrm{round}(0.5\cdot w_{win}),3,30)
$$

### 4.3 Rescue patience adaptativo

$$
patience_{eff}=\mathrm{clip}(\mathrm{round}(2.0\cdot w_{win}),10,120)
$$

Interpretacion:

1. Problemas mas grandes => ventana mas larga.
2. Mayor ventana => mayor cooldown y mayor paciencia para evaluar rescate.

## 5) Trigger de estancamiento relativo

Ventana reciente:

$$
W_t = \{f_{best}(t-w_{win}+1),...,f_{best}(t)\}
$$

Escala robusta local:

$$
S_f(t)=\mathrm{median}(|W_t|)+\epsilon_0,\quad \epsilon_0=10^{-12}
$$

Indicador relativo:

$$
z_{stag}(t)=\frac{\mathrm{MAD}(W_t)}{S_f(t)}
$$

Umbral adaptativo por cuantil historico:

$$
\tau_{stag,base}(t)=Q_{0.25}(\{z_{stag}(k)\}_{k=t-h}^{t-1}),\quad h=\max(50,3w_{win})
$$

Ajuste proporcional por complejidad:

$$
\tau_{stag}(t)=\mathrm{clip}(\tau_{stag,base}(t)\cdot(1+0.20(\nu_D-1)-0.10(\nu_N-1)),10^{-6},0.20)
$$

Trigger candidato:

$$
z_{stag}(t) < \tau_{stag}(t)
$$

## 6) Delta tolerance relativo al problema

En vez de delta fijo absoluto, usar:

$$
\delta_{tol,abs}(t)=k_\Delta(t)\cdot S_f(t)
$$

con:

$$
k_\Delta(t)=\mathrm{clip}(0.0015\cdot S_x\cdot(1+0.15(\nu_D-1)),10^{-5},5\times10^{-3})
$$

Criterio de baja mejora esperada:

$$
\widehat{\Delta}(t) < \delta_{tol,abs}(t)
$$

## 7) Importancia estadistica de pesos LIME

Para cada feature i:

1. Umbral adaptativo de magnitud:

$$
\tau_{w,i}(t)=Q_{0.75}(\{|w_i(k)|\}_{k=t-h}^{t-1})
$$

2. Significancia por intervalo de confianza (bootstrap):

$$
sig_i(t)=1 \iff 0 \notin CI_{95}(w_i(t))
$$

3. Feature importante:

$$
important_i(t)=1 \iff |w_i(t)|>\tau_{w,i}(t)\ \wedge\ sig_i(t)=1
$$

## 8) Severidad, evidencia y ganancia adaptativa

Severidad normalizada:

$$
S_{sev}(t)=\mathrm{clip}\left(\frac{\tau_{stag}(t)-z_{stag}(t)}{\tau_{stag}(t)+\epsilon_0},0,1\right)
$$

Evidencia explicable normalizada:

$$
E(t)=\mathrm{clip}\left(\frac{1}{m}\sum_{i\in I_t}\frac{|w_i(t)|}{\tau_{w,i}(t)+\epsilon_0},0,2\right)
$$

con:

1. I_t: conjunto de features importantes.
2. m = max(1, |I_t|).

Ganancia global:

$$
G(t)=\mathrm{clip}(0.6\cdot S_{sev}(t)+0.4\cdot E(t)/2,0.2,1.8)
$$

Fuerza efectiva del rescate:

$$
\eta_{eff}(t)=\eta_{base}\cdot G(t)
$$

$$
levy\_scale_{eff}(t)=levy\_scale_{base}\cdot G(t)
$$

## 9) Umbral de mejora minima para rollback (relativo)

En vez de rescue_min_improvement fijo:

$$
\Delta_{min,rollback}(t)=k_R\cdot S_f(t)
$$

con:

$$
k_R=\mathrm{clip}(5\times10^{-4}\cdot S_x,10^{-7},10^{-2})
$$

Rollback si no se cumple mejora minima al terminar patience_eff.

## 10) Features usadas por LIME en este esquema

1. r1
2. mag_browniano
3. mag_levy
4. r2
5. mag_predacion
6. alpha
7. theta

Nota:

1. r3 se mantiene fuera del wrapper local de LIME mientras la explicacion sea salto local (movimiento+depredacion).

## 11) Ejemplos numericos rapidos

### Caso A: D=10, N=30, T=500

1. nu_D ~= 0.70
2. nu_N = 1.00
3. nu_T = 1.00
4. Phi ~= 0.864
5. w_win ~= 9
6. cooldown_eff ~= 5
7. patience_eff ~= 18

### Caso B: D=100, N=30, T=500

1. nu_D ~= 1.34
2. nu_N = 1.00
3. nu_T = 1.00
4. Phi ~= 1.155
5. w_win ~= 12
6. cooldown_eff ~= 6
7. patience_eff ~= 24

### Caso C: D=500, N=60, T=1000

1. nu_D ~= 1.81
2. nu_N ~= 1.41
3. nu_T ~= 1.41
4. Phi ~= 1.593
5. w_win ~= 16
6. cooldown_eff ~= 8
7. patience_eff ~= 32

## 12) Criterio practico de implementacion

Implementar por fases:

1. Fase 1: w_win, cooldown_eff, patience_eff adaptativos.
2. Fase 2: delta y trigger relativos (S_f, z_stag, tau_stag).
3. Fase 3: umbrales de pesos con CI bootstrap y ganancia G(t).

Con esto, la adaptacion queda proporcional al problema, estadisticamente trazable y estable entre escalas diferentes.