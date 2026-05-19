# Especificacion formal: SHOA+LIME adaptativo (sin perfiles)

Version: 1.0
Fecha: 2026-05-18
Estado: Propuesta de implementacion

## 1. Objetivo

Definir un controlador SHOA+LIME adaptativo que:

1. Elimine perfiles fijos (soft, medium, hard).
2. Use thresholds relativos al problema (dimension, escala de fitness, presupuesto iterativo).
3. Valide estadisticamente la evidencia usada para disparar rescate.
4. Mantenga explicabilidad local con 7 features en LIME.

## 2. Alcance

Incluye:

1. Trigger de estancamiento adaptativo.
2. Diagnostico LIME con 7 features.
3. Umbrales estadisticos para importancia y mejora esperada.
4. Magnitud de rescate adaptativa.
5. Esquema de logging para auditoria.

No incluye:

1. Cambios en formulacion base de SHO (Eq.4, Eq.7, Eq.10, Eq.12, Eq.13).
2. Explicacion global cross-run (se considera extension futura).

## 3. Features LIME obligatorias

Vector explicable local:

1. r1
2. mag_browniano
3. mag_levy
4. r2
5. mag_predacion
6. alpha
7. theta

Interpretacion:

1. r1, r2: selectores de rama.
2. mag_*: magnitud (intensidad) de componentes estocasticos.
3. alpha, theta: parametros estructurales de Eq.10 y Eq.4.

## 3.1 Diccionario formal de variables y simbolos

Para evitar ambiguedades, se usa la siguiente notacion:

| Simbolo / variable | Significado | Tipo | Origen |
|---|---|---|---|
| D | dimension del problema | entero | problema |
| N | tamano de poblacion (pop_size) | entero | configuracion |
| T | maximo de iteraciones (max_iter) | entero | configuracion |
| LB, UB | cotas inferior/superior por dimension | vector | problema |
| S_x | escala espacial promedio del dominio, mean(UB-LB) | real >= 0 | derivada |
| f_best(t) | mejor fitness observado en la iteracion t | real | dinamica online |
| w_win | longitud de la ventana deslizante para estancamiento | entero | adaptativo |
| W_t | ventana de fitness reciente, {f_best(t-w_win+1),...,f_best(t)} | conjunto/serie | dinamica online |
| S_f(t) | escala robusta local de fitness, median(abs(W_t))+epsilon_0 | real > 0 | derivada |
| epsilon_0 | constante pequena de estabilidad numerica | real > 0 | configuracion |
| z_stag(t) | indicador relativo de estancamiento, MAD(W_t)/S_f(t) | real >= 0 | derivada |
| tau_stag(t) | umbral adaptativo de estancamiento | real >= 0 | estadistico |
| Q_p(.) | operador cuantil de nivel p | operador | estadistico |
| p_stag | cuantil usado para tau_stag | real en (0,1) | configuracion |
| h | horizonte historico para estimar cuantiles | entero | configuracion |
| w_i(t) | peso local de LIME de la feature i en t | real | explicabilidad |
| CI95(w_i)=[l_i,u_i] | intervalo de confianza del peso local | intervalo | explicabilidad |
| sig_i | indicador de significancia, 1 si 0 no pertenece a CI95(w_i) | binaria | explicabilidad |
| tau_w,i(t) | umbral adaptativo de magnitud para la feature i | real >= 0 | estadistico |
| p_w | cuantil para tau_w,i | real en (0,1) | configuracion |
| Delta_hat(t) | mejora esperada local predicha por LIME | real | explicabilidad |
| Delta_rel(t) | mejora esperada relativa, Delta_hat/S_f(t) | real | derivada |
| tau_Delta(t) | umbral adaptativo para mejora esperada relativa | real | estadistico |
| m | minimo de features significativas requeridas | entero | configuracion |
| I_t | conjunto de features importantes y significativas en t | conjunto | explicabilidad |
| S_sev(t) | severidad normalizada de estancamiento | real en [0,1] | derivada |
| E(t) | evidencia explicable agregada en t | real >= 0 | derivada |
| G(t) | ganancia global adaptativa del controlador | real en [G_min,G_max] | control |
| lambda_1, lambda_2 | pesos de mezcla para severidad y evidencia | reales >= 0 | configuracion |
| G_min, G_max | limites de ganancia adaptativa | reales | configuracion |
| eta_eff(t) | intensidad efectiva de rescate tipo repulsion | real >= 0 | control |
| levy_scale_eff(t) | intensidad efectiva de rescate tipo Levy | real >= 0 | control |

Nota de notacion:

1. w_win (tamano de ventana) es distinto de w_i (peso de LIME).
2. Delta_hat y Delta_rel se escriben con Delta para diferenciar mejora absoluta vs relativa.

## 3.2 Definiciones en orden de ecuaciones

Esta seccion define variables en el mismo orden en que aparecen en las ecuaciones del paper y del controlador adaptativo.

### Eq.(4): movimiento helicoidal con Levy

$$
x' = x + step\_length \cdot \left((Elite-x)\cdot helix + Elite\right)
$$

Definiciones:

1. x: posicion actual del agente.
2. x': posicion candidata tras movimiento Eq.(4).
3. Elite: mejor posicion global conocida.
4. step_length: amplitud del salto Levy (escalada por mag_levy).
5. theta: angulo helicoidal.
6. row: radio helicoidal, row = u*exp(theta*v).
7. helix: termino geometrico, (row*cos(theta))*(row*sin(theta))*(row*theta).
8. u, v: constantes de forma del movimiento.
9. r1: selector de rama; Eq.(4) aplica si r1>0.

### Eq.(7): movimiento browniano

$$
x' = x + rand\cdot l\cdot beta\cdot(x-beta\cdot Elite)
$$

Definiciones:

1. x: posicion actual del agente.
2. x': posicion candidata tras movimiento Eq.(7).
3. rand: escalar aleatorio uniforme en [0,1].
4. l: constante de escala browniana.
5. beta: ruido gaussiano por dimension.
6. Elite: mejor posicion global conocida.
7. r1: selector de rama; Eq.(7) aplica si r1<=0.
8. mag_browniano: multiplicador de intensidad del componente browniano en la implementacion XAI.

### Eq.(10): depredacion

Forma A (si r2>=0.1):

$$
x'' = alpha\cdot(Elite-rand\cdot x') + (1-alpha)\cdot Elite
$$

Forma B (si r2<0.1):

$$
x'' = (1-alpha)\cdot(x'-rand\cdot Elite) + alpha\cdot x'
$$

Definiciones:

1. x': salida de movimiento (Eq.4 o Eq.7).
2. x'': salida de depredacion.
3. r2: selector de variante de Eq.(10).
4. alpha: factor temporal exploracion/explotacion.
5. rand: escalar aleatorio uniforme.
6. mag_predacion: multiplicador de intensidad del escalado de depredacion en la implementacion XAI.

### Eq.(12): seleccion de padres

Definiciones:

1. index: ranking por fitness de la poblacion post-depredacion.
2. father: mitad superior del ranking.
3. mother: mitad complementaria del ranking.

### Eq.(13): reproduccion

$$
offspring = r3\cdot father + (1-r3)\cdot mother
$$

Definiciones:

1. r3: coeficiente aleatorio de mezcla.
2. offspring: descendencia candidata.

Nota:

1. r3 no se incluye en el vector local de LIME si el wrapper explica solo salto local (movimiento+depredacion).

### Ecuaciones adaptativas del controlador (online)

1. Ventana adaptativa:

$$
w_{win}=\mathrm{clip}(\lfloor a_0+a_1\log_2(D+1)+a_2\sqrt{N}+a_3\sqrt{T/100}\rfloor,w_{min},w_{max})
$$

2. Escala robusta de fitness:

$$
S_f(t)=\mathrm{median}(|W_t|)+\epsilon_0
$$

3. Indicador de estancamiento:

$$
z_{stag}(t)=\frac{\mathrm{MAD}(W_t)}{S_f(t)}
$$

4. Umbral adaptativo de estancamiento:

$$
τ_{stag}(t)=Q_{p_{stag}}(\{z_{stag}(k)\}_{k=t-h}^{t-1})
$$

5. Mejora relativa explicada:

$$
\Delta_{rel}(t)=\frac{\widehat{\Delta}(t)}{S_f(t)}
$$

6. Severidad y ganancia de rescate:

$$
S_{sev}(t)=\mathrm{clip}\left(\frac{\tau_{stag}(t)-z_{stag}(t)}{\tau_{stag}(t)+\epsilon_0},0,1\right)
$$

$$
G(t)=\mathrm{clip}(\lambda_1S_{sev}(t)+\lambda_2E(t),G_{min},G_{max})
$$

### Explicacion operativa de las ecuaciones adaptativas

1. w_win define cuantas iteraciones recientes se usan para decidir estancamiento.
2. S_f(t) normaliza por la escala local del fitness para evitar umbrales absolutos.
3. z_stag(t) mide cuanta variacion relativa queda en la ventana reciente.
4. tau_stag(t) aprende un umbral dinamico desde el historial reciente.
5. Si z_stag(t) es menor que tau_stag(t), se considera candidato de estancamiento.
6. Delta_rel(t) mide la mejora esperada relativa predicha por LIME.
7. S_sev(t) cuantifica que tan severo es el estancamiento en escala [0,1].
8. E(t) resume la evidencia explicable (pesos relevantes y significativos).
9. G(t) combina severidad y evidencia para regular la fuerza final del rescate.

### Glosario de simbolos, operadores y parametros

Definiciones solicitadas:

1. clip(x, a, b): recorta x al intervalo [a,b].
   1. Si x < a, devuelve a.
   2. Si x > b, devuelve b.
   3. Si a <= x <= b, devuelve x.

2. floor(x): parte entera inferior de x.

3. log2(x): logaritmo base 2 de x.

4. sqrt(x): raiz cuadrada de x.

5. MAD(W): mediana de desviaciones absolutas respecto de la mediana de W.

6. median(W): valor central robusto de la muestra W.

7. Q_p(A): cuantil de nivel p del conjunto A.

8. a0, a1, a2, a3: coeficientes de calibracion de la ventana adaptativa.
   1. a0 controla el tamano base de ventana.
   2. a1 controla sensibilidad a dimension D.
   3. a2 controla sensibilidad a poblacion N.
   4. a3 controla sensibilidad a presupuesto T.

9. D: dimension del problema de optimizacion.

10. N: tamano de poblacion (pop_size).

11. T: numero maximo de iteraciones (max_iter).

12. w_min, w_max: cotas inferior y superior permitidas para la ventana adaptativa.

13. h: horizonte historico usado para estimar cuantiles adaptativos.

14. p_stag: nivel de cuantil para tau_stag(t).

15. p_w: nivel de cuantil para tau_w,i(t).

16. epsilon_0: constante pequena para estabilidad numerica en divisiones.

17. lambda_1, lambda_2: pesos que combinan severidad S_sev(t) y evidencia E(t) dentro de G(t).

18. G_min, G_max: limites de la ganancia adaptativa G(t).

## 4. Variables de escala del problema

Dadas por corrida:

1. D = dimension.
2. N = pop_size.
3. T = max_iter.
4. LB, UB = limites del dominio.

Derivadas:

$$
S_x = \mathrm{mean}(UB-LB)
$$

$$
W_t = \{f_{best}(t-w_{win}+1),...,f_{best}(t)\}
$$

$$
S_f(t) = \mathrm{median}(|W_t|)+\epsilon_0
$$

Donde $S_f(t)$ es la escala robusta local del fitness.

## 5. Trigger de estancamiento adaptativo

### 5.1 Ventana adaptativa

En vez de fija, usar:

$$
w_{win} = \mathrm{clip}(\lfloor a_0 + a_1\log_2(D+1) + a_2\sqrt{N} + a_3\sqrt{T/100}\rfloor, w_{min}, w_{max})
$$

### 5.2 Indicador de estancamiento relativo

$$
z_{stag}(t)=\frac{\mathrm{MAD}(W_t)}{S_f(t)}
$$

### 5.3 Threshold adaptativo

Definir $\tau_{stag}(t)$ como cuantil de historial reciente de $z_{stag}$:

$$
\tau_{stag}(t)=Q_{p_{stag}}\left(\{z_{stag}(k)\}_{k=t-h}^{t-1}\right)
$$

Trigger candidato si:

$$
z_{stag}(t) < \tau_{stag}(t)
$$

y no hay cooldown activo.

## 6. Diagnostico LIME con validez estadistica

### 6.1 Salidas requeridas por feature

Para cada feature i:

1. peso local $w_i(t)$.
2. intervalo de confianza $CI_{95}(w_i)=[l_i,u_i]$.
3. flag de significancia: $sig_i = 1$ si $0 \notin CI_{95}(w_i)$.

### 6.2 Importancia adaptativa

En vez de importance_threshold fijo:

$$
\tau_{w,i}(t)=Q_{p_w}\left(\{|w_i(k)|\}_{k=t-h}^{t-1}\right)
$$

feature i es importante si:

$$
|w_i(t)| > \tau_{w,i}(t) \;\wedge\; sig_i=1
$$

### 6.3 Mejora esperada relativa

$$
\Delta_{rel}(t)=\frac{\widehat{\Delta}(t)}{S_f(t)}
$$

No hay mejora suficiente si:

$$
\Delta_{rel}(t) < \tau_{\Delta}(t)
$$

con $\tau_{\Delta}(t)$ definido por cuantil historico o baseline por dimension.

### 6.4 Criterio POSITIVE_STAGNATION

Disparar diagnostico positivo si se cumplen simultaneamente:

1. trigger candidato de estancamiento.
2. mejora esperada relativa baja.
3. fidelidad local suficiente.
4. al menos m features importantes con significancia estadistica.

## 7. Fuerza adaptativa del rescate

Definir severidad de estancamiento:

$$
S_{sev}(t)=\mathrm{clip}\left(\frac{\tau_{stag}(t)-z_{stag}(t)}{\tau_{stag}(t)+\epsilon_0},0,1\right)
$$

Definir evidencia explicable:

$$
E(t)=\frac{1}{m}\sum_{i \in I_t}|w_i(t)|
$$

con $I_t$ = features importantes y significativas.

Magnitud global del controlador:

$$
G(t)=\mathrm{clip}(\lambda_1 S_{sev}(t)+\lambda_2 E(t),G_{min},G_{max})
$$

donde:

1. S_sev(t) mide cuan por debajo del umbral de estancamiento esta la dinamica.
2. E(t) resume cuanta evidencia explicable significativa hay.
3. G(t) controla la fuerza final del rescate en linea.

Aplicar:

$$
\eta_{eff}(t)=\eta_{base}\cdot G(t)
$$

$$
levy\_scale_{eff}(t)=levy\_scale_{base}\cdot G(t)
$$

## 8. Politica sin perfiles

Eliminar perfiles predefinidos.

Reemplazo:

1. Un solo modo adaptativo por corrida.
2. Parametros iniciales base + ajuste online via umbrales relativos.
3. Sin matriz soft/medium/hard.

## 9. Seleccion adaptativa del modo de rescate (opcional)

Mantener operadores base (levy_teleport y leader_repulsion), pero elegir dinamicamente:

$$
score_{levy}=|w_{mag\_levy}|+0.5|w_{theta}|
$$

$$
score_{rep}=|w_{mag\_browniano}|+|w_{mag\_predacion}|
$$

Si $score_{levy} \ge score_{rep}$ usar levy_teleport, si no leader_repulsion.

## 10. Cambios de logging requeridos

Agregar a salidas por iteracion:

1. window_size_eff
2. z_stag
3. tau_stag
4. delta_rel
5. tau_delta
6. G
7. eta_eff
8. levy_scale_eff

Agregar a contribuciones LIME:

1. weight_alpha, abs_weight_alpha
2. weight_theta, abs_weight_theta
3. ci_low_feature, ci_high_feature por las 7 features
4. significant_feature por las 7 features

## 11. Criterios de aceptacion

1. No usar thresholds absolutos fijos para estancamiento y delta.
2. Trigger estable en dimensiones diferentes (ej: D=10, 30, 100) sin retuning manual extremo.
3. Disminucion de falsos positivos de rescate frente a baseline fijo.
4. Registro completo para auditoria de decisiones online.
5. Compatibilidad con pipeline actual de benchmark (CSV y resumenes).

## 12. Riesgos y mitigaciones

1. Riesgo: sobreajuste de thresholds adaptativos a ruido local.
   Mitigacion: cuantiles robustos + ventana historica minima.
2. Riesgo: costo adicional por bootstrap de CI en LIME.
   Mitigacion: activar CI solo cuando trigger candidato sea verdadero.
3. Riesgo: inestabilidad temprana por poca historia.
   Mitigacion: warmup con defaults conservadores y transicion gradual.

## 13. Nota sobre r3

r3 pertenece a reproduccion (Eq.13) y no a la simulacion local de salto motor+depredacion.

Por eso, esta especificacion NO obliga r3 dentro del vector local de LIME salvo que se rediseñe el wrapper para explicar iteracion completa (movimiento+depredacion+reproduccion+seleccion).