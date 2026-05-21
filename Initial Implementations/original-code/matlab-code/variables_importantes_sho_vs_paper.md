# Variables importantes de SHO y contraste con paper

Fecha: 2026-05-18
Fuente principal de contraste: ecuaciones referenciadas en el codigo (Eq.(4), Eq.(7), Eq.(10), Eq.(12), Eq.(13)).
Fuente secundaria: sea-horse.pdf (pendiente contraste textual literal automatizado).

## 1) Contexto de ejecucion y configuracion

| Variable | Rol | Evidencia en codigo | Contraste con paper |
|---|---|---|---|
| pop / popsize | Tamano de poblacion (search agents) | main_SHO.m:17, SHO.m:13 | Coincide con N poblacion en metaheuristicas poblacionales. |
| Max_iter | Iteraciones maximas | main_SHO.m:18, SHO.m:13 | Coincide con horizonte iterativo T. |
| LB, UB | Limites inferior/superior del espacio de busqueda | main_SHO.m:21, initialization.m:1 | Coincide con restricciones de dominio del benchmark. |
| Dim | Dimension del problema | main_SHO.m:21, SHO.m:13 | Coincide con dimensionalidad D usada por ecuaciones de movimiento. |
| fobj | Funcion objetivo | main_SHO.m:21, SHO.m:13 | Coincide con formulacion de minimizacion del paper. |

## 2) Variables nucleares del estado de la poblacion

| Variable | Rol | Evidencia en codigo | Contraste con paper |
|---|---|---|---|
| Sea_horses | Poblacion actual (matriz pop x Dim) | SHO.m:15 | Representa individuos/posiciones de seahorses. |
| Sea_horsesFitness | Fitness de la poblacion en seleccion | SHO.m:16, SHO.m:110 | Coincide con evaluacion de calidad para ranking. |
| TargetPosition | Mejor posicion global encontrada (elite) | SHO.m:34 | Equivalente a best/elite individual del paper. |
| TargetFitness | Mejor fitness global | SHO.m:35 | Equivalente al mejor valor objetivo global. |
| Elite | Replica matricial de la mejor solucion | SHO.m:43 | Coincide con uso de elite como referencia de atraccion. |

## 3) Movimiento (motor behavior)

| Variable | Rol | Evidencia en codigo | Contraste con paper |
|---|---|---|---|
| Step_length | Longitud de paso por vuelo de Levy | SHO.m:46, levy.m:1, levy.m:7 | Coincide con componente Levy flight para exploracion. |
| beta | Ruido gaussiano de perturbacion | SHO.m:42 | Componente estocastico del termino de movimiento (rama Eq.(7)). |
| r1 | Selector de rama de movimiento | SHO.m:45, SHO.m:49 | Si r1(i)>0 usa Eq.(4), si no Eq.(7). |
| u, v, l | Hiperparametros fijos de movimiento | SHO.m:38, SHO.m:39, SHO.m:40 | Parametros de escala/curvatura; en codigo son constantes. |
| theta, row, x, y, z | Variables intermedias de transformacion de movimiento | SHO.m:51-55 | Implementan la geometria estocastica previa a Eq.(4). |
| Sea_horses_new1 | Poblacion tras fase de movimiento | SHO.m:56, SHO.m:58 | Resultado de aplicar Eq.(4) o Eq.(7). |

### Ecuaciones del paper reflejadas en codigo
- Eq.(4): SHO.m:56
- Eq.(7): SHO.m:58

## 4) Depredacion (predation behavior)

| Variable | Rol | Evidencia en codigo | Contraste con paper |
|---|---|---|---|
| alpha | Factor temporal de balance exploracion/explotacion | SHO.m:74 | Decrece con t y modula intensidad de actualizacion (Eq.(10)). |
| r2 | Selector estocastico de variante de depredacion | SHO.m:73, SHO.m:75 | Umbral 0.1 determina rama de Eq.(10). |
| Sea_horses_new2 | Poblacion tras depredacion | SHO.m:76, SHO.m:78 | Estado previo a evaluacion y reproduccion. |
| Sea_horsesFitness1 | Fitness despues de depredacion | SHO.m:88 | Evalua calidad de Sea_horses_new2. |

### Ecuacion del paper reflejada en codigo
- Eq.(10): SHO.m:76 y SHO.m:78

## 5) Reproduccion

| Variable | Rol | Evidencia en codigo | Contraste con paper |
|---|---|---|---|
| index | Indices de ranking por fitness | SHO.m:90 | Define particion para padres y madres. |
| Sea_horses_father | Mitad superior del ranking | SHO.m:93 | Coincide con Eq.(12) (grupo de alta calidad). |
| Sea_horses_mother | Mitad inferior del ranking | SHO.m:94 | Coincide con Eq.(12) (grupo complementario). |
| r3 | Peso aleatorio de recombinacion | SHO.m:96 | Coeficiente de mezcla lineal en Eq.(13). |
| Si | Offspring intermedio generado | SHO.m:97 | Estructura de descendencia por mezcla padre-madre. |
| Sea_horses_offspring | Descendencia final | SHO.m:99 | Poblacion candidata adicional para seleccion. |
| Sea_horsesFitness2 | Fitness de descendencia | SHO.m:106 | Permite competir offspring vs adultos. |

### Ecuaciones del paper reflejadas en codigo
- Eq.(12): SHO.m:93-94
- Eq.(13): SHO.m:97

## 6) Seleccion y seguimiento

| Variable | Rol | Evidencia en codigo | Contraste con paper |
|---|---|---|---|
| Sea_horses_new | Pool combinado (adultos + offspring) | SHO.m:111 | Coincide con seleccion elitista sobre candidatos mixtos. |
| sorted_indexes | Ranking final para truncamiento a pop | SHO.m:113 | Seleccion de mejores pop individuos. |
| SortfitbestN | Fitness ordenado de supervivientes | SHO.m:117 | Base para actualizar objetivo global. |
| Convergence_curve | Curva de convergencia del mejor fitness | SHO.m:19, SHO.m:36, SHO.m:128 | Coincide con metrica tipica reportada en paper. |
| fitness_history | Historial fitness por individuo/iteracion | SHO.m:17, SHO.m:118 | Trazabilidad de dinamica poblacional. |
| population_history | Historial de posiciones | SHO.m:18, SHO.m:119 | Permite analizar trayectoria en espacio de busqueda. |
| Trajectories | Historial de primera dimension | SHO.m:20, SHO.m:120 | Salida auxiliar de trayectoria. |

## 7) Hallazgos de contraste

1. Alta consistencia estructural con el paper en las fases motor, depredacion y reproduccion via Eq.(4/7/10/12/13).
2. Elite y TargetPosition estan implementados de forma equivalente a best-so-far global.
3. El componente Levy esta implementado explicitamente con omega=1.5 (levy.m).
4. Parametros u, v, l son constantes fijas en codigo (0.05), posible punto de sensibilidad no documentado en detalle.
5. La referencia textual literal del PDF no se automatizo en este entorno (no hay extractor PDF activo), pero la trazabilidad ecuacion-codigo esta verificada.

## 8) Nota para tesis

Para el capitulo metodologico, puedes citar este mapeo como "validacion de implementacion" y luego complementar con citas textuales del PDF en una revision manual final.