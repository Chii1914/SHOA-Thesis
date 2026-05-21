# Clasificacion de ecuaciones del paper segun fases de SHO

Fecha: 2026-05-18
Base de trazabilidad: implementacion MATLAB en Tests/original-code/matlab-code/SHO.m
Paper: sea-horse.pdf (lectura textual completa no automatizada en este entorno; clasificacion hecha con Eq.(...) citadas en el codigo)

## Tabla de clasificacion

| Fase | Ecuacion(es) paper | Bloque en MATLAB | Variables clave | Interpretacion operacional |
|---|---|---|---|---|
| Movimiento (motor behavior) | Eq.(4), Eq.(7) | SHO.m:44-60 | Sea_horses, Sea_horses_new1, Step_length, Elite, r1, beta, u, v, l, theta, row, x, y, z | Genera el primer estado candidato por agente/dimension. Si r1(i)>0 usa rama helicoidal-escalada por Levy (Eq.4). Si no, usa perturbacion browniana-escalada (Eq.7). |
| Depredacion (predation behavior) | Eq.(10) (dos formas) | SHO.m:69-80 | Sea_horses_new1, Sea_horses_new2, Elite, alpha, r2 | Segunda actualizacion del estado candidato. r2(i) define una de dos variantes de Eq.(10). alpha modula balance exploracion/explotacion en funcion de t/Max_iter. |
| Reproduccion | Eq.(12), Eq.(13) | SHO.m:92-100 | Sea_horses_father, Sea_horses_mother, index, r3, Si, Sea_horses_offspring | Eq.(12): separa padres/madres por ranking de fitness. Eq.(13): mezcla lineal padre-madre para formar offspring. |
| Seleccion | (sin numero Eq explicito en el codigo) | SHO.m:109-116 | Sea_horsesFitness, Sea_horses_new, sorted_indexes, Sea_horses | Seleccion elitista por ordenamiento sobre adultos+descendencia; conserva mejores pop individuos para la siguiente iteracion. |
| Convergencia y actualizacion del mejor | (sin numero Eq explicito en el codigo) | SHO.m:123-128 y SHO.m:34-36 | TargetPosition, TargetFitness, SortfitbestN, Convergence_curve | Actualiza mejor global si mejora el fitness y registra la curva de convergencia por iteracion. |

## Evidencia puntual (lineas clave)

- Eq.(4): SHO.m:56
- Eq.(7): SHO.m:58
- Eq.(10): SHO.m:76 y SHO.m:78
- Eq.(12): SHO.m:93 y SHO.m:94
- Eq.(13): SHO.m:97
- Seleccion elitista: SHO.m:110-116
- Actualizacion mejor global y convergencia: SHO.m:123-128

## Nota metodologica para tesis

1. Las ecuaciones numeradas del paper presentes en el codigo se cubren en movimiento, depredacion y reproduccion.
2. Seleccion y convergencia estan implementadas explicitamente en codigo, pero sin etiqueta Eq.(...) en comentarios.
3. Esta clasificacion es consistente con la version MATLAB original y con su traduccion Python (SHO.py).