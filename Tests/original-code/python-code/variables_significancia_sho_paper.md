# Variables con mayor significancia en SHO y variables fijas

Fecha: 2026-05-18
Base: implementacion MATLAB original en [Tests/original-code/matlab-code/SHO.m](Tests/original-code/matlab-code/SHO.m) y traduccion Python en [Tests/original-code/python-code/SHO.py](Tests/original-code/python-code/SHO.py).

Criterio de clasificacion:
- **Impacto muy alto**: altera de forma directa la direccion, magnitud o seleccion de soluciones.
- **Impacto alto**: participa en la actualizacion pero como variable intermedia o de apoyo.
- **Impacto medio/bajo**: sirve para trazabilidad, evaluacion o logging.
- **Fijas**: constantes o umbrales que parametrizan la ecuacion pero no cambian durante la corrida.

## 1) Variables de mayor significancia e impacto

| Prioridad | Variable | Fase | Impacto en el algoritmo | Relacion con ecuacion/paper |
|---|---|---|---|---|
| Muy alta | `target_position` / `Elite` | Movimiento / predacion | Define el vector guia hacia el mejor individuo encontrado; si cambia, cambia toda la direccion de busqueda. | Referencia central en Eq.(4), Eq.(7), Eq.(10). |
| Muy alta | `step_length` | Movimiento | Controla la amplitud del desplazamiento Levy; gobierna exploracion global. | Componente principal de Eq.(4). |
| Muy alta | `beta` | Movimiento | Define la rama browniana de la actualizacion y puede dominar el salto cuando `r1(i) <= 0`; su efecto es directo sobre la direccion y magnitud del candidato. | Componente estocastica clave de Eq.(7). |
| Muy alta | `alpha` | Depredacion | Modula el balance exploracion-explotacion con el tiempo; afecta la agresividad de la actualizacion. | Factor temporal de Eq.(10). |
| Muy alta | `r1` | Movimiento | Selecciona la rama del motor behavior; determina si se usa Eq.(4) o Eq.(7). | Selector de bifurcacion en el bloque motor. |
| Muy alta | `r2` | Depredacion | Selecciona una de las dos variantes de Eq.(10); cambia el tipo de actualizacion. | Umbral estocastico de Eq.(10). |
| Muy alta | `r3` | Reproduccion | Controla la mezcla padre-madre; afecta la herencia de caracteristicas. | Coeficiente de Eq.(13). |
| Alta | `sea_horses_new1` | Movimiento | Es el primer estado candidato; si queda mal posicionado, arrastra la siguiente fase. | Salida de Eq.(4)/(7). |
| Alta | `sea_horses_new2` | Depredacion | Es el estado tras la depredacion; es el insumo directo de la reproduccion. | Salida de Eq.(10). |
| Alta | `sea_horses_fitness1` | Seleccion previa | Ordena los candidatos adultos antes de reproducir. | Base para Eq.(12). |
| Alta | `sea_horses_fitness2` | Reproduccion | Evalua la descendencia antes de la seleccion final. | Soporte de seleccion posterior a Eq.(13). |
| Alta | `sea_horses_father` | Reproduccion | Concentra la mejor mitad del ranking; define la calidad de los padres. | Parte superior de Eq.(12). |
| Alta | `sea_horses_mother` | Reproduccion | Aporta diversidad desde la otra mitad del ranking. | Parte complementaria de Eq.(12). |
| Alta | `si` / `sea_horses_offspring` | Reproduccion | Genera la descendencia; introduce nueva combinacion de soluciones. | Resultado directo de Eq.(13). |
| Alta | `Sea_horses_new` / `sea_horses_new` | Seleccion | Pool combinado de candidatos adultos + offspring. | Soporte de la seleccion elitista. |
| Alta | `sea_horses_fitness_all` | Seleccion | Decide que individuos sobreviven al truncar la poblacion. | Seleccion elitista posterior a Eq.(13). |
| Alta | `sorted_indexes` | Seleccion | Define el ranking final de supervivencia; impacta el reemplazo poblacional. | Ordenamiento de fitness para seleccion. |
| Alta | `SortfitbestN` / `sortfitbestn` | Convergencia | Guarda el subconjunto mejor rankeado para actualizar el elite global. | Soporte de la actualizacion del mejor global. |
| Alta | `target_fitness` | Convergencia | Conserva el mejor valor global; si mejora, la curva baja. | Best-so-far del algoritmo. |
| Alta | `convergence_curve` | Convergencia | Resume el rendimiento por iteracion. | Metrica final de convergencia. |

## 2) Variables de impacto intermedio

| Variable | Fase | Impacto | Comentario |
|---|---|---|---|
| `beta` | Movimiento | Muy alto | Introduce ruido browniano en la rama Eq.(7) y afecta directamente el candidato cuando se activa la segunda rama del motor behavior. |
| `theta`, `row`, `x`, `y`, `z` | Movimiento | Medio-alto | Construyen la geometria del salto helicoidal en Eq.(4). Son intermedias pero alteran la trayectoria. |
| `Sea_horses` / `sea_horses` | Estado global | Medio-alto | Es la poblacion completa; su valor cambia en cada seleccion. |
| `Sea_horsesFitness` / `sea_horses_fitness` | Estado global | Medio | Sirve para evaluar y ordenar, pero no define por si sola la dinamica. |
| `Sea_horsesFitness1` / `sea_horses_fitness1` | Estado intermedio | Alto | Evalua el estado post-depredacion antes de la reproduccion. |
| `Sea_horsesFitness2` / `sea_horses_fitness2` | Estado intermedio | Alto | Evalua la descendencia antes de la seleccion final. |
| `Sea_horses_new` / `sea_horses_new` | Estado intermedio | Alto | Contenedor de candidatos para el ranking final. |
| `SortfitbestN` / `sortfitbestn` | Estado intermedio | Medio-alto | Subconjunto seleccionado que alimenta la actualizacion del mejor global. |
| `population_history`, `fitness_history`, `Trajectories` | Trazabilidad | Bajo | No afectan la busqueda; solo registran resultados. |

## 3) Variables fijas o constantes dentro de las ecuaciones

Estas son las que conviene dejar aparte porque no cambian durante la corrida y solo parametrizan las ecuaciones:

| Variable / constante | Valor | Donde aparece | Rol |
|---|---|---|---|
| `u` | `0.05` | [Tests/original-code/matlab-code/SHO.m](Tests/original-code/matlab-code/SHO.m) y [Tests/original-code/python-code/SHO.py](Tests/original-code/python-code/SHO.py) | Escala base del componente helicoidal. |
| `v` | `0.05` | Idem | Ajusta la exponentiacion en la rama de movimiento. |
| `l` | `0.05` | Idem | Escala la rama browniana de Eq.(7). |
| `omega` | `1.5` | [Tests/original-code/matlab-code/SHO.m](Tests/original-code/matlab-code/SHO.m) y [Tests/original-code/python-code/SHO.py](Tests/original-code/python-code/SHO.py), ademas de [Tests/original-code/matlab-code/levy.m](Tests/original-code/matlab-code/levy.m) | Parametro fijo del vuelo de Levy. |
| `0.1` | umbral fijo | Eq.(10) | Decide cual rama de depredacion se aplica. |
| `pop/2` | particion fija por mitad | Eq.(12) | Divide padres y madres por ranking. |
| `1` en `rand()` / `randn()` | generacion aleatoria | todo el algoritmo | Factor estocastico fijo como tipo de muestreo, no parametro de control. |

## 4) Lectura rapida para tesis

Si quieres destacar solo lo que realmente mueve el algoritmo, prioriza este orden:

1. `target_position` / `Elite`
2. `step_length`
3. `beta`
4. `alpha`
5. `r1`
6. `r2`
7. `r3`
8. `sea_horses_new1` y `sea_horses_new2`
9. `sea_horses_fitness1`, `sea_horses_fitness2`
10. `sea_horses_father`, `sea_horses_mother`, `si`
11. `Sea_horses_new` / `sea_horses_new`, `sea_horses_fitness_all`, `sorted_indexes`, `SortfitbestN`, `target_fitness`, `convergence_curve`

## 5) Validacion de cobertura contra el paper y el codigo

Variables del flujo algorítmico que quedan cubiertas por esta tabla:

- Movimiento: `Elite` / `target_position`, `step_length`, `beta`, `r1`, `theta`, `row`, `x`, `y`, `z`.
- Depredacion: `alpha`, `r2`.
- Reproduccion: `sea_horses_father`, `sea_horses_mother`, `r3`, `si`.
- Seleccion: `Sea_horses_new` / `sea_horses_new`, `sea_horses_fitness_all`, `sorted_indexes`, `SortfitbestN`.
- Convergencia: `target_fitness`, `convergence_curve`.

Variables de implementacion que no aparecen como variables propias del paper, pero si del codigo:

- `sea_horses_fitness1`, `sea_horses_fitness2`
- `Sea_horsesFitness` / `sea_horses_fitness`
- `population_history`, `fitness_history`, `Trajectories`

Conclusion de validacion: no falta ninguna variable clave del flujo; lo que aparece adicional son variables de soporte, ranking o trazabilidad del codigo.

## 6) Conclusion

- Las variables con mayor significancia son las que alteran directamente direccion, amplitud o seleccion de soluciones: `Elite`, `step_length`, `beta`, `alpha`, `r1`, `r2`, `r3`.
- Las variables intermedias (`beta`, `theta`, `row`, `x`, `y`, `z`) son importantes, pero su efecto es instrumental y depende de las variables de control.
- Las fijas (`u`, `v`, `l`, `omega`, `0.1`, `pop/2`) conviene reportarlas aparte como hiperparametros o constantes de la ecuacion, no como variables dinamicas.