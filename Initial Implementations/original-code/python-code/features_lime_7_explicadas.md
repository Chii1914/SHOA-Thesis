# 7 features de LIME para SHOA+LIME

Fecha: 2026-05-18

Este documento resume las 7 features propuestas para el diagnostico local con LIME en SHOA+LIME, su relacion con las ecuaciones del paper y como interpretar sus pesos.

## Features seleccionadas

1. r1
2. mag_browniano
3. mag_levy
4. r2
5. mag_predacion
6. alpha
7. theta

## Que significa cada feature

| Feature | Fase | Relacion con ecuaciones | Rol en la dinamica |
|---|---|---|---|
| r1 | Movimiento | Eq.(4) vs Eq.(7) | Selector de rama del movimiento. Si cambia, cambia la forma del salto. |
| mag_browniano | Movimiento | Eq.(7) | Magnitud (intensidad) del componente browniano en la rama de perturbacion. |
| mag_levy | Movimiento | Eq.(4) | Magnitud (intensidad) del salto Levy en la rama helicoidal/exploratoria. |
| r2 | Depredacion | Eq.(10) | Selector de variante de depredacion por umbral (rama A vs rama B). |
| mag_predacion | Depredacion | Eq.(10) | Magnitud (intensidad) del escalado estocastico durante depredacion. |
| alpha | Depredacion | Eq.(10) | Factor temporal de balance exploracion-explotacion. |
| theta | Movimiento | Eq.(4) | Controla la geometria del movimiento helicoidal (angulo). |

## Que significa mag

La palabra mag significa magnitud.

En este contexto:
- mag_browniano = cuanto aporta el termino browniano al desplazamiento.
- mag_levy = cuanto aporta el salto Levy al desplazamiento.
- mag_predacion = cuanto se escala el paso de depredacion.

No representa direccion por si sola; representa intensidad del mecanismo estocastico.

## Como interpretar los pesos de LIME

Para cada diagnostico local, LIME entrega un peso por feature.

Regla practica:
- Peso positivo: esa feature empuja hacia mayor mejora esperada local.
- Peso negativo: esa feature empuja hacia menor mejora esperada local o estancamiento.
- Valor absoluto grande: feature con mayor influencia local en ese diagnostico.

Importante:
- Los pesos son locales (por iteracion/diagnostico), no globales para toda la corrida.
- Se recomienda analizar tendencia en multiples diagnosticos para conclusiones robustas.

## Por que estas 7 y no menos

Estas 7 cubren decisiones de alto impacto en movimiento y depredacion:
- Selectores de rama: r1, r2.
- Intensidades estocasticas: mag_browniano, mag_levy, mag_predacion.
- Parametros estructurales de ecuacion: alpha, theta.

Con esto, la explicacion local queda alineada con la estructura del paper en Eq.(4), Eq.(7) y Eq.(10).

## Nota sobre reproduccion (r3)

r3 es importante en Eq.(13), pero pertenece a la fase de reproduccion poblacional.

Si el wrapper de LIME explica solo salto local (movimiento + depredacion), incluir r3 sin redisenar la simulacion puede producir interpretaciones inestables.

Para incluir r3 correctamente, se recomienda explicar una iteracion completa (movimiento + depredacion + reproduccion + seleccion) o usar estadisticas agregadas de reproduccion.
