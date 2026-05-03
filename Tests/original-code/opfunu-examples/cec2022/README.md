# Ejemplos Opfunu CEC 2022

Estos ejemplos generan graficos para funciones benchmark CEC 2022 de Opfunu.

Como tu version instalada de Opfunu no expone los metodos plot_2d y plot_3d,
los scripts dibujan manualmente con matplotlib evaluando la funcion en una malla.

## Requisitos

Instala en tu entorno:

- opfunu
- matplotlib
- numpy
- setuptools==75.8.0

Nota: Opfunu usa pkg_resources en su version actual; por eso se fija setuptools en 75.8.0.

## 1) Graficar una sola funcion

```bash
python opfunu-examples/cec2022/plot_single_cec2022.py --function F12022 --points 120
```

Salida por defecto:

- opfunu-examples/cec2022/outputs/single/F12022_2d.png
- opfunu-examples/cec2022/outputs/single/F12022_3d.png

## 2) Graficar todas las funciones CEC 2022

```bash
python opfunu-examples/cec2022/plot_all_cec2022.py --points 120
```

Salida por defecto:

- Una carpeta por funcion dentro de opfunu-examples/cec2022/outputs/all
- Un resumen CSV: opfunu-examples/cec2022/outputs/all/cec2022_summary.csv

## Clases cubiertas (CEC 2022)

F12022, F22022, F32022, F42022, F52022, F62022, F72022, F82022, F92022, F102022, F112022, F122022.
