# SHOA-COMBINED

Esta carpeta ahora se organiza por problema objetivo:

- cec2022/
- tmlap/

## cec2022

Implementacion combinada completa (SHOA + detector de estancamiento + LIME + reinicio guiado).

Entrar a cec2022/ y ejecutar:

python run_cec2022_combined.py --functions 1 --dim 10 --pop 20 --max-iter 50 --runs 1

## tmlap

Implementacion minima combinada para instancias TMLAP con el mismo controlador hibrido.

Entrar a tmlap/ y ejecutar:

python run_tmlap_combined.py --instances 1.instancia_simple.txt --pop 20 --max-iter 50 --runs 1

## Nota

En esta etapa se priorizo dejar ambas rutas ejecutables. La variante cec2022 conserva toda la funcionalidad avanzada; la variante tmlap es una base funcional minima para estandarizar estructura y contratos de salida.
