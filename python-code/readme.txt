Sea-Horse optimizer (SHO) - Python translation

April 4, 2026

Original authors (MATLAB version):
Shijie Zhao and Tianran Zhang
email: zhaoshijie@lntu.edu.cn
       ztr20010118@126.com

This folder contains a direct Python translation of the MATLAB files in matlab-code.

Files:
BenchmarkFunctions.py
Benchmark functions used by SHO (F1 to F23).

SHO.py
Core implementation of the Sea-Horse Optimizer algorithm.

initialization.py
Population initialization helper.

levy.py
Levy-flight random coefficient generator.

main_SHO.py
Main runnable script that executes SHO on one benchmark and plots the convergence curve.

Dependencies:
- numpy
- matplotlib
