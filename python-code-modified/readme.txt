Sea-Horse optimizer (SHO) - Python translation

April 4, 2026

Original authors (MATLAB version):
Shijie Zhao and Tianran Zhang
email: zhaoshijie@lntu.edu.cn
       ztr20010118@126.com

This folder contains the modified Python implementation with SHO + LIME controller
and CEC function selection through Opfunu.

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
Main runnable script (style similar to original main_SHO.py) that executes SHO + LIME
on a selected CEC function by editing variable f_name (example: F12022).

SHO_LIME_Controller.py
SHO implementation with stagnation trigger, LIME-based diagnosis, and rescue protocols.

main_SHO_LIME.py
Alternative CLI runnable script for SHO + LIME over Opfunu CEC functions.

run_multiseed_sho_lime.py
Batch runner that executes SHO + LIME for multiple seeds (default 50 runs) and
exports per-iteration logs, per-run summaries, and aggregate statistics.

Dependencies:
- numpy
- matplotlib
- lime
- opfunu

Compatibility note:
If Opfunu fails with "No module named pkg_resources", install:
- setuptools==75.8.0

Examples:
- Main execution (edit f_name and dim_req inside main_SHO.py):
       python main_SHO.py
- CLI execution:
       python main_SHO_LIME.py --function F112022 --ndim 10 --max-iter 80 --lime-samples 800
- Multi-seed batch execution (default 50 runs):
       python run_multiseed_sho_lime.py --function F112022 --ndim 20 --runs 50 --seed-start 1 --max-iter 15000 --pop-size 60
