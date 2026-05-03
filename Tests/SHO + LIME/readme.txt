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

run_cec2022_benchmark.py
Benchmark runner for CEC 2022 (F1..F12) that executes multiple runs per
function/dimension and exports raw runs, full iteration traceability
(full_output.csv, now including per-iteration LIME columns),
LIME contributions (lime_contributions.csv),
summary tables, rankings, and config.
Uses aggressive SHO+LIME profile by default (override with --profile paper).

run_cec2022_benchmark_shoa_puro.py
Benchmark runner for CEC 2022 (F1..F12) with SHOA puro (sin LIME), keeping
the same output structure for side-by-side comparison.

plot_lime_contributions.py
Script to generate and save LIME contribution plots from
lime_contributions.csv, with filters by iteration/function/run/seed.

benchmark_sholime_gui.py
Interactive GUI player for benchmark outputs. Shows convergence +
LIME contribution bars and supports playback/navigation across the full
execution (back/forward, jump +/-10, play/pause, case switch).

Dependencies:
- numpy
- matplotlib
- pandas
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
- CEC2022 benchmark paper-like defaults (30 runs, pop=30, max_iter=500):
       python run_cec2022_benchmark.py --functions all --dims 10 --runs 30
- CEC2022 benchmark with paper profile (less aggressive trigger/rescue):
       python run_cec2022_benchmark.py --functions all --dims 10 --runs 30 --profile paper
- CEC2022 benchmark SHOA puro (same protocol, no LIME):
       python run_cec2022_benchmark_shoa_puro.py --functions all --dims 10 --runs 30
- Quick smoke test over two CEC2022 functions:
       python run_cec2022_benchmark.py --functions F12022,F22022 --dims 10 --runs 2 --max-iter 40 --lime-samples 400
- Plot and save one LIME explanation (single match):
       python plot_lime_contributions.py --csv-path benchmark_logs/<run_folder>/lime_contributions.csv --iteration 492 --function F12022 --run-id 6 --seed 6
- Plot and save all matches for an iteration:
       python plot_lime_contributions.py --csv-path benchmark_logs/<run_folder>/lime_contributions.csv --iteration 492 --all-matches
- Open interactive playback GUI over a benchmark folder:
       python benchmark_sholime_gui.py --run-dir benchmark_logs/<run_folder>
