"""Parallel batch fobj evaluation via ProcessPoolExecutor.

Workers each initialize their own MLPAPObjective and pre-warm the Numba JIT cache.
Only the solution vector (n floats) is serialized per call — no large array transfer.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

_worker_obj = None


def _worker_init(instance_path: str, penalty_scale: float) -> None:
    global _worker_obj
    # Prevent each worker's numpy/OpenBLAS from spawning its own thread pool.
    # Without this, 16 workers × N BLAS threads = severe CPU over-subscription.
    for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[_var] = "1"
    from mlpap_problem import MLPAPObjective
    _worker_obj = MLPAPObjective(instance_path, penalty_scale)
    # Pre-warm Numba JIT so compilation does not happen during a real evaluation
    _dummy = np.random.rand(_worker_obj.dimension)
    for _ in range(3):
        _worker_obj(_dummy)
    _worker_obj.nfev = 0


def _worker_eval(vector_list: list) -> float:
    return _worker_obj(np.asarray(vector_list, dtype=float))


class ParallelFobj:
    """Context manager for parallel batch fobj evaluation.

    Usage:
        with ParallelFobj(str(instance_path), obj.penalty_scale) as batch:
            fitnesses = batch(population_matrix)   # [n_agents, dim] -> [n_agents]
            fobj.nfev += population_matrix.shape[0]  # keep nfev accurate manually
    """

    def __init__(
        self,
        instance_path: str,
        penalty_scale: float,
        n_workers: int | None = None,
    ) -> None:
        self.instance_path = str(instance_path)
        self.penalty_scale = float(penalty_scale)
        # M2 Ultra: 16 Performance + 8 Efficiency cores.
        # 16 workers saturates the P-cores; E-cores stay free for OS and orchestrator.
        self.n_workers = n_workers or max(1, min(16, (os.cpu_count() or 4) - 1))
        self._executor: ProcessPoolExecutor | None = None

    def start(self) -> None:
        self._executor = ProcessPoolExecutor(
            max_workers=self.n_workers,
            initializer=_worker_init,
            initargs=(self.instance_path, self.penalty_scale),
        )

    def stop(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None

    def __call__(self, population: np.ndarray) -> np.ndarray:
        """Evaluate all rows of the population matrix in parallel."""
        rows = [row.tolist() for row in population]
        return np.array(list(self._executor.map(_worker_eval, rows)), dtype=float)

    def __enter__(self) -> "ParallelFobj":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()
