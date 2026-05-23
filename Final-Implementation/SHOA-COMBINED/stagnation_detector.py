"""Paper-style stagnation detector based on MinSFEs + MaxFEs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StagnationSnapshot:
    fe: int
    best_fitness: float
    last_improvement_fe: int
    sfes: int
    min_sfes: int
    stagnated: bool
    improved: bool
    event: str
    iteration: int


class StagnationDetector:
    """Tracks SFEs and detects stagnation when SFEs >= MinSFEs."""

    def __init__(self, max_fes: int, min_sfes_ratio: float = 0.04) -> None:
        if max_fes <= 0:
            raise ValueError("max_fes must be > 0")
        if min_sfes_ratio <= 0:
            raise ValueError("min_sfes_ratio must be > 0")

        self.max_fes = int(max_fes)
        self.min_sfes_ratio = float(min_sfes_ratio)
        self.min_sfes = max(1, int(round(self.max_fes * self.min_sfes_ratio)))

        self.best_fitness = float("inf")
        self.last_improvement_fe = 0
        self.stagnated = False
        self.initialized = False

    def initialize(self, best_fitness: float, fe: int, iteration: int = 0) -> StagnationSnapshot:
        self.best_fitness = float(best_fitness)
        self.last_improvement_fe = int(fe)
        self.stagnated = False
        self.initialized = True

        return StagnationSnapshot(
            fe=int(fe),
            best_fitness=float(self.best_fitness),
            last_improvement_fe=int(self.last_improvement_fe),
            sfes=0,
            min_sfes=int(self.min_sfes),
            stagnated=False,
            improved=True,
            event="initialized",
            iteration=int(iteration),
        )

    def update(self, best_fitness: float, fe: int, iteration: int) -> StagnationSnapshot:
        if not self.initialized:
            return self.initialize(best_fitness=best_fitness, fe=fe, iteration=iteration)

        fe_int = int(fe)
        improved = float(best_fitness) < float(self.best_fitness)

        if improved:
            self.best_fitness = float(best_fitness)
            self.last_improvement_fe = fe_int

        sfes = fe_int - int(self.last_improvement_fe)
        was_stagnated = self.stagnated
        self.stagnated = bool(sfes >= self.min_sfes)

        event = "none"
        if (not was_stagnated) and self.stagnated:
            event = "stagnation_start"
        elif was_stagnated and (not self.stagnated):
            event = "recovered"

        return StagnationSnapshot(
            fe=fe_int,
            best_fitness=float(self.best_fitness),
            last_improvement_fe=int(self.last_improvement_fe),
            sfes=int(sfes),
            min_sfes=int(self.min_sfes),
            stagnated=bool(self.stagnated),
            improved=bool(improved),
            event=event,
            iteration=int(iteration),
        )
