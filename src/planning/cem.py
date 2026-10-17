"""Seeded integer CEM with receding-horizon execution and explicit diagnostics."""
from dataclasses import dataclass
import time
import numpy as np


@dataclass(frozen=True)
class Goal:
    holding: float = 1.0
    backlog: float = 1.0
    transport: float = 1.0
    uncertainty: float = 0.0

    def __post_init__(self):
        if not all(np.isfinite(v) and v >= 0 for v in vars(self).values()):
            raise ValueError('Goal weights must be finite and nonnegative')

    def score(self, outcomes):
        """[ensemble, candidates, horizon, 3] -> candidate score.

        Model dispersion is a heuristic penalty, not calibrated CVaR.
        """
        x = np.asarray(outcomes)
        if x.ndim != 4 or x.shape[-1] != 3:
            raise ValueError('Expected [ensemble,candidates,horizon,3]')
        total = (x * np.array([self.holding, self.backlog, self.transport])).sum((-1, -2))
        return total.mean(0) + self.uncertainty * total.std(0)


@dataclass
class PlanResult:
    actions: np.ndarray
    score: float
    predicted_components: list
    evaluations: int
    seed: int
    elapsed: float
    fallback: bool
    reason: str = ''


class CEMPlanner:
    def __init__(self, space, horizon=4, samples=64, elites=8, iterations=4, init_std=None):
        if not (horizon > 0 and iterations > 0 and 1 <= elites <= samples):
            raise ValueError('Invalid search configuration')
        self.space, self.horizon, self.init_std = space, horizon, init_std
        self.samples, self.elites, self.iterations = samples, elites, iterations

    def plan(self, observation, predict, goal=Goal(), seed=0, init=None, extra=()):
        """init: optional [horizon, size] warm-start mean; extra: candidates always evaluated first."""
        start = time.perf_counter()
        rng = np.random.default_rng(seed)
        shape = (self.horizon, self.space.size)
        mean = np.full(shape, self.space.maximum/2) if init is None else np.asarray(init, dtype=float)
        std = np.full(shape, self.space.maximum/2 if self.init_std is None else self.init_std)
        best, best_score, breakdown, evaluations = None, np.inf, [], 0
        try:
            for _ in range(self.iterations):
                candidates = self.space.project(rng.normal(mean, std, (self.samples, *shape)))
                candidates[0] = self.space.project(mean)
                candidates[1 if self.samples > 1 else 0] = 0
                for j, x in enumerate(extra):
                    candidates[2 + j] = self.space.project(x)
                if best is not None:
                    candidates[-1] = best
                outcomes = predict(observation, candidates)
                scores = goal.score(outcomes)
                evaluations += len(candidates)
                if scores.shape != (self.samples,):
                    raise ValueError('Prediction batch mismatch')
                valid = np.flatnonzero(np.isfinite(scores))
                if not len(valid):
                    raise FloatingPointError('All candidates nonfinite')
                elite = valid[np.argsort(scores[valid])[:self.elites]]
                index = elite[0]
                if scores[index] < best_score:
                    best, best_score = candidates[index].copy(), float(scores[index])
                    breakdown = np.asarray(outcomes)[:,index].mean(0).sum(0).tolist()
                mean = candidates[elite].mean(0)
                std = np.maximum(candidates[elite].std(0), 1.)
        except (ValueError, RuntimeError, FloatingPointError) as error:
            return PlanResult(np.zeros(shape, dtype=np.int64), float('inf'), [], evaluations,
                              seed, time.perf_counter()-start, True, str(error))
        for a in best:
            self.space.validate(a)
        return PlanResult(best, best_score, breakdown, evaluations, seed,
                          time.perf_counter()-start, False)


class RecedingHorizonController:
    def __init__(self, planner, predictor, goal=Goal(), seed=0):
        self.planner, self.predictor, self.goal, self.seed = planner, predictor, goal, seed

    def run(self, env, observation):
        records = []
        while env.day < env.days:
            result = self.planner.plan(observation, self.predictor, self.goal, self.seed+env.day)
            observation, cost, done, info = env.step(result.actions[0])
            records.append(dict(**info, predicted_score=result.score,
                                predicted_components=result.predicted_components,
                                evaluations=result.evaluations, fallback=result.fallback,
                                fallback_reason=result.reason, latency=result.elapsed))
            if done:
                break
        return records
