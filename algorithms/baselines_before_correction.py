"""
Vectorized GPU baseline metaheuristics for the MVDO project.

All classes inherit from algorithms.base.BaseOptimizer and follow the same API
as MVDO:
    opt = Algorithm(fn, dim, lb, ub, pop_size, max_iter, num_runs, device, dtype, seed)
    result = opt.run()

Implemented baselines:
    PSO  - Particle Swarm Optimization
    GWO  - Grey Wolf Optimizer
    WOA  - Whale Optimization Algorithm
    SCA  - Sine Cosine Algorithm
    HHO  - Harris Hawks Optimizer
    AOA  - Archimedes Optimization Algorithm
    EO   - Equilibrium Optimizer

The implementations are vectorized over R independent runs and N agents.
They are intended for fair benchmarking against MVDO under the same population
size, iteration budget, device, dtype, and random seed protocol.
"""

import math
import torch

from .base import BaseOptimizer


# -----------------------------------------------------------------------------
# Shared helpers
# -----------------------------------------------------------------------------

def _mantegna_levy(shape, beta, generator, device, dtype):
    """Levy flight samples using Mantegna's algorithm."""
    num = math.gamma(1.0 + beta) * math.sin(math.pi * beta / 2.0)
    den = math.gamma((1.0 + beta) / 2.0) * beta * (2.0 ** ((beta - 1.0) / 2.0))
    sigma_u = (num / den) ** (1.0 / beta)
    u = torch.randn(*shape, generator=generator, device=device, dtype=dtype) * sigma_u
    v = torch.randn(*shape, generator=generator, device=device, dtype=dtype)
    return u / (torch.abs(v) ** (1.0 / beta) + 1e-12)


def _gather_random_agents(population, indices):
    """Gather random agents per run.

    population: [R, N, D]
    indices:    [R, N]
    returns:    [R, N, D]
    """
    R, N, _ = population.shape
    run_idx = torch.arange(R, device=population.device).view(R, 1).expand(R, N)
    return population[run_idx, indices]


def _topk_positions(population, fitness, k):
    """Return top-k positions and fitness values per run."""
    sorted_fit, sorted_idx = torch.sort(fitness, dim=1)
    top_idx = sorted_idx[:, :k]
    R = population.shape[0]
    run_idx = torch.arange(R, device=population.device).view(R, 1).expand(R, k)
    return population[run_idx, top_idx].clone(), sorted_fit[:, :k].clone()


# -----------------------------------------------------------------------------
# PSO
# -----------------------------------------------------------------------------

class PSO(BaseOptimizer):
    """Particle Swarm Optimization.

    Default parameters follow a common linearly decaying inertia setup.
    """

    def __init__(self, fn, dim, lb, ub, pop_size=30, max_iter=1000, num_runs=30,
                 w_max=0.9, w_min=0.4, c1=2.0, c2=2.0,
                 device='cuda:0', dtype=torch.float64, seed=None):
        super().__init__(fn, dim, lb, ub, pop_size, max_iter, num_runs, device, dtype, seed)
        self.w_max = w_max
        self.w_min = w_min
        self.c1 = c1
        self.c2 = c2
        self.velocity = None
        self.pbest_position = None
        self.pbest_fitness = None

    def _lazy_init(self):
        if self.velocity is None:
            span = (self.ub - self.lb).view(1, 1, self.dim)
            self.velocity = 0.1 * span * self._randn(self.num_runs, self.pop_size, self.dim)
            self.pbest_position = self.population.clone()
            self.pbest_fitness = self.fitness.clone()

    def step(self, t):
        self._lazy_init()
        denom = max(1, self.max_iter - 1)
        w = self.w_max - (self.w_max - self.w_min) * (t / denom)
        r1 = self._rand(self.num_runs, self.pop_size, self.dim)
        r2 = self._rand(self.num_runs, self.pop_size, self.dim)
        gbest = self.best_position.unsqueeze(1)
        self.velocity = (w * self.velocity
                         + self.c1 * r1 * (self.pbest_position - self.population)
                         + self.c2 * r2 * (gbest - self.population))
        self.population = self._clip_to_bounds(self.population + self.velocity)
        self.fitness = self.fn(self.population)

        improved = self.fitness < self.pbest_fitness
        self.pbest_fitness = torch.where(improved, self.fitness, self.pbest_fitness)
        self.pbest_position = torch.where(improved.unsqueeze(-1), self.population, self.pbest_position)
        self._update_best()


# -----------------------------------------------------------------------------
# GWO
# -----------------------------------------------------------------------------

class GWO(BaseOptimizer):
    """Grey Wolf Optimizer."""

    def step(self, t):
        denom = max(1, self.max_iter - 1)
        a = 2.0 - 2.0 * (t / denom)
        leaders, _ = _topk_positions(self.population, self.fitness, 3)
        alpha = leaders[:, 0, :].unsqueeze(1)
        beta = leaders[:, 1, :].unsqueeze(1)
        delta = leaders[:, 2, :].unsqueeze(1)

        def move_toward(leader):
            A = 2.0 * a * self._rand(self.num_runs, self.pop_size, self.dim) - a
            C = 2.0 * self._rand(self.num_runs, self.pop_size, self.dim)
            D = torch.abs(C * leader - self.population)
            return leader - A * D

        X1 = move_toward(alpha)
        X2 = move_toward(beta)
        X3 = move_toward(delta)
        self.population = self._clip_to_bounds((X1 + X2 + X3) / 3.0)
        self.fitness = self.fn(self.population)
        self._update_best()


# -----------------------------------------------------------------------------
# WOA
# -----------------------------------------------------------------------------

class WOA(BaseOptimizer):
    """Whale Optimization Algorithm."""

    def __init__(self, fn, dim, lb, ub, pop_size=30, max_iter=1000, num_runs=30,
                 b=1.0, device='cuda:0', dtype=torch.float64, seed=None):
        super().__init__(fn, dim, lb, ub, pop_size, max_iter, num_runs, device, dtype, seed)
        self.b = b

    def step(self, t):
        R, N, D = self.num_runs, self.pop_size, self.dim
        denom = max(1, self.max_iter - 1)
        a = 2.0 - 2.0 * (t / denom)

        X = self.population
        X_best = self.best_position.unsqueeze(1)
        rand_idx = self._randint(0, N, R, N)
        X_rand = _gather_random_agents(X, rand_idx)

        r = self._rand(R, N, D)
        A = 2.0 * a * r - a
        C = 2.0 * self._rand(R, N, D)
        p = self._rand(R, N, 1)
        l = -1.0 + 2.0 * self._rand(R, N, D)

        # Encircling around best.
        D_best = torch.abs(C * X_best - X)
        X_encircle = X_best - A * D_best

        # Random exploration around a randomly selected whale.
        D_rand = torch.abs(C * X_rand - X)
        X_explore = X_rand - A * D_rand

        # Spiral around best.
        D_spiral = torch.abs(X_best - X)
        X_spiral = D_spiral * torch.exp(self.b * l) * torch.cos(2.0 * math.pi * l) + X_best

        absA_mean = torch.mean(torch.abs(A), dim=-1, keepdim=True)
        use_explore = (absA_mean >= 1.0)
        first_branch = torch.where(use_explore, X_explore, X_encircle)
        X_new = torch.where(p < 0.5, first_branch, X_spiral)

        self.population = self._clip_to_bounds(X_new)
        self.fitness = self.fn(self.population)
        self._update_best()


# -----------------------------------------------------------------------------
# SCA
# -----------------------------------------------------------------------------

class SCA(BaseOptimizer):
    """Sine Cosine Algorithm."""

    def __init__(self, fn, dim, lb, ub, pop_size=30, max_iter=1000, num_runs=30,
                 a=2.0, device='cuda:0', dtype=torch.float64, seed=None):
        super().__init__(fn, dim, lb, ub, pop_size, max_iter, num_runs, device, dtype, seed)
        self.a = a

    def step(self, t):
        R, N, D = self.num_runs, self.pop_size, self.dim
        denom = max(1, self.max_iter - 1)
        r1 = self.a - self.a * (t / denom)
        r2 = 2.0 * math.pi * self._rand(R, N, D)
        r3 = 2.0 * self._rand(R, N, D)
        r4 = self._rand(R, N, D)
        target = self.best_position.unsqueeze(1)
        dist = torch.abs(r3 * target - self.population)
        X_sin = self.population + r1 * torch.sin(r2) * dist
        X_cos = self.population + r1 * torch.cos(r2) * dist
        X_new = torch.where(r4 < 0.5, X_sin, X_cos)
        self.population = self._clip_to_bounds(X_new)
        self.fitness = self.fn(self.population)
        self._update_best()


# -----------------------------------------------------------------------------
# HHO
# -----------------------------------------------------------------------------

class HHO(BaseOptimizer):
    """Harris Hawks Optimizer."""

    def __init__(self, fn, dim, lb, ub, pop_size=30, max_iter=1000, num_runs=30,
                 levy_beta=1.5, device='cuda:0', dtype=torch.float64, seed=None):
        super().__init__(fn, dim, lb, ub, pop_size, max_iter, num_runs, device, dtype, seed)
        self.levy_beta = levy_beta

    def step(self, t):
        R, N, D = self.num_runs, self.pop_size, self.dim
        T = max(1, self.max_iter)
        X = self.population
        rabbit = self.best_position.unsqueeze(1)
        X_mean = torch.mean(X, dim=1, keepdim=True)

        E0 = 2.0 * self._rand(R, N, 1) - 1.0
        E = 2.0 * E0 * (1.0 - t / T)
        absE = torch.abs(E)
        q = self._rand(R, N, 1)
        r = self._rand(R, N, 1)
        J = 2.0 * (1.0 - self._rand(R, N, 1))

        rand_idx = self._randint(0, N, R, N)
        X_rand = _gather_random_agents(X, rand_idx)

        # Exploration phase.
        X_explore_1 = X_rand - self._rand(R, N, D) * torch.abs(X_rand - 2.0 * self._rand(R, N, D) * X)
        random_bounds = self.lb.view(1, 1, D) + self._rand(R, N, D) * (self.ub - self.lb).view(1, 1, D)
        X_explore_2 = (rabbit - X_mean) - self._rand(R, N, D) * random_bounds
        X_explore = torch.where(q >= 0.5, X_explore_1, X_explore_2)

        # Exploitation phase.
        deltaX = rabbit - X
        X_soft = deltaX - E * torch.abs(J * rabbit - X)
        X_hard = rabbit - E * torch.abs(deltaX)

        # Progressive rapid dives.
        Y_soft = rabbit - E * torch.abs(J * rabbit - X)
        Y_hard = rabbit - E * torch.abs(J * rabbit - X_mean)
        S = self._rand(R, N, D)
        LF = _mantegna_levy((R, N, D), self.levy_beta, self.gen, self.device, self.dtype)
        Z_soft = Y_soft + S * LF
        Z_hard = Y_hard + S * LF
        Y_soft = self._clip_to_bounds(Y_soft)
        Y_hard = self._clip_to_bounds(Y_hard)
        Z_soft = self._clip_to_bounds(Z_soft)
        Z_hard = self._clip_to_bounds(Z_hard)

        fY_soft = self.fn(Y_soft)
        fZ_soft = self.fn(Z_soft)
        fY_hard = self.fn(Y_hard)
        fZ_hard = self.fn(Z_hard)

        X_dive_soft = torch.where((fY_soft < self.fitness).unsqueeze(-1), Y_soft,
                                  torch.where((fZ_soft < self.fitness).unsqueeze(-1), Z_soft, X))
        X_dive_hard = torch.where((fY_hard < self.fitness).unsqueeze(-1), Y_hard,
                                  torch.where((fZ_hard < self.fitness).unsqueeze(-1), Z_hard, X))

        exploit_no_dive = torch.where(absE >= 0.5, X_soft, X_hard)
        exploit_dive = torch.where(absE >= 0.5, X_dive_soft, X_dive_hard)
        X_exploit = torch.where(r >= 0.5, exploit_no_dive, exploit_dive)

        X_new = torch.where(absE >= 1.0, X_explore, X_exploit)
        self.population = self._clip_to_bounds(X_new)
        self.fitness = self.fn(self.population)
        self._update_best()


# -----------------------------------------------------------------------------
# AOA
# -----------------------------------------------------------------------------

class AOA(BaseOptimizer):
    """Archimedes Optimization Algorithm."""

    def __init__(self, fn, dim, lb, ub, pop_size=30, max_iter=1000, num_runs=30,
                 C1=2.0, C2=6.0, C3=2.0, C4=0.5,
                 acc_l=0.1, acc_u=0.9,
                 device='cuda:0', dtype=torch.float64, seed=None):
        super().__init__(fn, dim, lb, ub, pop_size, max_iter, num_runs, device, dtype, seed)
        self.C1 = C1
        self.C2 = C2
        self.C3 = C3
        self.C4 = C4
        self.acc_l = acc_l
        self.acc_u = acc_u
        self.den = None
        self.vol = None
        self.acc = None

    def _lazy_init(self):
        if self.den is None:
            R, N, D = self.num_runs, self.pop_size, self.dim
            self.den = self._rand(R, N, D)
            self.vol = self._rand(R, N, D)
            self.acc = self.lb.view(1, 1, D) + self._rand(R, N, D) * (self.ub - self.lb).view(1, 1, D)

    def step(self, t):
        self._lazy_init()
        R, N, D = self.num_runs, self.pop_size, self.dim
        eps = 1e-12
        T = max(1, self.max_iter)

        best_idx = torch.argmin(self.fitness, dim=1)
        run_idx = torch.arange(R, device=self.device)
        x_best = self.population[run_idx, best_idx].unsqueeze(1)
        den_best = self.den[run_idx, best_idx].unsqueeze(1)
        vol_best = self.vol[run_idx, best_idx].unsqueeze(1)
        acc_best = self.acc[run_idx, best_idx].unsqueeze(1)

        self.den = self.den + self._rand(R, N, D) * (den_best - self.den)
        self.vol = self.vol + self._rand(R, N, D) * (vol_best - self.vol)

        TF = math.exp((t - T) / T)
        d = math.exp((T - t) / T) - (t / T)

        if TF <= 0.5:
            mr_idx = self._randint(0, N, R, N)
            den_mr = _gather_random_agents(self.den, mr_idx)
            vol_mr = _gather_random_agents(self.vol, mr_idx)
            acc_mr = _gather_random_agents(self.acc, mr_idx)
            acc_new = (den_mr + vol_mr * acc_mr) / (self.den * self.vol + eps)
            x_rand = _gather_random_agents(self.population, mr_idx)
            X_new = self.population + self.C1 * self._rand(R, N, D) * acc_new * d * (x_rand - self.population)
        else:
            acc_new = (den_best + vol_best * acc_best) / (self.den * self.vol + eps)
            acc_min = acc_new.amin(dim=1, keepdim=True)
            acc_max = acc_new.amax(dim=1, keepdim=True)
            acc_norm = self.acc_l + (self.acc_u - self.acc_l) * (acc_new - acc_min) / (acc_max - acc_min + eps)
            T_factor = self.C3 * TF
            P = 2.0 * self._rand(R, N, D) - self.C4
            F_dir = torch.where(P <= 0.5, torch.ones_like(P), -torch.ones_like(P))
            X_new = x_best + F_dir * self.C2 * self._rand(R, N, D) * acc_norm * d * (T_factor * x_best - self.population)
            acc_new = acc_norm

        self.acc = acc_new
        self.population = self._clip_to_bounds(X_new)
        self.fitness = self.fn(self.population)
        self._update_best()


# -----------------------------------------------------------------------------
# EO
# -----------------------------------------------------------------------------

class EO(BaseOptimizer):
    """Equilibrium Optimizer."""

    def __init__(self, fn, dim, lb, ub, pop_size=30, max_iter=1000, num_runs=30,
                 a1=2.0, a2=1.0, GP=0.5,
                 device='cuda:0', dtype=torch.float64, seed=None):
        super().__init__(fn, dim, lb, ub, pop_size, max_iter, num_runs, device, dtype, seed)
        self.a1 = a1
        self.a2 = a2
        self.GP = GP

    def step(self, t):
        R, N, D = self.num_runs, self.pop_size, self.dim
        eps = 1e-12
        T = max(1, self.max_iter)
        top, _ = _topk_positions(self.population, self.fitness, 4)
        pool_avg = torch.mean(top, dim=1, keepdim=True)
        pool = torch.cat([top, pool_avg], dim=1)  # [R, 5, D]

        pool_idx = self._randint(0, 5, R, N)
        run_idx = torch.arange(R, device=self.device).view(R, 1).expand(R, N)
        Ceq = pool[run_idx, pool_idx]

        lamb = self._rand(R, N, D)
        r = self._rand(R, N, D)
        t_norm = (1.0 - t / T) ** (self.a2 * t / T)
        F = self.a1 * torch.sign(r - 0.5) * (torch.exp(-lamb * t_norm) - 1.0)

        r1 = self._rand(R, N, D)
        r2 = self._rand(R, N, D)
        GCP = torch.where(r2 >= self.GP, 0.5 * r1, torch.zeros_like(r1))
        G0 = GCP * (Ceq - lamb * self.population)
        G = G0 * F
        X_new = Ceq + (self.population - Ceq) * F + (G / (lamb + eps)) * (1.0 - F)

        self.population = self._clip_to_bounds(X_new)
        self.fitness = self.fn(self.population)
        self._update_best()
