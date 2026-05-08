"""
Corrected/vectorized GPU baseline metaheuristics for the MVDO project.

Drop-in replacement for algorithms/baselines.py.

Main audit corrections relative to the first baseline file:
    * GWO keeps historical Alpha/Beta/Delta leaders, as in the common GWO
      pseudocode where leaders are updated only when better wolves are found.
    * WOA uses scalar coefficient sampling per whale (A, C, l, p shaped
      [R, N, 1]) to match the common original MATLAB implementation style.
    * AOA normalizes acceleration in both exploration and exploitation phases,
      and keeps a historical best object with its density/volume/acceleration.
    * EO keeps a historical equilibrium pool of the four best candidates.

All classes inherit from algorithms.base.BaseOptimizer and are compatible with:
    opt = Algorithm(fn, dim, lb, ub, pop_size, max_iter, num_runs, device, dtype, seed)
    result = opt.run()
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
    """Gather agents per run.

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


def _topk_pool(existing_pos, existing_fit, population, fitness, k):
    """Merge an existing top-k pool with the current population and return top-k.

    existing_pos: [R, k, D]
    existing_fit: [R, k]
    population:   [R, N, D]
    fitness:      [R, N]
    """
    pool_pos = torch.cat([existing_pos, population], dim=1)
    pool_fit = torch.cat([existing_fit, fitness], dim=1)
    sorted_fit, sorted_idx = torch.sort(pool_fit, dim=1)
    top_idx = sorted_idx[:, :k]
    R = population.shape[0]
    run_idx = torch.arange(R, device=population.device).view(R, 1).expand(R, k)
    return pool_pos[run_idx, top_idx].clone(), sorted_fit[:, :k].clone()


# -----------------------------------------------------------------------------
# PSO
# -----------------------------------------------------------------------------

class PSO(BaseOptimizer):
    """Particle Swarm Optimization.

    Uses a common linearly decaying inertia setup. Velocity clamping is enabled
    by default to avoid unstable jumps in CEC search ranges.
    """

    def __init__(self, fn, dim, lb, ub, pop_size=30, max_iter=1000, num_runs=30,
                 w_max=0.9, w_min=0.4, c1=2.0, c2=2.0, v_max_frac=0.2,
                 device='cuda:0', dtype=torch.float64, seed=None):
        super().__init__(fn, dim, lb, ub, pop_size, max_iter, num_runs, device, dtype, seed)
        self.w_max = w_max
        self.w_min = w_min
        self.c1 = c1
        self.c2 = c2
        self.v_max_frac = v_max_frac
        self.velocity = None
        self.pbest_position = None
        self.pbest_fitness = None

    def _lazy_init(self):
        if self.velocity is None:
            span = (self.ub - self.lb).view(1, 1, self.dim)
            vmax = self.v_max_frac * span
            self.velocity = (2.0 * self._rand(self.num_runs, self.pop_size, self.dim) - 1.0) * vmax
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
        span = (self.ub - self.lb).view(1, 1, self.dim)
        vmax = self.v_max_frac * span
        self.velocity = torch.maximum(torch.minimum(self.velocity, vmax), -vmax)

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
    """Grey Wolf Optimizer with historical Alpha/Beta/Delta leaders."""

    def __init__(self, fn, dim, lb, ub, pop_size=30, max_iter=1000, num_runs=30,
                 device='cuda:0', dtype=torch.float64, seed=None):
        super().__init__(fn, dim, lb, ub, pop_size, max_iter, num_runs, device, dtype, seed)
        self.leader_pos = None  # [R, 3, D]
        self.leader_fit = None  # [R, 3]

    def _update_leaders(self):
        current_pos, current_fit = _topk_positions(self.population, self.fitness, 3)
        if self.leader_pos is None:
            self.leader_pos = current_pos
            self.leader_fit = current_fit
        else:
            self.leader_pos, self.leader_fit = _topk_pool(
                self.leader_pos, self.leader_fit, self.population, self.fitness, 3
            )

    def step(self, t):
        self._update_leaders()
        denom = max(1, self.max_iter - 1)
        a = 2.0 - 2.0 * (t / denom)

        alpha = self.leader_pos[:, 0, :].unsqueeze(1)
        beta = self.leader_pos[:, 1, :].unsqueeze(1)
        delta = self.leader_pos[:, 2, :].unsqueeze(1)

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
        self._update_leaders()


# -----------------------------------------------------------------------------
# WOA
# -----------------------------------------------------------------------------

class WOA(BaseOptimizer):
    """Whale Optimization Algorithm.

    Uses scalar A, C, l, and p per whale/run, broadcast over dimensions. This
    matches common original MATLAB-style implementations more closely than
    drawing separate coefficients for every dimension.
    """

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

        r1 = self._rand(R, N, 1)
        r2 = self._rand(R, N, 1)
        A = 2.0 * a * r1 - a
        C = 2.0 * r2
        p = self._rand(R, N, 1)
        l = -1.0 + 2.0 * self._rand(R, N, 1)

        D_best = torch.abs(C * X_best - X)
        X_encircle = X_best - A * D_best

        D_rand = torch.abs(C * X_rand - X)
        X_explore = X_rand - A * D_rand

        D_spiral = torch.abs(X_best - X)
        X_spiral = D_spiral * torch.exp(self.b * l) * torch.cos(2.0 * math.pi * l) + X_best

        use_explore = torch.abs(A) >= 1.0
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
    """Harris Hawks Optimizer.

    Note: this implementation evaluates additional rapid-dive candidates inside
    each iteration. Use a real function-evaluation budget when making final
    claims, or run it with fewer iterations as an approximation.
    """

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

        X_explore_1 = X_rand - self._rand(R, N, D) * torch.abs(X_rand - 2.0 * self._rand(R, N, D) * X)
        random_bounds = self.lb.view(1, 1, D) + self._rand(R, N, D) * (self.ub - self.lb).view(1, 1, D)
        X_explore_2 = (rabbit - X_mean) - self._rand(R, N, D) * random_bounds
        X_explore = torch.where(q >= 0.5, X_explore_1, X_explore_2)

        deltaX = rabbit - X
        X_soft = deltaX - E * torch.abs(J * rabbit - X)
        X_hard = rabbit - E * torch.abs(deltaX)

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
    """Archimedes Optimization Algorithm with corrected acceleration handling."""

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
        self.best_den = None
        self.best_vol = None
        self.best_acc = None
        self.best_obj_fit = None
        self.best_obj_pos = None

    def _lazy_init(self):
        if self.den is None:
            R, N, D = self.num_runs, self.pop_size, self.dim
            self.den = self._rand(R, N, D)
            self.vol = self._rand(R, N, D)
            self.acc = self.lb.view(1, 1, D) + self._rand(R, N, D) * (self.ub - self.lb).view(1, 1, D)
            self._update_aoa_best(force=True)

    def _update_aoa_best(self, force=False):
        R = self.num_runs
        run_idx = torch.arange(R, device=self.device)
        cur_idx = torch.argmin(self.fitness, dim=1)
        cur_fit = self.fitness[run_idx, cur_idx]
        cur_pos = self.population[run_idx, cur_idx]
        cur_den = self.den[run_idx, cur_idx]
        cur_vol = self.vol[run_idx, cur_idx]
        cur_acc = self.acc[run_idx, cur_idx]

        if force or self.best_obj_fit is None:
            self.best_obj_fit = cur_fit.clone()
            self.best_obj_pos = cur_pos.clone()
            self.best_den = cur_den.clone()
            self.best_vol = cur_vol.clone()
            self.best_acc = cur_acc.clone()
            return

        improved = cur_fit < self.best_obj_fit
        self.best_obj_fit = torch.where(improved, cur_fit, self.best_obj_fit)
        self.best_obj_pos = torch.where(improved.unsqueeze(-1), cur_pos, self.best_obj_pos)
        self.best_den = torch.where(improved.unsqueeze(-1), cur_den, self.best_den)
        self.best_vol = torch.where(improved.unsqueeze(-1), cur_vol, self.best_vol)
        self.best_acc = torch.where(improved.unsqueeze(-1), cur_acc, self.best_acc)

    def _normalize_acc(self, acc_raw):
        # Per-run scalar min/max over all agents and dimensions.
        acc_min = acc_raw.amin(dim=(1, 2), keepdim=True)
        acc_max = acc_raw.amax(dim=(1, 2), keepdim=True)
        return self.acc_l + (self.acc_u - self.acc_l) * (acc_raw - acc_min) / (acc_max - acc_min + 1e-12)

    def step(self, t):
        self._lazy_init()
        R, N, D = self.num_runs, self.pop_size, self.dim
        eps = 1e-12
        T = max(1, self.max_iter)

        x_best = self.best_obj_pos.unsqueeze(1)
        den_best = self.best_den.unsqueeze(1)
        vol_best = self.best_vol.unsqueeze(1)
        acc_best = self.best_acc.unsqueeze(1)

        self.den = self.den + self._rand(R, N, D) * (den_best - self.den)
        self.vol = self.vol + self._rand(R, N, D) * (vol_best - self.vol)

        TF = math.exp((t - T) / T)
        d = math.exp((T - t) / T) - (t / T)

        if TF <= 0.5:
            mr_idx = self._randint(0, N, R, N)
            den_mr = _gather_random_agents(self.den, mr_idx)
            vol_mr = _gather_random_agents(self.vol, mr_idx)
            acc_mr = _gather_random_agents(self.acc, mr_idx)
            acc_raw = (den_mr + vol_mr * acc_mr) / (self.den * self.vol + eps)
            acc_norm = self._normalize_acc(acc_raw)
            x_rand = _gather_random_agents(self.population, mr_idx)
            X_new = self.population + self.C1 * self._rand(R, N, D) * acc_norm * d * (x_rand - self.population)
        else:
            acc_raw = (den_best + vol_best * acc_best) / (self.den * self.vol + eps)
            acc_norm = self._normalize_acc(acc_raw)
            T_factor = self.C3 * TF
            P = 2.0 * self._rand(R, N, D) - self.C4
            F_dir = torch.where(P <= 0.5, torch.ones_like(P), -torch.ones_like(P))
            X_new = x_best + F_dir * self.C2 * self._rand(R, N, D) * acc_norm * d * (T_factor * x_best - self.population)

        self.acc = acc_norm
        self.population = self._clip_to_bounds(X_new)
        self.fitness = self.fn(self.population)
        self._update_best()
        self._update_aoa_best(force=False)


# -----------------------------------------------------------------------------
# EO
# -----------------------------------------------------------------------------

class EO(BaseOptimizer):
    """Equilibrium Optimizer with a historical equilibrium pool."""

    def __init__(self, fn, dim, lb, ub, pop_size=30, max_iter=1000, num_runs=30,
                 a1=2.0, a2=1.0, GP=0.5,
                 device='cuda:0', dtype=torch.float64, seed=None):
        super().__init__(fn, dim, lb, ub, pop_size, max_iter, num_runs, device, dtype, seed)
        self.a1 = a1
        self.a2 = a2
        self.GP = GP
        self.eq_pos = None
        self.eq_fit = None

    def _update_equilibrium_pool(self):
        top, top_fit = _topk_positions(self.population, self.fitness, 4)
        if self.eq_pos is None:
            self.eq_pos = top
            self.eq_fit = top_fit
        else:
            self.eq_pos, self.eq_fit = _topk_pool(self.eq_pos, self.eq_fit, self.population, self.fitness, 4)

    def step(self, t):
        self._update_equilibrium_pool()
        R, N, D = self.num_runs, self.pop_size, self.dim
        eps = 1e-12
        T = max(1, self.max_iter)
        pool_avg = torch.mean(self.eq_pos, dim=1, keepdim=True)
        pool = torch.cat([self.eq_pos, pool_avg], dim=1)  # [R, 5, D]

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
        self._update_equilibrium_pool()
