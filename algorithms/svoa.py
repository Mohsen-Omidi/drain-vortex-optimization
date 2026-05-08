"""
Stratospheric Vortex Optimization Algorithm (SVOA)
==================================================

GPU-vectorized PyTorch implementation for the DVO/MVDO experimental pipeline.

This implementation follows the mathematical model and pseudo-code reported in:
He et al., "Stratospheric vortex optimization algorithm: a new meta-heuristic
algorithm, applied to continuous optimization and engineering applications",
Cluster Computing, 2026.

Main mechanisms implemented:
    1. Vortex centre selection: k = 3 in the first half, k = 1 in the second half.
    2. Optional SSW-inspired centre splitting after stagnation.
    3. Radial velocity driven by fitness gradient toward/away from a vortex centre.
    4. Azimuthal velocity using a random orthogonal direction.
    5. Exploration update with Levy/SSW perturbation for t < 2T/3.
    6. Exploitation update guided by the global best for t >= 2T/3.
    7. Greedy selection and best-so-far update.
"""

import math
import torch

from .base import BaseOptimizer


def _svoa_mantegna_levy(shape, beta, generator, device, dtype):
    """Mantegna Levy-flight sample with stable index beta in (0, 2)."""
    num = math.gamma(1.0 + beta) * math.sin(math.pi * beta / 2.0)
    den = math.gamma((1.0 + beta) / 2.0) * beta * (2.0 ** ((beta - 1.0) / 2.0))
    sigma_u = (num / den) ** (1.0 / beta)
    u = torch.randn(*shape, generator=generator, device=device, dtype=dtype) * sigma_u
    v = torch.randn(*shape, generator=generator, device=device, dtype=dtype)
    return u / (torch.abs(v) ** (1.0 / beta) + 1e-12)


class SVOA(BaseOptimizer):
    """Stratospheric Vortex Optimization Algorithm.

    Parameters from the SVOA paper:
        kr_max = 0.6, kr_min = 0.4
        ktheta_max = 1.0, ktheta_min = 0.6
    """

    def __init__(self, fn, dim, lb, ub,
                 pop_size=30, max_iter=1000, num_runs=30,
                 kr_max=0.6, kr_min=0.4,
                 ktheta_max=1.0, ktheta_min=0.6,
                 levy_beta=1.5,
                 stagnation_limit=5,
                 pv_threshold=0.1,
                 rossby_threshold=0.5,
                 device='cuda:0', dtype=torch.float64, seed=None):
        super().__init__(fn=fn, dim=dim, lb=lb, ub=ub,
                         pop_size=pop_size, max_iter=max_iter, num_runs=num_runs,
                         device=device, dtype=dtype, seed=seed)

        self.kr_max = kr_max
        self.kr_min = kr_min
        self.ktheta_max = ktheta_max
        self.ktheta_min = ktheta_min
        self.levy_beta = levy_beta
        self.stagnation_limit = int(stagnation_limit)
        self.pv_threshold = pv_threshold
        self.rossby_threshold = rossby_threshold

        self.stagnation_count = torch.zeros(self.num_runs, device=self.device, dtype=torch.long)

    def _safe_random_perpendicular(self, unit_radial):
        R, N, D = unit_radial.shape
        q = self._randn(R, N, D)
        proj = (q * unit_radial).sum(dim=-1, keepdim=True)
        tangential = q - proj * unit_radial
        norm = torch.norm(tangential, dim=-1, keepdim=True)
        bad = (norm < 1e-12).expand(-1, -1, D)
        if bad.any():
            q2 = self._randn(R, N, D)
            proj2 = (q2 * unit_radial).sum(dim=-1, keepdim=True)
            tangential2 = q2 - proj2 * unit_radial
            norm2 = torch.norm(tangential2, dim=-1, keepdim=True)
            tangential = torch.where(bad, tangential2, tangential)
            norm = torch.where(bad, norm2, norm)
        return tangential / (norm + 1e-12)

    def _current_best_and_worst(self):
        R = self.num_runs
        run_idx = torch.arange(R, device=self.device)
        best_fit, best_idx = self.fitness.min(dim=1)
        worst_fit, worst_idx = self.fitness.max(dim=1)
        x_best = self.population[run_idx, best_idx]
        x_worst = self.population[run_idx, worst_idx]
        return x_best, best_fit, x_worst, worst_fit

    def _build_vortex_centres(self, t):
        R, N, D = self.num_runs, self.pop_size, self.dim
        max_centres = 4
        centres = torch.zeros(R, max_centres, D, device=self.device, dtype=self.dtype)
        centre_fitness = torch.full((R, max_centres), float("inf"), device=self.device, dtype=self.dtype)
        active = torch.zeros(R, max_centres, device=self.device, dtype=torch.bool)

        base_k = 3 if t < (self.max_iter / 2.0) else 1
        base_k = min(base_k, N)

        sorted_fit, sorted_idx = torch.sort(self.fitness, dim=1)
        run_idx = torch.arange(R, device=self.device).view(R, 1).expand(R, base_k)
        top_idx = sorted_idx[:, :base_k]
        centres[:, :base_k, :] = self.population[run_idx, top_idx]
        centre_fitness[:, :base_k] = sorted_fit[:, :base_k]
        active[:, :base_k] = True

        rossby_strength = abs(math.sin(2.0 * math.pi * float(t) / max(1.0, float(self.max_iter))))

        if t < (2.0 * self.max_iter / 3.0) and rossby_strength > self.rossby_threshold:
            x_best, f_best, x_worst, f_worst = self._current_best_and_worst()
            pv = torch.abs(f_best - f_worst) / (torch.norm(x_best - x_worst, dim=-1) + 1e-12)
            can_split = (self.stagnation_count >= self.stagnation_limit) & (pv < self.pv_threshold)
            if can_split.any() and base_k < max_centres:
                split_choice = torch.randint(low=0, high=base_k, size=(R,), generator=self.gen, device=self.device)
                rr = torch.arange(R, device=self.device)
                x_split = centres[rr, split_choice]
                delta = 0.2 * torch.norm(x_split - x_best, dim=-1, keepdim=True) * rossby_strength
                x_new = x_split + delta * self._randn(R, D)
                x_new = self._clip_to_bounds(x_new.unsqueeze(1)).squeeze(1)
                slot = base_k
                centres[can_split, slot, :] = x_new[can_split]
                f_new = self.fn(x_new.unsqueeze(1)).squeeze(1)
                centre_fitness[can_split, slot] = f_new[can_split]
                active[can_split, slot] = True

        return centres, centre_fitness, active, rossby_strength

    def _sample_vortex_for_agents(self, centres, centre_fitness, active):
        R, N, D = self.num_runs, self.pop_size, self.dim
        Kmax = centres.shape[1]
        probs = active.to(self.dtype)
        probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1.0)
        probs_flat = probs.unsqueeze(1).expand(R, N, Kmax).reshape(R * N, Kmax)
        idx = torch.multinomial(probs_flat, num_samples=1, replacement=True, generator=self.gen)
        idx = idx.view(R, N)
        run_idx = torch.arange(R, device=self.device).view(R, 1).expand(R, N)
        selected_centres = centres[run_idx, idx]
        selected_fit = centre_fitness[run_idx, idx]
        return selected_centres, selected_fit

    def _svoa_boundary_rule(self, x_new, x_v):
        lb = self.lb.view(1, 1, self.dim)
        ub = self.ub.view(1, 1, self.dim)
        upper_repair = x_v + ub / 3.0
        lower_repair = x_v + lb / 3.0
        repaired = torch.where(x_new > ub, upper_repair,
                               torch.where(x_new < lb, lower_repair, x_new))
        return self._clip_to_bounds(repaired)

    def step(self, t):
        R, N, D = self.num_runs, self.pop_size, self.dim
        eps = 1e-12
        old_run_best = self.fitness.min(dim=1).values

        centres, centre_fitness, active, rossby_strength = self._build_vortex_centres(t)
        x_v, f_v = self._sample_vortex_for_agents(centres, centre_fitness, active)

        x = self.population
        d = x - x_v
        dist = torch.norm(d, dim=-1)
        zero_dist = dist < 1e-12
        if zero_dist.any():
            random_d = self._randn(R, N, D)
            d = torch.where(zero_dist.unsqueeze(-1), random_d, d)
            dist = torch.norm(d, dim=-1)

        unit_radial = d / (dist.unsqueeze(-1) + eps)

        sigma_f = torch.std(self.fitness, dim=1, keepdim=True)
        ratio = sigma_f / torch.maximum(sigma_f, torch.full_like(sigma_f, eps))
        kr = self.kr_min + (self.kr_max - self.kr_min) * torch.exp(-ratio)
        ktheta = self.ktheta_min + (self.ktheta_max - self.ktheta_min) * (1.0 - torch.exp(-ratio))

        grad_like = (self.fitness - f_v) / (dist + eps)
        sr = torch.where(self._rand(R, N) < 0.5,
                         torch.ones(R, N, device=self.device, dtype=self.dtype),
                         -torch.ones(R, N, device=self.device, dtype=self.dtype))
        radial_scale = (0.5 + 0.5 * torch.tanh(grad_like)).unsqueeze(-1)
        v_radial = -sr.unsqueeze(-1) * kr.view(R, 1, 1) * d * radial_scale

        e_theta = self._safe_random_perpendicular(unit_radial)
        v_theta = ktheta.view(R, 1, 1) * dist.unsqueeze(-1) * e_theta
        velocity = v_radial + v_theta

        if t < (2.0 * self.max_iter / 3.0):
            levy_step = _svoa_mantegna_levy((R, N, D), self.levy_beta, self.gen, self.device, self.dtype)
            A = levy_step * rossby_strength
            x_candidate = x + 0.5 * velocity + 0.2 * A * (x_v - (x + velocity))
        else:
            x_best, _, _, _ = self._current_best_and_worst()
            rand_idx = torch.randint(low=0, high=N, size=(R, N), generator=self.gen, device=self.device)
            run_idx = torch.arange(R, device=self.device).view(R, 1).expand(R, N)
            x_r1 = self.population[run_idx, rand_idx]
            x_candidate = x + (x_best.unsqueeze(1) - x_r1)

        x_candidate = self._svoa_boundary_rule(x_candidate, x_v)

        f_candidate = self.fn(x_candidate)
        improved = f_candidate < self.fitness
        self.population = torch.where(improved.unsqueeze(-1), x_candidate, self.population)
        self.fitness = torch.where(improved, f_candidate, self.fitness)
        self._update_best()

        new_run_best = self.fitness.min(dim=1).values
        better_run = new_run_best < old_run_best
        self.stagnation_count = torch.where(
            better_run,
            torch.zeros_like(self.stagnation_count),
            self.stagnation_count + 1
        )
