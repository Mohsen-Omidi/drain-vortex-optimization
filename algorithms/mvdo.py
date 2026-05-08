"""
Multi-Vortex Drain Optimization (MVDO)
======================================

A population-based metaheuristic inspired by multi-drain free-vortex flow.
Search agents are treated as water particles. The best candidate solutions form
multiple drain vortices, and each agent is moved by a combination of:

    1. far-field random exploration with weak drainage drift,
    2. spiral inward motion with radial/tangential decomposition,
    3. local core exploitation around a drain vortex,
    4. stochastic vortex switching for basin-to-basin information exchange,
    5. splash-out using Levy flight for agents stagnating near a vortex core.

This version fixes the main implementation issues found in the first draft:
    - correct radial geometry in the spiral update,
    - scale-aware far-field noise,
    - normalized free-vortex tangential velocity C/rho,
    - robust rank/softmax-based vortex fitness weights,
    - true switching to a different vortex when K > 1,
    - vortex-center elitism using current population + previous population +
      previous vortex centers,
    - stuck-count update based on actual candidate improvement.

All operations are vectorized over R independent runs and N agents per run.
"""

import math
from typing import Optional, Tuple

import torch

from .base import BaseOptimizer


_EPS = 1e-12


def _mantegna_levy(shape, beta, generator, device, dtype):
    """Levy flight step using Mantegna's algorithm.

    Parameters
    ----------
    shape : tuple
        Output tensor shape.
    beta : float
        Stable distribution index in (0, 2). A common value is 1.5.
    generator : torch.Generator
        Random generator used by the optimizer.
    device, dtype
        Tensor placement and precision.

    Returns
    -------
    torch.Tensor
        Heavy-tailed Levy steps with the requested shape.
    """
    if not (0.0 < beta < 2.0):
        raise ValueError(f"levy_beta must be in (0, 2), got {beta}.")

    num = math.gamma(1.0 + beta) * math.sin(math.pi * beta / 2.0)
    den = math.gamma((1.0 + beta) / 2.0) * beta * (2.0 ** ((beta - 1.0) / 2.0))
    sigma_u = (num / den) ** (1.0 / beta)

    u = torch.randn(*shape, generator=generator, device=device, dtype=dtype) * sigma_u
    v = torch.randn(*shape, generator=generator, device=device, dtype=dtype)
    return u / (torch.abs(v) ** (1.0 / beta) + _EPS)


class MVDO(BaseOptimizer):
    """Multi-Vortex Drain Optimization.

    Parameters
    ----------
    fn : callable
        Vectorized objective function. Expected input shape: [R, N, D].
    dim : int
        Problem dimension.
    lb, ub : float or tensor-like
        Lower and upper bounds.
    pop_size : int
        Number of search agents per run.
    max_iter : int
        Maximum number of iterations.
    num_runs : int
        Number of independent runs executed in parallel.
    K : int
        Number of drain vortices. Internally clipped to pop_size.
    C : float
        Vortex strength in the normalized free-vortex term C / rho.
    gamma : float
        Radial shrink coefficient in the spiral phase.
    alpha1 : float
        Far-field drift coefficient toward the assigned vortex.
    beta1 : float
        Far-field random exploration coefficient, scaled by variable range.
    R_far_frac : float
        Far-field threshold on normalized distance rho = r / diameter.
    R_near_frac : float
        Core threshold on normalized distance rho = r / diameter.
    sigma0_frac : float
        Initial core Gaussian radius as a fraction of the search-space diameter.
    p_switch : float
        Probability of stochastic vortex switching per agent per iteration.
    tau_stay : int
        Number of consecutive non-improving core iterations before splash eligibility.
    p_splash : float
        Probability of splash once an agent is eligible.
    levy_beta : float
        Mantegna Levy-flight stability index.
    L_splash_frac : float
        Splash step size as a fraction of the search-space diameter.
    fitness_weight_mode : str
        'rank' is robust and recommended. 'fitness_shift' keeps the older behaviour.
    softmax_beta0, softmax_beta1 : float
        Rank/fitness pressure schedule for vortex weights. The value moves from
        beta0 to beta1 as iterations progress.
    use_greedy_agent_update : bool
        If True, each agent keeps its previous position when the candidate is worse.
        Default False preserves stronger exploration. Vortex centers are still elitist.
    """

    def __init__(self, fn, dim, lb, ub,
                 pop_size=30, max_iter=1000, num_runs=30,
                 K=6, C=0.2, gamma=0.5,
                 alpha1=0.1, beta1=0.5,
                 R_far_frac=0.5, R_near_frac=0.05,
                 sigma0_frac=0.1,
                 p_switch=0.08,
                 tau_stay=5, p_splash=0.1,
                 levy_beta=1.5, L_splash_frac=0.25,
                 fitness_weight_mode="rank",
                 softmax_beta0=1.0, softmax_beta1=6.0,
                 use_greedy_agent_update=True,
                 adaptive_spiral=True,
                 c_min_frac=0.15,
                 force_splash_replace=False,
                 splash_uniform_prob=0.35,
                 device='cuda:0', dtype=torch.float64, seed=None):
        super().__init__(fn=fn, dim=dim, lb=lb, ub=ub,
                         pop_size=pop_size, max_iter=max_iter, num_runs=num_runs,
                         device=device, dtype=dtype, seed=seed)

        if K < 1:
            raise ValueError(f"K must be >= 1, got {K}.")
        self.K = min(int(K), int(pop_size))

        self.C = float(C)
        self.gamma = float(gamma)
        self.alpha1 = float(alpha1)
        self.beta1 = float(beta1)
        self.R_far_frac = float(R_far_frac)
        self.R_near_frac = float(R_near_frac)
        self.p_switch = float(p_switch)
        self.tau_stay = int(tau_stay)
        self.p_splash = float(p_splash)
        self.levy_beta = float(levy_beta)
        self.fitness_weight_mode = str(fitness_weight_mode)
        self.softmax_beta0 = float(softmax_beta0)
        self.softmax_beta1 = float(softmax_beta1)
        self.use_greedy_agent_update = bool(use_greedy_agent_update)
        self.adaptive_spiral = bool(adaptive_spiral)
        self.c_min_frac = float(c_min_frac)
        self.force_splash_replace = bool(force_splash_replace)
        self.splash_uniform_prob = float(splash_uniform_prob)

        if self.R_near_frac <= 0.0:
            raise ValueError("R_near_frac must be positive.")
        if self.R_far_frac <= self.R_near_frac:
            raise ValueError("R_far_frac must be greater than R_near_frac.")

        # Geometry and scale. Keep tensors ready for GPU-vectorized operations.
        self.var_scale = (self.ub - self.lb).view(1, 1, self.dim)
        self.diameter = torch.norm(self.ub - self.lb).clamp_min(_EPS).item()
        self.sigma0 = float(sigma0_frac) * self.diameter
        self.L_splash = float(L_splash_frac) * self.diameter

        # Small normalized core radius prevents the C/r singularity.
        self.rho_core = max(0.5 * self.R_near_frac, 1e-6)

        # State for splash-out tracking. stuck_count[r, n] counts consecutive
        # non-improving iterations while the agent is in the core region.
        self.stuck_count: Optional[torch.Tensor] = None

        # Vortex centers: [R, K, D] and their fitness [R, K].
        self.vortices: Optional[torch.Tensor] = None
        self.vortex_fitness: Optional[torch.Tensor] = None

    # --------------------------------------------------------------
    # Schedules and weights
    # --------------------------------------------------------------

    def _progress(self, t: int) -> float:
        """Return normalized progress in [0, 1]."""
        if self.max_iter <= 1:
            return 1.0
        return float(min(max(t / (self.max_iter - 1), 0.0), 1.0))

    def _phi(self, t: int) -> float:
        """Linear control parameter decreasing from 2 to 0."""
        return 2.0 * (1.0 - self._progress(t))

    def _weight_beta(self, t: Optional[int]) -> float:
        """Selection pressure schedule for vortex weights."""
        if t is None:
            return self.softmax_beta0
        p = self._progress(t)
        return self.softmax_beta0 + (self.softmax_beta1 - self.softmax_beta0) * p

    def _vortex_probabilities(self, t: Optional[int] = None) -> torch.Tensor:
        """Return robust vortex attraction probabilities P_k for each run.

        The recommended default is rank-based weighting. It is stable for CEC
        functions with shifts, rotations, and biases, and also safe when objective
        values are zero, negative, or very differently scaled.
        """
        if self.vortex_fitness is None:
            raise RuntimeError("Vortex centers have not been initialized.")

        beta = self._weight_beta(t)
        R, K = self.vortex_fitness.shape

        if K == 1:
            return torch.ones(R, 1, device=self.device, dtype=self.dtype)

        if self.fitness_weight_mode == "rank":
            # Vortices are stored sorted from best to worst. rank 0 = best.
            ranks = torch.arange(K, device=self.device, dtype=self.dtype).view(1, K)
            logits = -beta * ranks / max(K - 1, 1)
            return torch.softmax(logits.expand(R, K), dim=1)

        if self.fitness_weight_mode == "fitness_shift":
            # Older scale-sensitive mode, kept for ablation/backward compatibility.
            max_fit = self.vortex_fitness.max(dim=1, keepdim=True).values
            weights = (max_fit - self.vortex_fitness).clamp_min(0.0) + _EPS
            return weights / weights.sum(dim=1, keepdim=True)

        if self.fitness_weight_mode == "fitness_softmax":
            # Normalize per run before softmax to reduce scale sensitivity.
            f = self.vortex_fitness
            mean = f.mean(dim=1, keepdim=True)
            std = f.std(dim=1, keepdim=True).clamp_min(_EPS)
            z = (f - mean) / std
            return torch.softmax(-beta * z, dim=1)

        raise ValueError(
            "fitness_weight_mode must be one of: 'rank', 'fitness_shift', 'fitness_softmax'."
        )

    # --------------------------------------------------------------
    # Vortex center management
    # --------------------------------------------------------------

    def _select_vortex_centers(self):
        """Pick top-K best agents from the current population."""
        self._select_vortex_centers_from_pool(self.population, self.fitness)

    def _select_vortex_centers_from_pool(self, pool_X: torch.Tensor, pool_fit: torch.Tensor):
        """Pick top-K best solutions per run from an arbitrary candidate pool.

        This enables true elitism for vortex centers. The pool can contain the
        current population, the previous population, and previous vortex centers.
        """
        sorted_fit, sorted_idx = torch.sort(pool_fit, dim=1)              # [R, M]
        topk_idx = sorted_idx[:, :self.K]                                  # [R, K]
        run_idx = torch.arange(self.num_runs, device=self.device).unsqueeze(1).expand(-1, self.K)
        self.vortices = pool_X[run_idx, topk_idx].clone()                  # [R, K, D]
        self.vortex_fitness = sorted_fit[:, :self.K].clone()               # [R, K]

    # --------------------------------------------------------------
    # Vortex assignment
    # --------------------------------------------------------------

    def _vortex_assignment(self, t: Optional[int] = None):
        """Assign each agent to one vortex by Score = P_k / (rho_{i,k}+eps).

        rho is normalized distance r / diameter. This makes the assignment more
        stable across dimensions and bound ranges.
        """
        R, N, K = self.num_runs, self.pop_size, self.K

        P = self._vortex_probabilities(t)                                  # [R, K]

        diff = self.population.unsqueeze(2) - self.vortices.unsqueeze(1)    # [R, N, K, D]
        dist = torch.norm(diff, dim=-1)                                     # [R, N, K]
        rho = dist / (self.diameter + _EPS)                                 # [R, N, K]

        score = P.unsqueeze(1) / (rho + _EPS)                               # [R, N, K]
        assign = torch.argmax(score, dim=-1)                                # [R, N]

        run_idx = torch.arange(R, device=self.device).view(R, 1).expand(R, N)
        agent_idx = torch.arange(N, device=self.device).view(1, N).expand(R, N)
        r_assigned = dist[run_idx, agent_idx, assign]                       # [R, N]
        rho_assigned = rho[run_idx, agent_idx, assign]                      # [R, N]
        v_assigned = self.vortices[run_idx, assign]                         # [R, N, D]

        return assign, r_assigned, rho_assigned, v_assigned, P, dist, rho

    # --------------------------------------------------------------
    # Stochastic Vortex Switching (SVS)
    # --------------------------------------------------------------

    def _stochastic_vortex_switching(self, assign, dist, rho, P):
        """Occasionally reassign an agent to a different vortex.

        Unlike the first draft, this version enforces a real switch when K > 1
        by setting the currently assigned vortex probability to zero before
        sampling the alternative vortex.
        """
        R, N, K = self.num_runs, self.pop_size, self.K

        if K <= 1 or self.p_switch <= 0.0:
            run_idx = torch.arange(R, device=self.device).view(R, 1).expand(R, N)
            agent_idx = torch.arange(N, device=self.device).view(1, N).expand(R, N)
            return assign, dist[run_idx, agent_idx, assign], rho[run_idx, agent_idx, assign], self.vortices[run_idx, assign]

        roll = self._rand(R, N)
        do_switch = roll < self.p_switch                                   # [R, N]

        # Per-agent categorical distribution over vortices, excluding current assignment.
        P_agent = P.unsqueeze(1).expand(R, N, K).clone()                    # [R, N, K]
        P_agent.scatter_(2, assign.unsqueeze(-1), 0.0)
        P_agent = P_agent / P_agent.sum(dim=2, keepdim=True).clamp_min(_EPS)

        cand = torch.multinomial(
            P_agent.reshape(R * N, K),
            num_samples=1,
            generator=self.gen,
            replacement=True,
        ).view(R, N)

        assign_new = torch.where(do_switch, cand, assign)

        run_idx = torch.arange(R, device=self.device).view(R, 1).expand(R, N)
        agent_idx = torch.arange(N, device=self.device).view(1, N).expand(R, N)
        r_new = dist[run_idx, agent_idx, assign_new]
        rho_new = rho[run_idx, agent_idx, assign_new]
        v_new = self.vortices[run_idx, assign_new]
        return assign_new, r_new, rho_new, v_new

    # --------------------------------------------------------------
    # Helper: random unit vector perpendicular to radial direction
    # --------------------------------------------------------------

    def _random_perp_unit(self, e_r):
        """Return random unit vectors perpendicular to e_r.

        e_r has shape [R, N, D]. For D = 1, a truly perpendicular direction does
        not exist. In that case we return zeros, so the tangential component is
        safely removed and MVDO behaves as a radial/core optimizer.
        """
        R, N, D = e_r.shape
        if D == 1:
            return torch.zeros_like(e_r)

        q = self._randn(R, N, D)
        proj = (q * e_r).sum(dim=-1, keepdim=True)
        e_t = q - proj * e_r
        norm = torch.norm(e_t, dim=-1, keepdim=True)

        # Degenerate residuals are rare but possible. Reroll once.
        bad = norm < 1e-8
        if bad.any():
            q2 = self._randn(R, N, D)
            proj2 = (q2 * e_r).sum(dim=-1, keepdim=True)
            e_t2 = q2 - proj2 * e_r
            norm2 = torch.norm(e_t2, dim=-1, keepdim=True)
            e_t = torch.where(bad.expand_as(e_t), e_t2, e_t)
            norm = torch.where(bad, norm2, norm)

        return e_t / (norm + _EPS)

    # --------------------------------------------------------------
    # Three motion phases
    # --------------------------------------------------------------

    def _phase1_far(self, X, V, phi):
        """Phase 1: scale-aware weak drift + Gaussian random walk."""
        R, N, D = X.shape
        eta = self._randn(R, N, D)

        # Noise is scaled by variable range and decays with phi.
        # Dividing by sqrt(D) avoids excessively large diagonal jumps in high D.
        dim_scale = math.sqrt(max(D, 1))
        noise = self.beta1 * (phi / 2.0) * eta * self.var_scale / dim_scale
        drift = self.alpha1 * phi * (V - X)
        return X + drift + noise

    def _phase2_spiral(self, X, V, r, rho, phi):
        """Phase 2: spiral inward with corrected radial geometry.

        Geometry convention:
            e_r = (X - V) / ||X - V|| points from the vortex center to the agent.
            X_new = V + offset, so the offset must be expressed using e_r.

        The tangential speed follows a regularized normalized free-vortex law:
            v_theta = C / (rho + rho_core)
        where rho = r / diameter. This keeps C meaningful across dimensions and
        avoids the singularity near the vortex core.
        """
        R, N, D = X.shape

        e_r = (X - V) / (r.unsqueeze(-1) + _EPS)                           # center -> agent
        e_t = self._random_perp_unit(e_r)

        omega = self._rand(R, N) * (2.0 * math.pi)
        cos_w = torch.cos(omega).unsqueeze(-1)
        sin_w = torch.sin(omega).unsqueeze(-1)

        # progress is 0 at the first iteration and 1 at the last iteration.
        # The first MVDO draft used the opposite pressure schedule, which made
        # radial drainage strongest early and weakest late. That caused premature
        # attraction in the beginning and weak final convergence. In MVDO-v2 we
        # invert the schedule: exploration is freer early, while drainage becomes
        # stronger toward the end.
        progress = 1.0 - (phi / 2.0)

        if self.adaptive_spiral:
            # Vortex circulation is annealed as the basin drains. Early iterations
            # keep stronger tangential motion for exploration; late iterations keep
            # a small non-zero circulation to preserve the free-vortex character but
            # prevent endless orbiting around the drain.
            c_floor = max(self.c_min_frac, 0.0)
            C_eff = self.C * (c_floor + (1.0 - c_floor) * (1.0 - progress))
            radial_pressure = 0.25 + 0.75 * progress
        else:
            C_eff = self.C
            radial_pressure = 0.25 + 0.75 * (phi / 2.0)

        # Regularized free-vortex tangential strength using normalized radius.
        v_theta = (C_eff / (rho + self.rho_core)).unsqueeze(-1)

        shrink = (1.0 - self.gamma * radial_pressure) * r.unsqueeze(-1)
        shrink = torch.clamp(shrink, min=0.0)

        # Keep the tangential component bounded so it cannot explode near the core.
        # This still preserves the free-vortex trend because v_theta increases as rho decreases.
        v_theta = torch.clamp(v_theta, max=10.0)

        offset = shrink * (cos_w * e_r + sin_w * v_theta * e_t)
        return V + offset

    def _phase3_core(self, V, phi):
        """Phase 3: tight Gaussian sampling around the assigned vortex."""
        R, N, D = V.shape
        sigma = self.sigma0 * (phi / 2.0)
        # Keep a tiny minimum radius to prevent complete freezing in the final iterations.
        sigma = max(float(sigma), 1e-12 * self.diameter)
        eta = self._randn(R, N, D)
        return V + sigma * eta / math.sqrt(max(D, 1))

    def _splash_out(self, X, V_best):
        """Re-launch stuck agents via a mixed splash mechanism.

        Two splash modes are used:
            1. Levy splash around the alpha vortex for long-range basin escape.
            2. Uniform splash inside the search box for true population renewal.

        The uniform component is important when greedy update is enabled; otherwise
        all agents can become personal-best holders and the population may lose
        its ability to explore new basins on composition functions.
        """
        R, N, D = X.shape
        levy = _mantegna_levy((R, N, D), self.levy_beta, self.gen, self.device, self.dtype)
        levy_pos = V_best.unsqueeze(1) + (self.L_splash / math.sqrt(max(D, 1))) * levy

        uniform_pos = self.lb.view(1, 1, D) + self._rand(R, N, D) * self.var_scale.view(1, 1, D)
        if self.splash_uniform_prob <= 0.0:
            return levy_pos
        if self.splash_uniform_prob >= 1.0:
            return uniform_pos
        use_uniform = self._rand(R, N, 1) < self.splash_uniform_prob
        return torch.where(use_uniform, uniform_pos, levy_pos)

    # --------------------------------------------------------------
    # One iteration
    # --------------------------------------------------------------

    @torch.no_grad()
    def step(self, t):
        # Lazy initialization: vortex centers and stuck counters need population/fitness.
        if self.vortices is None:
            self._select_vortex_centers()
            self.stuck_count = torch.zeros(
                self.num_runs, self.pop_size, device=self.device, dtype=torch.long
            )

        phi = self._phi(t)

        old_X = self.population.clone()
        old_fit = self.fitness.clone()
        old_V = self.vortices.clone()
        old_V_fit = self.vortex_fitness.clone()

        # 1. Vortex assignment and optional stochastic switching.
        assign, r, rho, V_assigned, P, dist_all, rho_all = self._vortex_assignment(t)
        assign, r, rho, V_assigned = self._stochastic_vortex_switching(
            assign, dist_all, rho_all, P
        )

        X = self.population

        # 2. Phase masks using normalized radius rho = r / diameter.
        m1 = rho > self.R_far_frac
        m3 = rho <= self.R_near_frac
        m2 = ~(m1 | m3)

        # 3. Candidate positions from the three phases.
        X1 = self._phase1_far(X, V_assigned, phi)
        X2 = self._phase2_spiral(X, V_assigned, r, rho, phi)
        X3 = self._phase3_core(V_assigned, phi)

        X_new = torch.where(m1.unsqueeze(-1), X1, torch.where(m2.unsqueeze(-1), X2, X3))

        # 4. Splash-out for agents that were already eligible at the start of this iteration.
        do_splash = torch.zeros(self.num_runs, self.pop_size, device=self.device, dtype=torch.bool)
        if self.tau_stay > 0 and self.p_splash > 0.0:
            eligible = (self.stuck_count >= self.tau_stay) & m3
            if eligible.any():
                roll = self._rand(self.num_runs, self.pop_size)
                do_splash = eligible & (roll < self.p_splash)
                if do_splash.any():
                    V_best = self.vortices[:, 0, :]
                    X_splash = self._splash_out(X, V_best)
                    X_new = torch.where(do_splash.unsqueeze(-1), X_splash, X_new)

        # 5. Boundary handling and evaluation.
        X_new = self._clip_to_bounds(X_new)
        new_fit = self.fn(X_new)

        # Greedy update preserves improvements, but forced splash replacement can
        # deliberately move stagnated agents even when the new sampled point is worse.
        # The global best archive is still preserved by _update_best(), so this does
        # not lose the best solution found so far.
        if self.use_greedy_agent_update:
            accept = new_fit < old_fit
            if self.force_splash_replace:
                accept = accept | do_splash
            X_final = torch.where(accept.unsqueeze(-1), X_new, old_X)
            fit_final = torch.where(accept, new_fit, old_fit)
        else:
            X_final = X_new
            fit_final = new_fit

        # 6. Update stuck counters using actual improvement of the candidate move.
        improved = fit_final < old_fit
        reset = improved | (~m3) | do_splash
        self.stuck_count = torch.where(reset, torch.zeros_like(self.stuck_count), self.stuck_count)
        self.stuck_count = torch.where(m3 & (~improved) & (~do_splash), self.stuck_count + 1, self.stuck_count)

        # 7. Update population and global best.
        self.population = X_final
        self.fitness = fit_final
        self._update_best()

        # 8. Elitist vortex-center refresh from current population, previous
        # population, and previous vortex centers. This implements V <- top-K(X ∪ V).
        pool_X = torch.cat([self.population, old_X, old_V], dim=1)
        pool_fit = torch.cat([self.fitness, old_fit, old_V_fit], dim=1)
        self._select_vortex_centers_from_pool(pool_X, pool_fit)
