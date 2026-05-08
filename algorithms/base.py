"""
Base optimizer class for vectorized GPU implementation.

All metaheuristic algorithms in this project inherit from BaseOptimizer.
The base class handles batched independent runs on GPU.

Tensor shapes (R = number of independent runs):
    population:       [R, N, D]   (R runs, N agents, D dimensions)
    fitness:          [R, N]
    best_position:    [R, D]
    best_fitness:     [R]
    convergence:      [R, T]      (best fitness per iteration)
"""

import torch
import time
from abc import ABC, abstractmethod


class BaseOptimizer(ABC):
    """Vectorized base class. Each instance runs R independent optimization runs in parallel."""

    def __init__(self, fn, dim, lb, ub, pop_size=30, max_iter=1000,
                 num_runs=30, device='cuda:0', dtype=torch.float64, seed=None):
        """
        Args:
            fn: callable taking [R, N, D] tensor, returning [R, N] fitness tensor.
                The function must be vectorized over both R and N dimensions.
            dim: problem dimension D.
            lb, ub: scalars or [D] tensors. Lower and upper bounds.
            pop_size: number of agents per run.
            max_iter: total iterations T.
            num_runs: number of independent runs R.
            device: 'cuda:0', 'cuda:1', or 'cpu'.
            dtype: torch.float64 (recommended for benchmark precision) or torch.float32.
            seed: master random seed. If None, uses torch default.
        """
        self.fn = fn
        self.dim = dim
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.num_runs = num_runs
        self.device = device
        self.dtype = dtype
        self.seed = seed

        # Bounds: broadcast scalars to [D] tensors.
        if isinstance(lb, (int, float)):
            self.lb = torch.full((dim,), float(lb), device=device, dtype=dtype)
        else:
            self.lb = torch.as_tensor(lb, device=device, dtype=dtype)
        if isinstance(ub, (int, float)):
            self.ub = torch.full((dim,), float(ub), device=device, dtype=dtype)
        else:
            self.ub = torch.as_tensor(ub, device=device, dtype=dtype)

        # Random generator. One per device for reproducibility.
        self.gen = torch.Generator(device=device)
        if seed is not None:
            self.gen.manual_seed(seed)

        # State (filled by run()).
        self.population = None        # [R, N, D]
        self.fitness = None           # [R, N]
        self.best_position = None     # [R, D]
        self.best_fitness = None      # [R]
        self.convergence = None       # [R, T]
        self.elapsed = None           # wall clock seconds

    # ------------------------- utilities -------------------------

    def _rand(self, *shape):
        """Uniform random tensor on device with given shape."""
        return torch.rand(*shape, generator=self.gen, device=self.device, dtype=self.dtype)

    def _randn(self, *shape):
        """Standard normal tensor on device with given shape."""
        return torch.randn(*shape, generator=self.gen, device=self.device, dtype=self.dtype)

    def _randint(self, low, high, *shape):
        """Random integers in [low, high)."""
        return torch.randint(low, high, shape, generator=self.gen, device=self.device)

    def _initialize_population(self):
        """Uniform random initialization in bounds."""
        R, N, D = self.num_runs, self.pop_size, self.dim
        self.population = self.lb + (self.ub - self.lb) * self._rand(R, N, D)
        self.fitness = self.fn(self.population)
        # Track best per run.
        best_idx = torch.argmin(self.fitness, dim=1)              # [R]
        run_idx = torch.arange(R, device=self.device)
        self.best_position = self.population[run_idx, best_idx].clone()  # [R, D]
        self.best_fitness = self.fitness[run_idx, best_idx].clone()      # [R]
        self.convergence = torch.empty(R, self.max_iter, device=self.device, dtype=self.dtype)

    def _clip_to_bounds(self, x):
        """Clip [..., D] tensor to bounds with broadcasting."""
        return torch.maximum(torch.minimum(x, self.ub), self.lb)

    def _update_best(self):
        """After population/fitness are updated, refresh best_position and best_fitness."""
        R = self.num_runs
        run_idx = torch.arange(R, device=self.device)
        cur_best_idx = torch.argmin(self.fitness, dim=1)
        cur_best_fit = self.fitness[run_idx, cur_best_idx]
        improved = cur_best_fit < self.best_fitness
        if improved.any():
            cur_best_pos = self.population[run_idx, cur_best_idx]
            self.best_fitness = torch.where(improved, cur_best_fit, self.best_fitness)
            self.best_position = torch.where(
                improved.unsqueeze(-1), cur_best_pos, self.best_position
            )

    # ------------------------- API -------------------------

    @abstractmethod
    def step(self, t):
        """One iteration of the algorithm. Subclasses implement.

        Must update self.population (with clipping) and self.fitness, then call
        self._update_best().

        Args:
            t: current iteration index, 0..max_iter-1.
        """
        raise NotImplementedError

    def run(self, verbose=False):
        """Execute all max_iter iterations on GPU. Returns dict of results."""
        torch.cuda.synchronize() if 'cuda' in self.device else None
        start = time.time()
        self._initialize_population()
        for t in range(self.max_iter):
            self.step(t)
            self.convergence[:, t] = self.best_fitness
            if verbose and (t + 1) % max(1, self.max_iter // 10) == 0:
                mean_best = self.best_fitness.mean().item()
                print(f"  iter {t+1}/{self.max_iter}  mean_best={mean_best:.6e}")
        torch.cuda.synchronize() if 'cuda' in self.device else None
        self.elapsed = time.time() - start
        return self.results()

    def results(self):
        """Return result dict with statistics over runs."""
        bf = self.best_fitness.detach().cpu().numpy()
        return {
            'algorithm': self.__class__.__name__,
            'best_fitness_per_run': bf,
            'best_position_per_run': self.best_position.detach().cpu().numpy(),
            'convergence': self.convergence.detach().cpu().numpy(),
            'mean': float(bf.mean()),
            'std': float(bf.std()),
            'min': float(bf.min()),
            'max': float(bf.max()),
            'median': float(torch.median(self.best_fitness).item()),
            'elapsed_sec': self.elapsed,
            'num_runs': self.num_runs,
            'pop_size': self.pop_size,
            'max_iter': self.max_iter,
            'dim': self.dim,
        }
