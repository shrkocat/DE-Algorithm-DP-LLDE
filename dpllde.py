import numpy as np
from typing import Callable, Tuple, Optional, List
import warnings

warnings.filterwarnings('ignore')


class DPLLDE:
    """
    Diversity-Preserving Level-Learning Differential Evolution (DP-LLDE)

    Extends LBLDE (Qiao et al., 2022) with three contributions:

    1. Sequential Level Assignment — on G=1, each level l is assigned a random
       starting learning level k_o[l] from {2, …, NL}.  Per generation k_t[l]
       decays by 1 toward 1 (elite).  Reset every RESET_INTERVAL generations so
       exploration is periodically refreshed instead of permanently collapsing.

    2. Diversity-Aware Exemplar Selection — the two difference-vector candidates
       (r1, r2) are checked: if ||r1-r2|| < diversity_threshold, the less-diverse
       one is replaced with the most distant individual in the eligible pool,
       keeping mutation directions informative.

    3. DP-LLDE Mutation with Normalized Pressure — retains LBLDE's
       current-to-pbest/1 base but scales the difference vector:

           V_i = X_i + F*(X_pbest - X_i) + F*(r1 - r2) * (NL - k) / NL

       The factor (NL-k)/NL:
         • ≈ 1 when k is large (lower quality level) — full exploration pressure
         • → 0 as k → 1 (elite level) — reduced pressure, fine exploitation
         • resets with k_o so the decay is not permanent

    All other mechanics (crossover, boundary handling, greedy selection,
    Lehmer-mean F update, arithmetic-mean CR update, NLB bottom-level CR=1)
    are retained from LBLDE for fair comparison.
    """

    # How often (in generations) to reset the sequential level counters.
    # Prevents permanent collapse of k_t to 1 for all levels.
    RESET_INTERVAL: int = 20

    def __init__(
        self,
        objective_func: Callable,
        bounds: np.ndarray,
        NP: int = 100,
        NL: int = 4,
        NLB: int = 1,
        mu_CR_ini: float = 0.35,
        mu_F: float = 0.5,
        c: float = 0.1,
        diversity_threshold_ratio: float = 0.1,
        max_fes: int = 10000,
        seed: Optional[int] = None
    ):
        """
        Parameters
        ----------
        objective_func : Callable
            Objective function f(x) -> float to minimise.
        bounds : np.ndarray, shape (D, 2)
            Lower and upper bounds per dimension.
        NP : int
            Population size (adjusted to be divisible by NL).
        NL : int
            Number of hierarchical levels (level 1 = elite).
        NLB : int
            Number of bottom levels that use CR = 1 (from LBLDE).
        mu_CR_ini : float
            Initial mean for CR Gaussian sampling (paper default 0.35).
        mu_F : float
            Initial mean for F Cauchy sampling (default 0.5).
        c : float
            Learning rate for self-adaptive parameter update.
        diversity_threshold_ratio : float
            Diversity threshold = ratio × D^(0).
        max_fes : int
            Maximum number of objective function evaluations.
        seed : int, optional
            Random seed for reproducibility.
        """
        self.objective_func = objective_func
        self.bounds         = np.array(bounds)
        self.D              = len(bounds)
        self.NP             = (NP // NL) * NL     # ensure divisibility
        self.NL             = NL
        self.NLB            = NLB
        self.LS             = self.NP // NL        # individuals per level
        self.mu_CR          = mu_CR_ini
        self.mu_F           = mu_F
        self.c              = c
        self.diversity_threshold_ratio = diversity_threshold_ratio
        self.max_fes        = max_fes

        if seed is not None:
            np.random.seed(seed)

        # State initialised during optimize()
        self.D_0                = None
        self.diversity_threshold = None
        self.S_count            = 0
        self.k_o                = np.zeros(NL, dtype=int)
        self.k_t                = np.zeros(NL, dtype=int)

        self.S_CR: List[float] = []
        self.S_F:  List[float] = []

        self.best_solution    = None
        self.best_fitness     = np.inf
        self.fitness_history: List[float] = []

    # =========================================================================
    # Population helpers
    # =========================================================================

    def initialize_population(self) -> Tuple[np.ndarray, np.ndarray]:
        """Random uniform initialisation + evaluate."""
        population = np.random.uniform(
            self.bounds[:, 0],
            self.bounds[:, 1],
            size=(self.NP, self.D)
        )
        fitness = np.array([self.objective_func(ind) for ind in population])
        return population, fitness

    def compute_diversity(self, population: np.ndarray) -> float:
        """
        Fast approximation of mean pairwise distance using variance.
        Replaces O(n^2) loop — gives a proportional diversity signal.
        """
        if len(population) < 2:
            return 0.0
        return float(np.mean(np.std(population, axis=0)))

    def sort_population(
        self, population: np.ndarray, fitness: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sort population ascending by fitness (index 0 = best)."""
        idx = np.argsort(fitness)
        return population[idx], fitness[idx]

    def get_level_slice(self, level_idx: int) -> slice:
        """Slice of the sorted array for 0-indexed level."""
        return slice(level_idx * self.LS, (level_idx + 1) * self.LS)

    # =========================================================================
    # Parameter generation
    # =========================================================================

    def generate_CR(self, level_idx: int) -> float:
        """
        CR ~ N(μ_CR, 0.1) clipped to [0,1].
        Bottom NLB levels use CR = 1 (from LBLDE) to maximise crossover.
        """
        if level_idx >= self.NL - self.NLB:
            return 1.0
        CR = np.random.normal(self.mu_CR, 0.1)
        while CR < 0.0:
            CR = np.random.normal(self.mu_CR, 0.1)
        return min(CR, 1.0)

    def generate_F(self) -> float:
        """F ~ Cauchy(μ_F, 0.1) clipped to (0, 1]."""
        F = np.random.standard_cauchy() * 0.1 + self.mu_F
        while F <= 0.0:
            F = np.random.standard_cauchy() * 0.1 + self.mu_F
        return min(F, 1.0)

    # =========================================================================
    # Sequential Level Assignment  (DP-LLDE contribution 1)
    # =========================================================================

    def assign_levels(self, G: int) -> None:
        """
        On G == 1 or every RESET_INTERVAL generations: randomly re-draw k_o[l]
        from {2, …, NL} for each level, then set k_t[l] = k_o[l].

        On other generations: decay k_t[l] by 1 toward 1 (elite).

        Periodic reset prevents all levels from permanently collapsing to k=1
        and restores the exploration benefit of sequential level learning.
        """
        reset = (G == 1) or (G % self.RESET_INTERVAL == 0)
        for i in range(self.NL):
            if reset:
                self.k_o[i] = np.random.randint(2, self.NL + 1)
                self.k_t[i] = self.k_o[i]
            else:
                self.k_t[i] = max(1, self.k_t[i] - 1)

    # =========================================================================
    # Diversity-Aware Difference Vector Selection  (DP-LLDE contribution 2)
    # =========================================================================

    def _select_diverse_pair(
        self,
        pool: np.ndarray,
        exclude_idx: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pick two distinct individuals from pool.
        If ||r1 - r2|| < diversity_threshold, replace the closer one with the
        most distant member of pool from r1, preserving directional diversity.
        """
        n = len(pool)
        available = [j for j in range(n) if j != exclude_idx]
        if len(available) < 2:
            # Edge case: not enough candidates
            idx1, idx2 = 0, min(1, n - 1)
        else:
            idx1, idx2 = np.random.choice(available, size=2, replace=False)

        r1, r2 = pool[idx1].copy(), pool[idx2].copy()

        if np.linalg.norm(r1 - r2) < self.diversity_threshold:
            # Replace r2 with the most distant point from r1 in the pool
            dists = np.array([
                np.linalg.norm(pool[j] - r1)
                for j in range(n) if j != idx1
            ])
            best_j = [j for j in range(n) if j != idx1][int(np.argmax(dists))]
            r2 = pool[best_j].copy()

        return r1, r2

    # =========================================================================
    # DP-LLDE Mutation  (DP-LLDE contribution 3)
    # =========================================================================

    def mutate(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
        target: np.ndarray,
        target_global_idx: int,
        level_idx: int,
        F: float,
        level_k: int
    ) -> np.ndarray:
        """
        current-to-pbest/1 mutation with normalised level-based pressure:

            V_i = X_i + F*(X_pbest - X_i) + F*(r1 - r2) * (NL - k) / NL

        X_pbest is drawn from the top p_i fraction of the sorted population
        (same pi rule as LBLDE).  r1 is drawn from levels above the current
        level; r2 from levels up to and including the current level.
        The scaling factor (NL-k)/NL naturally decays mutation amplitude as
        k_t progresses toward the elite level.
        """
        # --- pbest selection (LBLDE pi rule) ---------------------------------
        if level_idx == 0:
            pi = 0.05
        else:
            pi = (level_idx * self.LS) / self.NP
        top_n = max(1, int(pi * self.NP))
        pbest_idx = np.random.randint(0, top_n)
        X_pbest = population[pbest_idx].copy()

        # --- diversity-aware r1, r2 selection --------------------------------
        if level_idx == 0:
            # Level 0: both r1 and r2 from level 0, excluding self
            local_idx = target_global_idx  # already 0-indexed within level 0
            pool = population[self.get_level_slice(0)]
            local_pos = target_global_idx % self.LS
            r1, r2 = self._select_diverse_pair(pool, exclude_idx=local_pos)
        else:
            # r1 from levels 0 … level_idx-1
            upper_pool = population[:level_idx * self.LS]
            # r2 from levels 0 … level_idx
            full_pool  = population[:(level_idx + 1) * self.LS]
            r1_idx = np.random.randint(0, len(upper_pool))
            r1     = upper_pool[r1_idx].copy()

            # diversity check on r2 relative to r1
            r2_candidates = np.delete(full_pool, r1_idx, axis=0) \
                            if r1_idx < len(full_pool) else full_pool
            if len(r2_candidates) == 0:
                r2 = r1.copy()
            else:
                r2_idx = np.random.randint(0, len(r2_candidates))
                r2 = r2_candidates[r2_idx].copy()
                if np.linalg.norm(r1 - r2) < self.diversity_threshold:
                    dists = np.linalg.norm(r2_candidates - r1, axis=1)
                    r2 = r2_candidates[int(np.argmax(dists))].copy()

        # --- normalised pressure scaling -------------------------------------
        scaling = (self.NL - level_k) / self.NL   # ∈ [0, (NL-1)/NL]

        mutant = target + F * (X_pbest - target) + F * (r1 - r2) * scaling
        return mutant

    # =========================================================================
    # Crossover and boundary handling  (unchanged from LBLDE / DE)
    # =========================================================================

    def crossover(self, target: np.ndarray, mutant: np.ndarray, CR: float) -> np.ndarray:
        """Binomial (uniform) crossover."""
        trial  = np.copy(target)
        j_rand = np.random.randint(0, self.D)
        for j in range(self.D):
            if np.random.rand() < CR or j == j_rand:
                trial[j] = mutant[j]
        return trial

    def bound_constraint(self, individual: np.ndarray) -> np.ndarray:
        """Clip to search bounds."""
        return np.clip(individual, self.bounds[:, 0], self.bounds[:, 1])

    # =========================================================================
    # Self-adaptive parameter update  (unchanged from LBLDE)
    # =========================================================================

    def update_parameters(self) -> None:
        """
        μ_CR ← arithmetic mean of S_CR.
        μ_F  ← Lehmer mean of S_F.
        Only updates on successful generations.
        """
        if len(self.S_CR) > 0:
            self.mu_CR = (1 - self.c) * self.mu_CR + self.c * float(np.mean(self.S_CR))

        if len(self.S_F) > 0:
            S_F_arr = np.array(self.S_F)
            denom   = S_F_arr.sum()
            if denom > 1e-12:
                mean_L = float(np.sum(S_F_arr ** 2) / denom)
                self.mu_F = (1 - self.c) * self.mu_F + self.c * mean_L

    # =========================================================================
    # Main optimisation loop
    # =========================================================================

    def optimize(self, verbose: bool = True) -> Tuple[np.ndarray, float, List[float]]:
        """
        Run DP-LLDE.

        Returns
        -------
        best_solution   : np.ndarray
        best_fitness    : float
        fitness_history : list  (one entry per generation)
        """
        FES = 0
        G   = 0
        self.S_count = 0

        # --- Initialise population -------------------------------------------
        population, fitness = self.initialize_population()
        FES += self.NP

        # --- Compute initial diversity D^(0) ----------------------------------
        self.D_0               = self.compute_diversity(population)
        self.diversity_threshold = self.diversity_threshold_ratio * self.D_0
        # Guard against degenerate threshold
        if self.diversity_threshold < 1e-10:
            self.diversity_threshold = 1e-10

        self.k_o = np.zeros(self.NL, dtype=int)
        self.k_t = np.zeros(self.NL, dtype=int)

        best_idx             = int(np.argmin(fitness))
        self.best_solution   = population[best_idx].copy()
        self.best_fitness    = float(fitness[best_idx])
        self.fitness_history = [self.best_fitness]

        if verbose:
            print(f"Initial D^(0) = {self.D_0:.4e}  |  "
                  f"Diversity threshold = {self.diversity_threshold:.4e}")
            print(f"Generation 0: Best Fitness = {self.best_fitness:.6e}")

        # --- Main loop -------------------------------------------------------
        while FES < self.max_fes:
            G += 1
            self.S_CR = []
            self.S_F  = []

            # Sort and partition
            population, fitness = self.sort_population(population, fitness)

            # Sequential level assignment (DP-LLDE contribution 1)
            self.assign_levels(G)

            new_population: List[np.ndarray] = []
            new_fitness:    List[float]       = []

            for level_idx in range(self.NL):
                level_k = self.k_t[level_idx]      # 1-indexed learning level

                for j in range(self.LS):
                    global_idx = level_idx * self.LS + j
                    target     = population[global_idx].copy()

                    # Adaptive CR (NLB bottom levels use CR=1)
                    CR_i = self.generate_CR(level_idx)
                    F_i  = self.generate_F()

                    # DP-LLDE mutation (contributions 2 + 3)
                    mutant = self.mutate(
                        population, fitness,
                        target, global_idx,
                        level_idx, F_i, level_k
                    )

                    # Crossover → trial vector
                    trial = self.crossover(target, mutant, CR_i)
                    trial = self.bound_constraint(trial)

                    # Evaluate
                    trial_fitness = self.objective_func(trial)
                    FES += 1

                    # Greedy selection
                    if trial_fitness <= fitness[global_idx]:
                        new_population.append(trial)
                        new_fitness.append(trial_fitness)
                        # Archive successful params (skip CR for NLB bottom levels)
                        if level_idx < self.NL - self.NLB:
                            self.S_CR.append(CR_i)
                        self.S_F.append(F_i)
                        self.S_count += 1
                    else:
                        new_population.append(target)
                        new_fitness.append(float(fitness[global_idx]))

                    if FES >= self.max_fes:
                        break
                if FES >= self.max_fes:
                    break

            population = np.array(new_population)
            fitness    = np.array(new_fitness)

            # Parameter adaptation
            self.update_parameters()

            # Track global best
            current_best_idx = int(np.argmin(fitness))
            if fitness[current_best_idx] < self.best_fitness:
                self.best_fitness  = float(fitness[current_best_idx])
                self.best_solution = population[current_best_idx].copy()

            self.fitness_history.append(self.best_fitness)

            if verbose and G % 10 == 0:
                print(f"Generation {G:4d}: Best = {self.best_fitness:.6e}  "
                      f"FES = {FES:6d}  μ_F = {self.mu_F:.3f}  "
                      f"μ_CR = {self.mu_CR:.3f}  "
                      f"k_t = {self.k_t.tolist()}")

        if verbose:
            print(f"\nOptimization complete.")
            print(f"Best Fitness             = {self.best_fitness:.6e}")
            print(f"Total Function Evals     = {FES}")
            print(f"Total Successful Updates = {self.S_count}")

        return self.best_solution, self.best_fitness, self.fitness_history


# =============================================================================
# Multiple-trial runner
# =============================================================================

def run_multiple_trials(
    func: Callable,
    bounds: np.ndarray,
    D: int,
    n_runs: int = 51,
    max_fes: int = None,
    verbose: bool = False
) -> dict:
    """
    Run DP-LLDE n_runs independent times and return statistical summary.

    Parameters
    ----------
    func    : Callable
    bounds  : np.ndarray, shape (D, 2)
    D       : int
    n_runs  : int  (paper uses 51)
    max_fes : int  (defaults to 10 000 × D)
    verbose : bool

    Returns
    -------
    dict with keys: mean, std, median, min, max, all_best, all_histories
    """
    if max_fes is None:
        max_fes = 10000 * D

    NP = 100 if D <= 50 else 160

    best_fitness_values: List[float] = []
    all_histories: List[List[float]] = []

    print(f"Running {n_runs} independent trials  "
          f"(NP={NP}, D={D}, MaxFES={max_fes}) ...")

    for run in range(n_runs):
        if verbose or (run + 1) % 10 == 0:
            print(f"  Run {run + 1}/{n_runs}")

        opt = DPLLDE(
            objective_func=func,
            bounds=bounds,
            NP=NP,
            NL=4,
            NLB=1,
            mu_CR_ini=0.35,
            max_fes=max_fes,
            seed=run
        )
        _, best_fit, history = opt.optimize(verbose=False)
        best_fitness_values.append(best_fit)
        all_histories.append(history)

    arr = np.array(best_fitness_values)
    results = {
        'mean'         : float(np.mean(arr)),
        'std'          : float(np.std(arr)),
        'median'       : float(np.median(arr)),
        'min'          : float(np.min(arr)),
        'max'          : float(np.max(arr)),
        'all_best'     : arr,
        'all_histories': all_histories
    }

    print(f"\nResults Summary:")
    print(f"  Mean ± Std : {results['mean']:.6e} ± {results['std']:.6e}")
    print(f"  Median     : {results['median']:.6e}")
    print(f"  Min        : {results['min']:.6e}")
    print(f"  Max        : {results['max']:.6e}")

    return results


# =============================================================================
# Classical benchmark functions
# =============================================================================

def sphere(x):
    return float(np.sum(x ** 2))

def rastrigin(x):
    n = len(x)
    return float(10 * n + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x)))

def rosenbrock(x):
    return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))

def ackley(x):
    n = len(x)
    return float(
        -20 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / n))
        - np.exp(np.sum(np.cos(2 * np.pi * x)) / n)
        + 20 + np.e
    )

def griewank(x):
    return float(
        np.sum(x ** 2) / 4000
        - np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
        + 1
    )


# =============================================================================
# Quick smoke-test
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("DP-LLDE Algorithm — Quick Smoke Test")
    print("=" * 70)

    D = 10
    test_functions = [
        ("Sphere",     sphere,     [-100,   100]),
        ("Rastrigin",  rastrigin,  [-5.12, 5.12]),
        ("Rosenbrock", rosenbrock, [-30,    30]),
        ("Ackley",     ackley,     [-32,    32]),
        ("Griewank",   griewank,   [-600,   600]),
    ]

    for name, func, bound_range in test_functions:
        bounds = np.array([bound_range] * D)
        opt = DPLLDE(
            objective_func=func,
            bounds=bounds,
            NP=100, NL=4, NLB=1,
            mu_CR_ini=0.35,
            max_fes=10000 * D,
            seed=42
        )
        _, best_fit, _ = opt.optimize(verbose=False)
        print(f"  {name:<12}: {best_fit:.6e}")

    print("=" * 70)
