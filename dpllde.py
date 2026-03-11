import numpy as np
from typing import Callable, Tuple, Optional, List
import warnings

warnings.filterwarnings('ignore')


class DPLLDE:
    """
    Diversity-Preserving Level-Learning Differential Evolution (DP-LLDE)

    Extends LBLDE with three core contributions from the paper:

    1. Sequential Level Assignment — each individual in level l is assigned a
       learning source level k_o[l] drawn randomly from {2, …, NL} on the first
       generation.  In subsequent generations the target level decays by one per
       generation (k_t[l] = max(1, k_o[l] - (G-1))), guaranteeing strictly
       upward, monotonic progression toward the elite level (level 1).

    2. Diversity-Aware Exemplar Selection — two exemplars e1, e2 are drawn from
       the assigned learning level k.  If ||e1 - e2|| falls below a diversity
       threshold (scaled from the initial population diversity D^(0)), one
       exemplar is replaced with the most distant candidate in that level,
       preventing premature convergence through exemplar collapse.

    3. DP-LLDE Mutation with Normalized Pressure (Eq. 10 of the paper):
           V_i = e1 + F_i × (e2 - X_i) × (NL - k) / NL
       The factor (NL-k)/NL decays to zero as k → NL (elite level), naturally
       reducing mutation pressure as learning quality improves, and normalises
       the pressure across all level indices.

    Crossover, boundary handling, greedy selection and parameter self-adaptation
    (Lehmer mean for F, arithmetic mean for CR) are retained unchanged from DE.

    Reference pseudocode: Algorithm 1, DP-LLDE (Diversity Preserving
    Level-Based-Learning Differential Evolution).
    """

    def __init__(
        self,
        objective_func: Callable,
        bounds: np.ndarray,
        NP: int = 100,
        NL: int = 4,
        mu_CR_ini: float = 0.5,       # Paper default: 0.5 (differs from LBLDE's 0.35)
        mu_F: float = 0.5,
        c: float = 0.1,
        diversity_threshold_ratio: float = 0.1,  # threshold = ratio × D^(0)
        max_fes: int = 10000,
        seed: Optional[int] = None
    ):
        """
        Parameters
        ----------
        objective_func : Callable
            Objective function to minimise; signature f(x: np.ndarray) -> float.
        bounds : np.ndarray, shape (D, 2)
            Lower and upper bounds for each dimension.
        NP : int
            Population size.  Adjusted internally to be divisible by NL.
        NL : int
            Number of hierarchical levels (level 1 = elite / best fitness).
        mu_CR_ini : float
            Initial mean of the CR Gaussian sampling distribution (default 0.5).
        mu_F : float
            Initial mean of the F Cauchy sampling distribution (default 0.5).
        c : float
            Learning rate for self-adaptive parameter update.
        diversity_threshold_ratio : float
            Diversity threshold = ratio × D^(0).  When ||e1-e2|| drops below
            this threshold the less-diverse exemplar is replaced.
        max_fes : int
            Maximum number of objective function evaluations.
        seed : int, optional
            Random seed for reproducibility.
        """
        self.objective_func = objective_func
        self.bounds = np.array(bounds)
        self.D = len(bounds)
        self.NP = (NP // NL) * NL       # ensure divisibility
        self.NL = NL
        self.LS = self.NP // NL          # individuals per level
        self.mu_CR = mu_CR_ini
        self.mu_F = mu_F
        self.c = c
        self.diversity_threshold_ratio = diversity_threshold_ratio
        self.max_fes = max_fes

        if seed is not None:
            np.random.seed(seed)

        # --- State set during optimize() ---
        self.D_0 = None          # initial population diversity  (Line 3)
        self.diversity_threshold = None
        self.S_count = 0         # successful improvement counter (Line 3)
        self.k_o = np.zeros(NL, dtype=int)   # initial level assignments (Line 3)
        self.k_t = np.zeros(NL, dtype=int)   # current level assignments

        # Successful parameter archives
        self.S_CR: List[float] = []
        self.S_F:  List[float] = []

        # Output tracking
        self.best_solution = None
        self.best_fitness = np.inf
        self.fitness_history: List[float] = []

    # =========================================================================
    # Population helpers
    # =========================================================================

    def initialize_population(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create random uniform population and evaluate every individual."""
        population = np.random.uniform(
            self.bounds[:, 0],
            self.bounds[:, 1],
            size=(self.NP, self.D)
        )
        fitness = np.array([self.objective_func(ind) for ind in population])
        return population, fitness

    def compute_diversity(self, population: np.ndarray) -> float:
        """
        Compute population diversity D as the mean pairwise Euclidean distance
        between all individuals — used to set D^(0) at initialisation and
        to derive the diversity threshold for exemplar replacement.
        """
        n = len(population)
        if n < 2:
            return 0.0
        total = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                total += np.linalg.norm(population[i] - population[j])
                count += 1
        return total / count if count > 0 else 0.0

    def sort_population(
        self, population: np.ndarray, fitness: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sort population by fitness ascending (index 0 = best)."""
        idx = np.argsort(fitness)
        return population[idx], fitness[idx]

    def get_level_slice(self, level_idx: int) -> slice:
        """Return the slice of the sorted population array for level l."""
        return slice(level_idx * self.LS, (level_idx + 1) * self.LS)

    # =========================================================================
    # Parameter generation  (retained from LBLDE)
    # =========================================================================

    def generate_CR(self) -> float:
        """Sample CR ~ N(μ_CR, 0.1), clipped to [0, 1]."""
        CR = np.random.normal(self.mu_CR, 0.1)
        while CR < 0.0:
            CR = np.random.normal(self.mu_CR, 0.1)
        return min(CR, 1.0)

    def generate_F(self) -> float:
        """Sample F ~ Cauchy(μ_F, 0.1), clipped to (0, 1]."""
        F = np.random.standard_cauchy() * 0.1 + self.mu_F
        while F <= 0.0:
            F = np.random.standard_cauchy() * 0.1 + self.mu_F
        return min(F, 1.0)

    # =========================================================================
    # Sequential Level Assignment  (Algorithm lines 7-10)
    # =========================================================================

    def assign_levels(self, G: int) -> None:
        """
        Lines 7-10 of pseudocode.

        G == 1 : k_o[l] drawn uniformly from {2, …, NL}  (Eq. 7)
        G  > 1 : k_t[l] = max(1, k_o[l] - (G-1))         (Eq. 8 / line 9)

        This guarantees strictly upward, monotonic progression: learning
        starts from a randomly assigned intermediate level and migrates toward
        the elite level (k = 1) as generations advance.  Learning terminates
        when k_t reaches 1 (Eq. 9: k_t > L is handled by the max(1, …) floor).
        """
        for i in range(self.NL):
            if G == 1:
                # k_o ~ {2, 3, …, NL}  (never starts at level 1)
                self.k_o[i] = np.random.randint(2, self.NL + 1)
            # Decay toward level 1; floor at 1 so learning never stops entirely
            self.k_t[i] = max(1, self.k_o[i] - (G - 1))

    # =========================================================================
    # Diversity-Aware Exemplar Selection  (Algorithm lines 14-16)
    # =========================================================================

    def select_diverse_exemplars(
        self,
        population: np.ndarray,
        level_k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Lines 14-16 of pseudocode.

        Select two exemplars e1, e2 from level k (1-indexed, so the slice is
        level_k-1 in 0-indexed Python).  If ||e1 - e2|| < diversity_threshold,
        replace the exemplar closer to the other with the individual in level k
        that is furthest from e1 — preserving diversity in the mutation
        direction vector.
        """
        k_idx = level_k - 1                # convert 1-indexed level to 0-indexed
        k_idx = min(k_idx, self.NL - 1)    # safety clamp
        lvl_slice = self.get_level_slice(k_idx)
        level_members = population[lvl_slice]       # shape (LS, D)
        n = len(level_members)

        if n < 2:
            # Edge case: return the only member twice
            return level_members[0].copy(), level_members[0].copy()

        # Pick two distinct random exemplars from this level
        idx1, idx2 = np.random.choice(n, size=2, replace=False)
        e1, e2 = level_members[idx1].copy(), level_members[idx2].copy()

        # Diversity check — line 16
        if np.linalg.norm(e1 - e2) < self.diversity_threshold:
            # Replace the exemplar closer to the other with the most distant
            # individual in the level from e1
            distances = np.array([
                np.linalg.norm(level_members[j] - e1)
                for j in range(n)
                if j != idx1
            ])
            diverse_idx_in_filtered = np.argmax(distances)
            # Map back to original indices (skip idx1)
            candidate_indices = [j for j in range(n) if j != idx1]
            diverse_idx = candidate_indices[diverse_idx_in_filtered]
            e2 = level_members[diverse_idx].copy()

        return e1, e2

    # =========================================================================
    # DP-LLDE Mutation  (Algorithm lines 17-18, Eq. 10)
    # =========================================================================

    def mutate(
        self,
        target: np.ndarray,
        e1: np.ndarray,
        e2: np.ndarray,
        F: float,
        level_k: int
    ) -> np.ndarray:
        """
        DP-LLDE mutation with normalized pressure (Eq. 10):

            V_i = e1 + F_i × (e2 - X_i) × (NL - k) / NL

        The scaling factor (NL-k)/NL:
          • equals (NL-1)/NL ≈ 1 when k is small (low quality level) — full pressure
          • approaches 0 as k → NL (elite level) — reduced pressure near optimum
          • decouples mutation behaviour from absolute level index values
          • guarantees monotonic decay of mutation as learning quality improves
        """
        scaling = (self.NL - level_k) / self.NL
        mutant = e1 + F * (e2 - target) * scaling
        return mutant

    # =========================================================================
    # Crossover and boundary constraint  (retained from LBLDE)
    # =========================================================================

    def crossover(self, target: np.ndarray, mutant: np.ndarray, CR: float) -> np.ndarray:
        """Binomial (uniform) crossover — unchanged from DE."""
        trial = np.copy(target)
        j_rand = np.random.randint(0, self.D)
        for j in range(self.D):
            if np.random.rand() < CR or j == j_rand:
                trial[j] = mutant[j]
        return trial

    def bound_constraint(self, individual: np.ndarray) -> np.ndarray:
        """Clip individual to search bounds."""
        return np.clip(individual, self.bounds[:, 0], self.bounds[:, 1])

    # =========================================================================
    # Self-adaptive parameter update  (Algorithm line 23)
    # =========================================================================

    def update_parameters(self) -> None:
        """
        Update μ_CR (arithmetic mean) and μ_F (Lehmer mean) from the
        archives of successful control parameters collected during the
        current generation.
        """
        if len(self.S_CR) > 0:
            mean_A = np.mean(self.S_CR)
            self.mu_CR = (1 - self.c) * self.mu_CR + self.c * mean_A

        if len(self.S_F) > 0:
            S_F_arr = np.array(self.S_F)
            denom = S_F_arr.sum()
            if denom > 1e-12:
                mean_L = np.sum(S_F_arr ** 2) / denom
                self.mu_F = (1 - self.c) * self.mu_F + self.c * mean_L

    # =========================================================================
    # Main optimisation loop
    # =========================================================================

    def optimize(self, verbose: bool = True) -> Tuple[np.ndarray, float, List[float]]:
        """
        Run DP-LLDE optimisation.

        Returns
        -------
        best_solution : np.ndarray   — best solution found
        best_fitness  : float        — best objective value
        fitness_history : list       — best fitness logged once per generation
        """
        # ── Line 1 – Counters & defaults ────────────────────────────────────
        FES = 0
        G = 0
        self.S_count = 0

        # ── Line 2 – Initialise population ──────────────────────────────────
        population, fitness = self.initialize_population()
        FES += self.NP

        # ── Line 3 – Compute D^(0), initialise S_count and k_o ─────────────
        self.D_0 = self.compute_diversity(population)
        self.diversity_threshold = self.diversity_threshold_ratio * self.D_0
        self.k_o = np.zeros(self.NL, dtype=int)
        self.k_t = np.zeros(self.NL, dtype=int)

        # Seed best tracking
        best_idx = np.argmin(fitness)
        self.best_solution = population[best_idx].copy()
        self.best_fitness = float(fitness[best_idx])
        self.fitness_history = [self.best_fitness]

        if verbose:
            print(f"Initial D^(0) = {self.D_0:.4e}  |  "
                  f"Diversity threshold = {self.diversity_threshold:.4e}")
            print(f"Generation 0: Best Fitness = {self.best_fitness:.6e}")

        # ── Line 4 – Main loop ───────────────────────────────────────────────
        while FES < self.max_fes:
            G += 1
            self.S_CR = []
            self.S_F  = []

            # Line 6 – Sort ascending and partition into NL levels
            population, fitness = self.sort_population(population, fitness)

            # ── Lines 7-10 – Sequential Level Assignment ─────────────────────
            self.assign_levels(G)

            new_population = []
            new_fitness    = []

            # ── Lines 11-22 – Mutation and Crossover loop ────────────────────
            # Iterate over every level, then every individual within that level.
            # Each individual uses the k_t value assigned to its own level.
            for level_idx in range(self.NL):          # 0-indexed level
                level_k = self.k_t[level_idx]         # 1-indexed learning level

                for j in range(self.LS):
                    global_idx = level_idx * self.LS + j
                    target = population[global_idx]

                    # Line 12 – Generate adaptive CR and F
                    CR_i = self.generate_CR()
                    F_i  = self.generate_F()

                    # Lines 14-16 – Diversity-aware exemplar selection
                    e1, e2 = self.select_diverse_exemplars(population, level_k)

                    # Lines 17-18 – DP-LLDE mutation (Eq. 10)
                    mutant = self.mutate(target, e1, e2, F_i, level_k)

                    # Line 19 – Binomial crossover → trial vector
                    trial = self.crossover(target, mutant, CR_i)
                    trial = self.bound_constraint(trial)

                    # Lines 19-20 – Evaluate trial
                    trial_fitness = self.objective_func(trial)
                    FES += 1

                    # Line 21 – Greedy selection
                    if trial_fitness <= fitness[global_idx]:
                        new_population.append(trial)
                        new_fitness.append(trial_fitness)

                        # Archive successful parameters
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

            # Update population arrays
            population = np.array(new_population)
            fitness    = np.array(new_fitness)

            # Line 23 – Update μ_CR and μ_F
            self.update_parameters()

            # Track global best
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < self.best_fitness:
                self.best_fitness = float(fitness[current_best_idx])
                self.best_solution = population[current_best_idx].copy()

            self.fitness_history.append(self.best_fitness)

            if verbose and G % 10 == 0:
                print(f"Generation {G:4d}: Best = {self.best_fitness:.6e}  "
                      f"FES = {FES:6d}  μ_F = {self.mu_F:.3f}  μ_CR = {self.mu_CR:.3f}")

        if verbose:
            print(f"\nOptimization complete.")
            print(f"Best Fitness             = {self.best_fitness:.6e}")
            print(f"Total Function Evals     = {FES}")
            print(f"Total Successful Updates = {self.S_count}")

        return self.best_solution, self.best_fitness, self.fitness_history


# =============================================================================
# Multiple-trial runner (mirrors LBLDE's run_multiple_trials)
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
    func    : Callable — objective function
    bounds  : np.ndarray, shape (D, 2)
    D       : int — problem dimensionality
    n_runs  : int — number of independent runs (paper uses 51)
    max_fes : int — budget; defaults to 10 000 × D
    verbose : bool — print each run

    Returns
    -------
    dict with keys: mean, std, median, min, max, all_best, all_histories
    """
    if max_fes is None:
        max_fes = 10000 * D

    NP = 100 if D <= 50 else 160

    best_fitness_values: List[float] = []
    all_histories: List[List[float]] = []

    print(f"Running {n_runs} independent trials  (NP={NP}, D={D}, MaxFES={max_fes})...")

    for run in range(n_runs):
        if verbose or (run + 1) % 10 == 0:
            print(f"  Run {run + 1}/{n_runs}")

        optimizer = DPLLDE(
            objective_func=func,
            bounds=bounds,
            NP=NP,
            NL=4,
            mu_CR_ini=0.5,
            max_fes=max_fes,
            seed=run
        )

        _, best_fit, history = optimizer.optimize(verbose=False)
        best_fitness_values.append(best_fit)
        all_histories.append(history)

    arr = np.array(best_fitness_values)
    results = {
        'mean'        : float(np.mean(arr)),
        'std'         : float(np.std(arr)),
        'median'      : float(np.median(arr)),
        'min'         : float(np.min(arr)),
        'max'         : float(np.max(arr)),
        'all_best'    : arr,
        'all_histories': all_histories
    }

    print(f"\nResults Summary:")
    print(f"  Mean ± Std : {results['mean']:.6e} ± {results['std']:.6e}")
    print(f"  Median     : {results['median']:.6e}")
    print(f"  Min        : {results['min']:.6e}")
    print(f"  Max        : {results['max']:.6e}")

    return results


# =============================================================================
# Classical benchmark functions (shared with LBLDE for fair comparison)
# =============================================================================

def sphere(x):
    """Sphere — unimodal, global optimum = 0."""
    return float(np.sum(x ** 2))

def rastrigin(x):
    """Rastrigin — multimodal, global optimum = 0."""
    n = len(x)
    return float(10 * n + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x)))

def rosenbrock(x):
    """Rosenbrock — unimodal valley, global optimum = 0."""
    return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))

def ackley(x):
    """Ackley — multimodal, global optimum = 0."""
    n = len(x)
    return float(
        -20 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / n))
        - np.exp(np.sum(np.cos(2 * np.pi * x)) / n)
        + 20 + np.e
    )

def griewank(x):
    """Griewank — multimodal, global optimum = 0."""
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
            NP=100, NL=4,
            mu_CR_ini=0.5,
            max_fes=10000 * D,
            seed=42
        )
        _, best_fit, _ = opt.optimize(verbose=False)
        print(f"  {name:<12}: {best_fit:.6e}")

    print("\nTo use with CEC 2017:")
    print("  from opfunu.cec_based.cec2017 import F12017")
    print("  f1  = F12017(ndim=10)")
    print("  opt = DPLLDE(objective_func=f1.evaluate, bounds=f1.bounds, ...)")
    print("=" * 70)
