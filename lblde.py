import numpy as np
from typing import Callable, Tuple, Optional, List
import warnings

warnings.filterwarnings('ignore')


class LBLDE:
    """
    Level-Based Learning Differential Evolution (LBLDE)
    
    Reference:
    Qiao, K., Liang, J., Qu, B., Yu, K., Yue, C., & Song, H. (2022).
    Differential Evolution with Level-Based Learning Mechanism.
    Complex System Modeling and Simulation, 2(1), 35-58.
    """
    
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
        max_fes: int = 10000,
        seed: Optional[int] = None
    ):
        """
        Initialize LBLDE optimizer
        
        Parameters:
        -----------
        objective_func : Callable
            Objective function to minimize
        bounds : np.ndarray
            Bounds for each dimension, shape (D, 2) where D is dimension
        NP : int
            Population size
        NL : int
            Number of levels
        NLB : int
            Number of bottom levels where CR = 1
        mu_CR_ini : float
            Initial mean value for CR
        mu_F : float
            Initial mean value for F
        c : float
            Learning rate for parameter adaptation
        max_fes : int
            Maximum number of function evaluations
        seed : int, optional
            Random seed for reproducibility
        """
        self.objective_func = objective_func
        self.bounds = np.array(bounds)
        self.D = len(bounds)  # Dimension
        self.NP = NP
        self.NL = NL
        self.NLB = NLB
        self.LS = NP // NL  # Individuals per level
        self.mu_CR = mu_CR_ini
        self.mu_F = mu_F
        self.c = c
        self.max_fes = max_fes
        
        if seed is not None:
            np.random.seed(seed)
        
        # Storage for successful parameters
        self.S_CR = []
        self.S_F = []
        
        # Best solution tracking
        self.best_solution = None
        self.best_fitness = np.inf
        self.fitness_history = []
        
    def initialize_population(self) -> Tuple[np.ndarray, np.ndarray]:
        """Initialize random population within bounds"""
        population = np.random.uniform(
            self.bounds[:, 0],
            self.bounds[:, 1],
            size=(self.NP, self.D)
        )
        fitness = np.array([self.objective_func(ind) for ind in population])
        return population, fitness
    
    def sort_population(self, population: np.ndarray, fitness: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Sort population by fitness (ascending order)"""
        sorted_indices = np.argsort(fitness)
        return population[sorted_indices], fitness[sorted_indices]
    
    def partition_into_levels(self, population: np.ndarray) -> list:
        """Partition sorted population into NL levels"""
        levels = []
        for i in range(self.NL):
            start_idx = i * self.LS
            end_idx = (i + 1) * self.LS
            levels.append(population[start_idx:end_idx])
        return levels
    
    def generate_CR(self, level_idx: int) -> float:
        """Generate CR value for an individual"""
        # Bottom NLB levels have CR = 1
        if level_idx >= self.NL - self.NLB:
            return 1.0
        
        # Generate from normal distribution
        CR = np.random.normal(self.mu_CR, 0.1)
        
        # Truncate to [0, 1]
        if CR > 1:
            CR = 1.0
        elif CR < 0:
            # Regenerate if less than 0
            while CR < 0:
                CR = np.random.normal(self.mu_CR, 0.1)
            if CR > 1:
                CR = 1.0
        
        return CR
    
    def generate_F(self) -> float:
        """Generate F value from Cauchy distribution"""
        F = np.random.standard_cauchy() * 0.1 + self.mu_F
        
        # Truncate
        if F > 1:
            F = 1.0
        
        # Regenerate if F <= 0
        while F <= 0:
            F = np.random.standard_cauchy() * 0.1 + self.mu_F
            if F > 1:
                F = 1.0
        
        return F
    
    def calculate_pi(self, level_idx: int) -> float:
        """Calculate pi for level i (proportion of top individuals to select from)"""
        if level_idx == 0:
            return 0.05
        else:
            return (level_idx * self.LS) / self.NP
    
    def select_exemplar(self, population: np.ndarray, pi: float) -> np.ndarray:
        """Select exemplar from top pi individuals"""
        top_n = max(1, int(pi * self.NP))
        idx = np.random.randint(0, top_n)
        return population[idx]
    
    def select_difference_vectors(
        self, 
        population: np.ndarray, 
        level_idx: int,
        current_idx_in_level: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Select two individuals for difference vector"""
        if level_idx == 0:
            # First level: select from first level
            available_indices = list(range(self.LS))
            available_indices.remove(current_idx_in_level)
            selected = np.random.choice(available_indices, size=2, replace=False)
            return population[selected[0]], population[selected[1]]
        else:
            # Other levels: Xr1 from L1 to L(i-1), Xr2 from L1 to Li
            pi_minus_1 = ((level_idx - 1) * self.LS) / self.NP
            pi = (level_idx * self.LS) / self.NP
            
            top_n_minus_1 = max(1, int(pi_minus_1 * self.NP))
            top_n = max(1, int(pi * self.NP))
            
            idx1 = np.random.randint(0, top_n_minus_1)
            idx2 = np.random.randint(0, top_n)
            
            return population[idx1], population[idx2]
    
    def mutate(
        self,
        target: np.ndarray,
        exemplar: np.ndarray,
        r1: np.ndarray,
        r2: np.ndarray,
        F: float
    ) -> np.ndarray:
        """DE/current-to-pbest/1 mutation"""
        mutant = target + F * (exemplar - target) + F * (r1 - r2)
        return mutant
    
    def crossover(self, target: np.ndarray, mutant: np.ndarray, CR: float) -> np.ndarray:
        """Binomial crossover"""
        trial = np.copy(target)
        j_rand = np.random.randint(0, self.D)
        
        for j in range(self.D):
            if np.random.rand() < CR or j == j_rand:
                trial[j] = mutant[j]
        
        return trial
    
    def bound_constraint(self, individual: np.ndarray) -> np.ndarray:
        """Apply boundary constraints"""
        return np.clip(individual, self.bounds[:, 0], self.bounds[:, 1])
    
    def update_parameters(self):
        """Update mu_CR and mu_F using successful parameters"""
        if len(self.S_CR) > 0:
            mean_A = np.mean(self.S_CR)
            self.mu_CR = (1 - self.c) * self.mu_CR + self.c * mean_A
        
        if len(self.S_F) > 0:
            # Lehmer mean
            S_F_array = np.array(self.S_F)
            mean_L = np.sum(S_F_array ** 2) / np.sum(S_F_array)
            self.mu_F = (1 - self.c) * self.mu_F + self.c * mean_L
    
    def optimize(self, verbose: bool = True) -> Tuple[np.ndarray, float, list]:
        """
        Run LBLDE optimization
        
        Returns:
        --------
        best_solution : np.ndarray
            Best solution found
        best_fitness : float
            Best fitness value
        fitness_history : list
            History of best fitness over generations
        """
        # Initialize
        FES = 0
        G = 0
        
        # Create initial population
        population, fitness = self.initialize_population()
        FES += self.NP
        
        # Track best
        best_idx = np.argmin(fitness)
        self.best_solution = population[best_idx].copy()
        self.best_fitness = fitness[best_idx]
        self.fitness_history.append(self.best_fitness)
        
        if verbose:
            print(f"Generation 0: Best Fitness = {self.best_fitness:.6e}")
        
        # Main loop
        while FES < self.max_fes:
            G += 1
            self.S_CR = []
            self.S_F = []
            
            # Sort population and partition into levels
            population, fitness = self.sort_population(population, fitness)
            levels = self.partition_into_levels(population)
            
            # Process each level
            new_population = []
            new_fitness = []
            
            for i in range(self.NL):
                level = levels[i]
                pi = self.calculate_pi(i)
                
                for j in range(self.LS):
                    # Generate parameters
                    CR_ij = self.generate_CR(i)
                    F_ij = self.generate_F()
                    
                    # Get target individual
                    target_idx = i * self.LS + j
                    target = population[target_idx]
                    
                    # Select exemplar
                    exemplar = self.select_exemplar(population, pi)
                    
                    # Select difference vectors
                    r1, r2 = self.select_difference_vectors(population, i, j)
                    
                    # Mutation
                    mutant = self.mutate(target, exemplar, r1, r2, F_ij)
                    
                    # Crossover
                    trial = self.crossover(target, mutant, CR_ij)
                    
                    # Boundary constraint
                    trial = self.bound_constraint(trial)
                    
                    # Evaluation
                    trial_fitness = self.objective_func(trial)
                    FES += 1
                    
                    # Selection
                    if trial_fitness <= fitness[target_idx]:
                        new_population.append(trial)
                        new_fitness.append(trial_fitness)
                        
                        # Store successful parameters (not for bottom level CR)
                        if i != self.NL - 1:
                            self.S_CR.append(CR_ij)
                        self.S_F.append(F_ij)
                    else:
                        new_population.append(target)
                        new_fitness.append(fitness[target_idx])
                    
                    # Check if max FES reached
                    if FES >= self.max_fes:
                        break
                
                if FES >= self.max_fes:
                    break
            
            # Update population
            population = np.array(new_population)
            fitness = np.array(new_fitness)
            
            # Update parameters
            self.update_parameters()
            
            # Track best
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < self.best_fitness:
                self.best_fitness = fitness[current_best_idx]
                self.best_solution = population[current_best_idx].copy()
            
            self.fitness_history.append(self.best_fitness)
            
            if verbose and G % 10 == 0:
                print(f"Generation {G}: Best Fitness = {self.best_fitness:.6e}, FES = {FES}")
        
        if verbose:
            print(f"\nOptimization completed!")
            print(f"Final Best Fitness = {self.best_fitness:.6e}")
            print(f"Total Function Evaluations = {FES}")
        
        return self.best_solution, self.best_fitness, self.fitness_history


# =============================================================================
# CEC 2017 BENCHMARK TESTING UTILITIES
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
    Run LBLDE multiple times (as in the paper: 51 runs)
    
    Parameters:
    -----------
    func : Callable
        Objective function
    bounds : np.ndarray
        Search bounds
    D : int
        Dimension
    n_runs : int
        Number of independent runs (default: 51 as in paper)
    max_fes : int
        Maximum function evaluations (default: 10000*D)
    verbose : bool
        Print progress
        
    Returns:
    --------
    results : dict
        Dictionary containing statistical results
    """
    if max_fes is None:
        max_fes = 10000 * D
    
    # Determine population size based on dimension (as in paper)
    if D <= 50:
        NP = 100
    else:
        NP = 160
    
    best_fitness_values = []
    all_histories = []
    
    print(f"Running {n_runs} independent trials...")
    for run in range(n_runs):
        if verbose or (run + 1) % 10 == 0:
            print(f"  Run {run + 1}/{n_runs}")
        
        optimizer = LBLDE(
            objective_func=func,
            bounds=bounds,
            NP=NP,
            NL=4,
            NLB=1,
            mu_CR_ini=0.35,
            max_fes=max_fes,
            seed=run  # Different seed for each run
        )
        
        best_sol, best_fit, history = optimizer.optimize(verbose=False)
        best_fitness_values.append(best_fit)
        all_histories.append(history)
    
    best_fitness_values = np.array(best_fitness_values)
    
    results = {
        'mean': np.mean(best_fitness_values),
        'std': np.std(best_fitness_values),
        'median': np.median(best_fitness_values),
        'min': np.min(best_fitness_values),
        'max': np.max(best_fitness_values),
        'all_best': best_fitness_values,
        'all_histories': all_histories
    }
    
    print(f"\nResults Summary:")
    print(f"  Mean ± Std: {results['mean']:.6e} ± {results['std']:.6e}")
    print(f"  Median: {results['median']:.6e}")
    print(f"  Min: {results['min']:.6e}")
    print(f"  Max: {results['max']:.6e}")
    
    return results


# =============================================================================
# EXAMPLE TEST FUNCTIONS (Classical Benchmarks)
# =============================================================================

def sphere(x):
    """Sphere function (unimodal)"""
    return np.sum(x**2)

def rastrigin(x):
    """Rastrigin function (multimodal)"""
    n = len(x)
    return 10*n + np.sum(x**2 - 10*np.cos(2*np.pi*x))

def rosenbrock(x):
    """Rosenbrock function"""
    return np.sum(100*(x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def ackley(x):
    """Ackley function (multimodal)"""
    n = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(2*np.pi*x))
    return -20*np.exp(-0.2*np.sqrt(sum1/n)) - np.exp(sum2/n) + 20 + np.e

def griewank(x):
    """Griewank function (multimodal)"""
    sum_sq = np.sum(x**2) / 4000
    prod_cos = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x)+1))))
    return sum_sq - prod_cos + 1


# =============================================================================
# MAIN TESTING SCRIPT
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("LBLDE Algorithm - Comprehensive Testing")
    print("=" * 70)
    
    # Test dimensions
    D = 10
    
    # Define test functions
    test_functions = [
        ('Sphere', sphere, [-100, 100]),
        ('Rastrigin', rastrigin, [-5.12, 5.12]),
        ('Rosenbrock', rosenbrock, [-30, 30]),
        ('Ackley', ackley, [-32, 32]),
        ('Griewank', griewank, [-600, 600])
    ]
    
    print(f"\nTesting on {D}D problems")
    print("=" * 70)
    
    for func_name, func, bound_range in test_functions:
        print(f"\n{'='*70}")
        print(f"Function: {func_name}")
        print(f"{'='*70}")
        
        bounds = np.array([bound_range] * D)
        
        # Single run for demonstration
        print("\nSingle run:")
        optimizer = LBLDE(
            objective_func=func,
            bounds=bounds,
            NP=100,
            NL=4,
            NLB=1,
            mu_CR_ini=0.35,
            max_fes=10000 * D,
            seed=42
        )
        
        best_sol, best_fit, history = optimizer.optimize(verbose=False)
        print(f"Best fitness: {best_fit:.6e}")
        
    print("\n" + "=" * 70)
    print("Testing Complete!")
    print("=" * 70)
    print("\nTo use CEC 2017 benchmark:")
    print("1. Install: pip install opfunu")
    print("   or use: https://github.com/tilleyd/cec2017-py")
    print("\n2. Example usage:")
    print("   from opfunu.cec_based.cec2017 import F12017")
    print("   f1 = F12017(ndim=10)")
    print("   optimizer = LBLDE(objective_func=f1.evaluate, bounds=f1.bounds, ...)")
    print("=" * 70)