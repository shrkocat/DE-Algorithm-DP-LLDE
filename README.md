<div align="center">

# DP-LLDE on CEC 2017 Benchmark Functions
### Mitigating Premature Convergence with Diversity-Preserving Level-Guided Differential Evolution with Adaptive Mutation

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white)
![CEC2017](https://img.shields.io/badge/CEC2017-Duncan%20Tilley-4B8BBE?style=for-the-badge)
![Google Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=black)
![Differential Evolution](https://img.shields.io/badge/DP--LLDE-Custom%20Algorithm-6A0DAD?style=for-the-badge)

</div>

---

# DP-LLDE: Diversity-Preserving Level-Guided Differential Evolution with Adaptive Mutation

> A thesis project by Aaron Kent A. Celles, Lee Ryan A. Leviste, and Enrique S. Santeco  
> Mapua University — BS Computer Science, February 2026

## Overview

**DP-LLDE** (Diversity-Preserving Level-Guided Learning Differential Evolution) is a hybrid evolutionary algorithm designed to mitigate premature convergence in large-scale global optimization (LSGO). It extends the Level-Based Learning Differential Evolution (**LBLDE**) framework by Qiao et al. (2022) with two key innovations: **sequential level assignment** and **diversity-aware exemplar selection**, directly addressing the population diversity degradation identified as a limitation of LBLDE.

The algorithm is benchmarked against LBLDE on the **CEC 2017 Large-Scale Global Optimization Benchmark** across dimensions D = 10, 30, 50, and 100.

---

## The Problem

Standard DE and LBLDE suffer from **premature convergence** in high-dimensional spaces — as particles cluster around elite exemplars, population diversity collapses, limiting the algorithm's ability to escape local optima. DP-LLDE tackles this by:

- Preventing particles from immediately learning from the elite level
- Scaling mutation pressure based on hierarchical level
- Replacing over-similar exemplars with more spatially diverse candidates

---

## DP-LLDE Mutation Formula

The core mutation rule of DP-LLDE is:

$$v_i = e_1 + F \cdot (e_2 - x_i) \cdot \frac{L - k}{L}$$

Where:
- $e_1, e_2$ — two exemplars randomly selected from the **current learning level** $k$
- $x_i$ — the target particle
- $F$ — scaling factor controlling mutation strength
- $L$ — total number of levels
- $k$ — the level the particle is currently learning from
- $\frac{L-k}{L}$ — normalization factor that **decays mutation pressure** as the particle ascends toward elite levels

**Sequential Level Assignment** ensures no particle starts at the elite level:

$$k_0 \sim \{2, 3, \ldots, L\}$$

And level advancement follows a strict upward rule per generation $t$:

$$k_t = k_0 + t, \quad \text{stops when } k_t > L$$

This guarantees that particles explore intermediate fitness regions before converging toward elite solutions, preserving population diversity in early-to-mid optimization stages.

---

## Pseudocode

### LBLDE (Base Algorithm)
```
Algorithm 1: LBLDE
Input: NP, NL, LS, MaxFES
Output: Optimal solution X and fitness f(X)

1.  Set FES=0, G=0, μ_F=0.5, μ_CR=0.35, c=0.1
2.  Initialize random population P; evaluate all X in P
3.  FES = FES + NP
4.  while FES ≤ MaxFES do
5.    G = G + 1
6.    Sort P by fitness (ascending); partition into NL equal levels
7.    for i = 1 to NL do
8.      for j = 1 to LS do
9.        Generate CR_ij ~ N(μ_CR, 0.1), F_ij ~ rand(μ_F, 0.1)
10.       if i == 1: p_i = 0.05
11.       else: p_i = (i-1) × LS × 100% / NP
12.       Select exemplar X_best from top p_i individuals
13.       Select X_r1, X_r2 from top p_{i-1} individuals
14.       V_ij = X_ij + F_ij(X_best − X_ij) + F_ij(X_r1 − X_r2)  ← mutation
15.       Generate trial vector U_ij via binomial crossover
16.       FES = FES + 1
17.       if f(X_ij) ≤ f(U_ij): keep X_ij
18.       else if i ≠ NL: X_ij ← U_ij; store CR, F to success sets
19.       else: X_ij ← U_ij; store F to success set
20.   μ_CR = (1-c)μ_CR + c · mean_A(S_CR)
21.   μ_F  = (1-c)μ_F  + c · mean_L(S_F)
```

---

### DP-LLDE (Proposed Algorithm)
```
Algorithm 2: DP-LLDE
Input: NP, NL, LS, MaxFES
Output: Optimal solution X and fitness f(X)

1.  Set FES=0, G=0, μ_F=0.5, μ_CR=0.35, c=0.1
2.  Initialize random population P; evaluate all X in P
3.  FES = FES + NP
4.  Compute initial diversity D^(0); initialize k_0[] per particle
5.  while FES ≤ MaxFES do
6.    G = G + 1
7.    Sort P by fitness; partition into NL equal levels
8.    for i = 1 to NL do
9.      if G == 1: k_0[i] = rand_int(2, NL)     ← random initial level (not elite)
10.     k_t[i] = max(1, k_0[i] − (G − 1))       ← sequential upward progression
11.     for j = 1 to LS do
12.       Generate CR_ij ~ N(μ_CR, 0.1), F_ij ~ rand(μ_F, 0.1)
13.       if i == 1: p_i = 0.05
14.       else: p_i = (i-1) × LS × 100% / NP
15.       Select best exemplar e_best from top p_i individuals
16.       k = k_t[i]
17.       Select 2 exemplars e1, e2 from level k
18.       if ||e1 − e2|| < threshold: replace one with a more diverse candidate
19.       V_ij = e1 + F_ij · (e2 − X_ij) · (NL − k) / NL   ← DP mutation
20.       Generate trial vector U_ij via binomial crossover
21.       FES = FES + 1
22.       if f(X_ij) ≤ f(U_ij): keep X_ij
23.       else if i ≠ NL: X_ij ← U_ij; store CR, F to success sets
24.       else: X_ij ← U_ij; store F to success set
25.   μ_CR = (1-c)μ_CR + c · mean_A(S_CR)
26.   μ_F  = (1-c)μ_F  + c · mean_L(S_F)
```

---

## Key Differences: LBLDE vs DP-LLDE

| Feature | LBLDE | DP-LLDE |
|---|---|---|
| Exemplar source | Always from elite level | Sequential level traversal |
| Mutation scaling | Fixed | Normalized by level: `(L−k)/L` |
| Diversity check | None | Euclidean distance threshold |
| Initial level | Directly at elite | Random non-elite level |
| Premature convergence risk | Higher | Mitigated |

---

## Benchmark & Evaluation

- **Suite:** CEC 2017 Large-Scale Global Optimization (F1–F30)
- **Dimensions:** D = 10, 30, 50, 100
- **Metrics:** Population Diversity Curves, Error Values

---

## References

- Qiao et al. (2022). Differential Evolution with Level-Based Learning Mechanism. *CSMS*, 2(1), 35–58.
- Chen et al. (2018). A Level-Based Learning Swarm Optimizer for Large-Scale Optimization. *IEEE TEVC*, 22(4).
- Storn & Price (1997). Differential Evolution. *J. Global Optimization*, 11(4).
- Awad et al. (2017). CEC 2017 Benchmark Suite Definitions. Nanyang Technological University.
