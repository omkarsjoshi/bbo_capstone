# BBO Capstone: Bayesian Optimisation Pipeline

This repository contains the capstone project for the ICL Professional Certificate in Machine Learning and Artificial Intelligence (PCMLAI) programme: a configuration-driven Bayesian optimisation pipeline that queries eight black-box benchmark functions (F1-F8, ranging from 2 to 8 dimensions) once per week over a thirteen-round schedule, with the objective of finding each function's maximiser.

## Contents
- **[Pipeline notebook](https://github.com/omkarsjoshi/bbo_capstone/blob/main/Capstone_W13_Final.ipynb):** Orchestrates the modules below to load the accumulated data, run the LOO-CV trust gate and consensus panel for every function, generate this round's candidate points, and produce the accompanying diagnostic plots and tables.
- The pipeline itself ([`bbo_data.py`](https://github.com/omkarsjoshi/bbo_capstone/blob/main/bbo_data.py), [`bbo_pipeline.py`](https://github.com/omkarsjoshi/bbo_capstone/blob/main/bbo_pipeline.py), [`bbo_diagnostics.py`](https://github.com/omkarsjoshi/bbo_capstone/blob/main/bbo_diagnostics.py)), comprising per-function GP surrogates (Matérn kernel, ARD lengthscales), an LOO-CV trust gate, a four-method dimension-importance consensus panel, and four candidate-generation strategies (global acquisition ensemble, TuRBO and multi-basin TuRBO, local-GP "needle" refinement, and space-filling exploration), together with the associated diagnostic plots.
- **Query data** (`initial_data/function_<n>/`): Per-function input and output arrays (`initial_inputs.npy`, `initial_outputs.npy`) from the initial dataset.
- **Round log** ([`inputs_W12.txt`](https://github.com/omkarsjoshi/bbo_capstone/blob/main/inputs_W12.txt), [`outputs_W12.txt`](https://github.com/omkarsjoshi/bbo_capstone/blob/main/outputs_W12.txt)): Cumulative, per-round record of every input queried and output received from round 1 through round 12 (one row per function per round), appended by `bbo_data.py` onto the initial sample. Together with the query data above, this reconstructs each function's full accumulated dataset, from which round 13's final suggestions are generated.
- [`datasheet.md`](https://github.com/omkarsjoshi/bbo_capstone/blob/main/datasheet.md): Documents the query and evaluation dataset, including its contents, how it was collected, its intended uses, and its known limitations.
- [`model_card.md`](https://github.com/omkarsjoshi/bbo_capstone/blob/main/model_card.md): Documents the pipeline itself, including surrogate design, routing logic, current strategy by component, performance by function, and known assumptions and limitations.

## Inputs and outputs

### Inputs

Each black-box function $f_i : [0,1]^{d_i} \rightarrow \mathbb{R}$ accepts a normalised continuous input vector:

$$\mathbf{x} = [x_1, x_2, \ldots, x_{d_i}] \quad \text{where } x_j \in [0, 1] \ \forall j$$

The dimensionality $d_i$ varies across functions (F1–F8 range from 2D to 8D). All inputs are bounded within the unit hypercube — no additional domain constraints are imposed beyond this.

**Example query format:**

```python
x_query = np.array([0.42, 0.17, 0.88])  # For a 3D function such as F3
y_response = black_box_function(x_query)
```

### Outputs

Each query returns a single scalar response:

$$y = f_i(\mathbf{x}) + \varepsilon$$

where $\varepsilon$ represents any observational noise. The output is a real-valued performance signal with no guaranteed bounds; in practice, response magnitudes varied substantially across functions, e.g. F1 and F5 exhibited extreme outlier values orders of magnitude above the bulk of observations.

### Internal model outputs

At each round, the fitted GP surrogate additionally yields:

- **Posterior mean:** $\mu(\mathbf{x})$ — predicted output at any candidate point
- **Posterior variance:** $\sigma^2(\mathbf{x})$ — model uncertainty at that point
- **Acquisition function value:** $\alpha(\mathbf{x})$ — composite score driving next query selection

### Primary goal

The objective is to **maximise** each black-box function $f_i$ within the allowed query budget. The challenge is effectively:

$$\mathbf{x}^* = \arg\max_{\mathbf{x} \in [0,1]^{d_i}} f_i(\mathbf{x})$$

with the constraint that $f_i$ can only be evaluated a limited number of times across weekly rounds.

### Key constraints and limitations

| Constraint                     | Description                                                                                                       |
| ------------------------------ | ----------------------------------------------------------------------------------------------------------------- |
| **Budget**                     | Fixed number of queries per round; total budget finite across all weeks                                           |
| **Unknown structure**          | No gradient, no analytical form, no domain knowledge about $f_i$                                                  |
| **Varied dimensionality**      | Functions range from 2D to 8D, requiring dimensionality-aware strategies                                          |
| **Extreme response variation** | Some functions (F1, F5) exhibit highly spiky behaviour (near-flat with isolated peaks), while others are smoother |
| **No ground truth**            | Cannot verify whether a returned maximum is global or local; model uncertainty estimates are the only guide       |

### Non-technical summary
This project set out to find the best possible settings for eight unknown mathematical functions, using only a small number of test queries per function, similar to tuning an unfamiliar machine by trial and error under a tight budget. A predictive model learned from each result and decided how much to trust its own predictions before choosing whether to search broadly, refine locally, or hedge across multiple promising regions. This trust was checked periodically for every query round. Every function showed improvement compared to the initial dataset, and a few improved dramatically, including one that jumped 660-fold from a single late discovery. The key lesson was that models can appear correct while quietly being incorrect, and only periodic audits can diagnose this.


### Details of the strategy in the latest round

A full week-by-week account of how the pipeline reached its current form, including approaches that were tried and later discarded (an SVM classifier, neural-network sensitivity analysis, PyTorch-based gradient ascent, and an XGBoost feature-importance model), is provided in the model_card. The summary below describes the current design of each component and the rationale behind it, rather than the full development history.

**Kernel.** A Matérn kernel is used, with ν=1.5 for functions exhibiting sharper local structure (F1, F3, F4, F6, F7) and ν=2.5 for smoother functions (F2, F5, F8), together with automatic relevance determination (ARD) lengthscales and an additive `WhiteKernel` noise term (bounds 0.05–10,000) to prevent the GP from fitting noise with unrealistically short lengthscales. The Matérn kernel replaced an initial RBF kernel, which assumes infinite differentiability and was a poor fit for functions with local, spiky behaviour such as F1 and F2.

**Output warping.** A monotonic warp is applied to `y` prior to fitting for four of the eight functions, selected on a per-function basis via `compare_warps` rather than applied uniformly: F2 uses a log transform, F1, F3 and F8 use Yeo-Johnson; and all others are fitted directly on the raw output scale, having shown no benefit from warping. Predictions are inverted back to the raw output scale, with numerical safeguarding near the transform's singularity. This safeguard was introduced after an inversion produced a NaN for LOO-CV R² for F7, which had silently affected the function's routing decision.

**Acquisition, global branch.** An ensemble of Upper Confidence Bound, Expected Improvement, and Probability of Improvement is used, with `xi` scaled to each function's output range and `beta` scaled following Srinivas et al. (2009), rather than using fixed constants, since a single fixed value is not appropriate given the wide variation in output scale across the eight functions. Candidate points are optimised using differential evolution with an L-BFGS-B local polish, which replaced an earlier random-sampling approach once higher-dimensional functions required a more targeted search.

![Climbing animation](/resources/de_climbing.gif)

**Trust gate.** A function is routed to the global acquisition ensemble described above if its leak-free LOO-CV R² is at least 0.30; otherwise it is routed to TuRBO. F1's scatter plot is the visual counterpart of its R² value of -0.047: all but one point lie close to y = 0, and the model predicts a near-zero value for the needle point as well, since it has no basis for inferring a spike from data that otherwise appears uniformly flat. F5 shows the opposite pattern: the two points near the vertex value of 8000 or higher are predicted almost exactly, which raises the overall R² despite comparatively noisier predictions in the 0 to 2000 range.

F4, F5, and F8 show a trustworthy fit with high LOO-CV R² among all functions. This is reflected in their respective improvements over the weekly rounds: F4 and F8 increased gradually, while F5 found the vertex optimum quite rapidly.

![Alternative Text](/resources/r2.png)

**Local search (TuRBO).** The trust region expands or contracts based on consecutive success and failure streaks, with its size determined by the GP's own lengthscales. A diagnostic tool (`TuRBOTracer`) tracks the trust-region radius across iterations so that a collapsing region can be identified explicitly rather than inferred after the fact. F1 uses a multi-basin variant that alternates the trust region across several well-separated high-value regions rather than anchoring on a single point.

**Dimension importance.** A consensus panel comprising ARD lengthscales, GP posterior-gradient sensitivity, random-forest permutation importance, and SHAP is used to identify dimensions that can be frozen. Under this panel, F3 has x1 frozen, and F7 has both x1 and x3 frozen. While x3 was frozen because the consensus panel deemed it irrelevant, the incumbent currently has x1 at 0, and based on the GP slice visualisations and the output, it steeply decreases at higher values of x1. Hence, x1 was also frozen at 0. 

This panel previously included neural-network sensitivity as a fifth method; it was removed once its scores were found to correlate closely with the GP and random-forest methods while being the least reliable of the group given the small number of available points per function.

![Alternative Text](/resources/consensus_panel.png)

**Reproducibility.** Every GP instance uses a fixed random seed (`random_state=42`), with separate seeded sub-streams per function and per week. This was introduced after identifying that unseeded GP restarts produced inconsistent LOO-CV R² values across repeated runs of identical code, most noticeably for F1 and F2.


## Final results
The 3D slice for the GP is rendered below, considering the top two important dimensions while holding every other dimension fixed at the incumbent. The slices clearly show how much of the observed scatter the surrogate is (and isn't) explaining. Also, F1's needle-like spike on the otherwise flat landscape is consistent with its negative LOO R²  against F4's clean converged dome.

F1's surface renders as essentially flat across the entire domain, since a stationary kernel cannot represent a single spike. This contrasts with F4's smooth, well-explained surface that peaks in a very narrow region.

![Alternative Text](/resources/gp_mean_surface.png)

The tabulation below illustrates different diagnostic metrics. The before and after columns correspond to two distinct LOO-CV procedures, `loo_cv_leaky_warpedscale` and `loo_cv_leakproof_rawscale`. In the former, the warp is fitted once on the full dataset, so each fold's target transform is partly informed by the point it later holds out. Since duplicate queries were made for F2 and F5 to confirm whether they were stochastic, these rows are treated as independent, allowing a duplicate's counterpart to remain in training while its pair is withheld. Moreover, R² is computed on the warped scale. In the latter, the warp is refitted within each fold using only that fold's training data, near-duplicate rows are withheld together, and predictions are inverted to the raw scale before R² is computed. As these three differences act cumulatively, the two columns should be read as related but non-equivalent quantities, highlighting the importance of an appropriate validation workflow when warping is used.

F3 shows the largest decrease between these metrics, with leak-prone R² of 0.732 vs the leakfree R2 of 0.336. This is expected for a Yeo-Johnson-warped function, since `loo_cv_leaky_warpedscale` is computed entirely in warped space and is therefore never exposed to the tail-amplification effect that the raw-scale `loo_cv_leakproof_rawscale` metric captures. 

| Function | Dim | Strategy Used | LOO-CV R² (leakproof) | LOO-CV R² (leaky, warped) | Trim1 R² (worst fold removed) | Best y  |
|----------|-----|----------------|:----------------------:|:---------------------------:|:-------------------------------:|:---------------------:|
| F1       | 2D  | TuRBO (multi-basin) | **−0.047** | −0.553 | −38.143 | 0.4962 |
| F2       | 2D  | Global (EI/UCB/PI)  | **+0.559** | +0.524 | +0.655  | 0.6548  |
| F3       | 3D  | Global (EI/UCB/PI)  | **+0.336** | +0.732 | +0.740  | −0.0029  |
| F4       | 4D  | "needle"    | **+0.924** | +0.924 | +0.938  | 0.6565 |
| F5       | 4D  | "corners" (forced/manual override) | **+0.933** | +0.934 | +0.973  | 8662.4050 |
| F6       | 5D  | Global (EI/UCB/PI)  | **+0.752** | +0.752 | +0.760  | −0.1692  |
| F7       | 6D  | Global (EI/UCB/PI)  | **+0.842** | +0.842 | +0.881  | 1.9734 |
| F8       | 8D  | Global (EI/UCB/PI)  | **+0.984** | +0.984 | +0.987  | 9.9720 |

The trend below shows how the best value observed for each function evolved over the campaign, excluding the initial sampling phase, so it shows only the sequential decision-making phase. F1 and F5 both show a single large-step improvement rather than gradual progress. F1's spike was discovered around round 10, and F5's vertex was discovered around round 3. F2, F3, F6, F7, and F8 have shown essentially no improvement over several recent rounds, which suggests these campaigns may be approaching their practical ceiling for the current evaluation budget.
![Alternative Text](/resources/progress.png)



## Notes on transparency

Known limitations are documented alongside the results wherever identified, rather than omitted, including the difference between fast-development-mode and full-report-mode LOO-CV figures, and the evidence for F2's likely stochasticity. Further detail is provided in the datasheet and model card.
