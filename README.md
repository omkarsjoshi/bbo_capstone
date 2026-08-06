# BBO Capstone — Bayesian Optimisation Pipeline

This repository contains the capstone project for the ICL Professional Certificate in Machine Learning and Artificial Intelligence (PCMLAI) programme: a configuration-driven Bayesian optimisation pipeline that queries eight black-box benchmark functions (F1-F8, ranging from 2 to 8 dimensions) once per week over a ten-round schedule, with the objective of finding each function's maximiser.

## Contents

- **Pipeline notebook:** The pipeline itself, comprising per-function GP surrogates (Matérn kernel, ARD lengthscales), an LOO-CV trust gate, a four-method dimension-importance consensus panel, and four candidate-generation strategies (global acquisition ensemble, TuRBO and multi-basin TuRBO, local-GP "needle" refinement, and space-filling exploration), together with the associated diagnostic plots.
- **Query data** (`initial_data_9/function_<n>/`): Per-function input and output arrays (`initial_inputs.npy`, `initial_outputs.npy`) accumulated over the first nine rounds, used to generate round 10's suggestions.
- [`datasheet.md`](https://github.com/datasheet.md) : Documents the query and evaluation dataset, including its contents, how it was collected, its intended uses, and its known limitations.
- [`model_card.md`](https://github.com/chat/model_card.md) : Documents the pipeline itself, including surrogate design, routing logic, current strategy by component, performance by function, and known assumptions and limitations.

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

### Details of the strategy in the latest round

A full week-by-week account of how the pipeline reached its current form, including approaches that were tried and later discarded (an SVM classifier, neural-network sensitivity analysis, PyTorch-based gradient ascent, and an XGBoost feature-importance model), is provided in the model_card. The summary below describes the current design of each component and the rationale behind it, rather than the full development history.

**Kernel.** A Matérn kernel is used, with ν=1.5 for functions exhibiting sharper local structure (F1, F3, F4, F6, F7) and ν=2.5 for smoother functions, together with automatic relevance determination (ARD) lengthscales and an additive `WhiteKernel` noise term (bounds 0.05–10,000) to prevent the GP from fitting noise with unrealistically short lengthscales. The Matérn kernel replaced an initial RBF kernel, which assumes infinite differentiability and was found to be a poor fit for functions with local, spiky behaviour such as F1 and F2.

**Output warping.** A Yeo-Johnson power transform is applied to `y` prior to fitting for six of the eight functions (F3–F8). Predictions are inverted back to the raw output scale, with numerical safeguarding near the transform's singularity. This safeguard was introduced after an inversion produced a NaN for LOO-CV R² for F7, which had silently affected the function's routing decision (see the README for the diagnosis).

**Acquisition, global branch.** An ensemble of Upper Confidence Bound, Expected Improvement, and Probability of Improvement is used, with `xi` scaled to each function's output range and `beta` scaled following Srinivas et al. (2009), rather than using fixed constants, since a single fixed value is not appropriate given the wide variation in output scale across the eight functions. Candidate points are optimised using differential evolution with an L-BFGS-B local polish, which replaced an earlier random-sampling approach once higher-dimensional functions required a more targeted search.

**Trust gate.** A function is routed to the global acquisition ensemble described above if its LOO-CV R² is at least 0.30; otherwise it is routed to TuRBO. This rule replaced an earlier manual classification of functions (as "blind", "stalled", or "TuRBO") with a systematic criterion re-evaluated every round.

**Local search (TuRBO).** The trust region expands or contracts based on consecutive success and failure streaks, with its size determined by the GP's own lengthscales. A diagnostic tool (`TuRBOTracer`) tracks the trust-region radius across iterations so that a collapsing region can be identified explicitly rather than inferred after the fact. F2 uses a multi-basin variant that alternates the trust region across several well-separated high-value regions rather than anchoring on a single point.

**Dimension importance.** A consensus panel comprising ARD lengthscales, GP posterior-gradient sensitivity, random-forest permutation importance, and SHAP is used to identify dimensions that can be frozen. This panel previously included neural-network sensitivity as a fifth method; it was removed once its scores were found to correlate closely with the GP and random-forest methods while being the least reliable of the group given the small number of available points per function.

**Reproducibility.** Every GP instance uses a fixed random seed (`random_state=42`), with separate seeded sub-streams per function and per week. This was introduced after identifying that unseeded GP restarts produced inconsistent LOO-CV R² values across repeated runs of identical code, most noticeably for F1 and F2.

## Notes on transparency

Known limitations are documented alongside the results wherever identified, rather than omitted, including the difference between fast-development-mode and full-report-mode LOO-CV figures, and the evidence for F2's likely stochasticity. Further detail is provided in the datasheet and model card.
