# Model Card: BBO Capstone Optimisation Pipeline

## Overview

- **Name**: BBO Capstone Pipeline, Week 10 build.
- **Type**: not a single trained model, but a Bayesian optimisation pipeline. For each function, the pipeline combines a Gaussian Process (GP) surrogate, a rule-based router driven by leave-one-out cross-validation (LOO-CV), and one of four candidate-generation strategies: a global acquisition ensemble, TuRBO (including a multi-basin variant), local-GP "needle" refinement, or space-filling exploration.
- **Version**: the Week 10 configuration-driven build, incorporating the fixes for warp inversion  identified during the Week 9 audit where NaN was produced for LOO-CV  R², saturated ARD lengthscales interpreted as evidence of multimodality rather than irrelevance, multi-basin round-robin TuRBO, and local-GP needle refinement.
- **Developer**: developed by a single individual contributor as part of the ICL PCMLAI capstone.
- **Licensing**: intended for academic use within the ICL PCMLAI programme only; no commercial licence is claimed. The eight benchmark functions F1-F8 are black-box oracles supplied by the course and are not included in the repository; only the queries submitted and the values returned are documented here.

## Intended use

The pipeline's primary task is to propose the next query point, once per week, for each of eight black-box benchmark functions (F1-F8, ranging from 2 to 8 dimensions), with the aim of maximising each function within a fixed 13-query budget. The intended users are the pipeline's author, for the purpose of the capstone assessment, and secondarily course facilitators and peers reviewing the submission.

The intended use case is sample-efficient sequential optimisation under a very small query budget, where each query is costly and must be chosen deliberately rather than sampled at random. This mirrors real-world settings where an experiment is costly (either because it is prohibitively long or significantly expensive), making a model-guided approach preferable to grid search.

The pipeline should not be used on problems structurally unlike F1-F8, such as those with much higher dimensionality, discrete or combinatorial inputs, or domains not naturally bounded on `[0,1]`, without re-tuning. It is not intended as a general-purpose regression model, since the GP surrogates are tuned specifically for the trust-gate and routing role rather than for standalone predictive accuracy. It is also not intended for use in settings requiring a fixed budget where the trust-region state and per-function configuration would need to be reinitialised from scratch.

## Training data

There is no fixed training dataset in the conventional sense. Each function's GP surrogate is refitted every round on that function's full query history to date, comprising the initial seed points supplied by the course plus every point submitted since. Full documentation of this history (including its composition, collection process, and known gaps) is provided in the accompanying datasheet (`datasheet.md`) and is not duplicated here. Because the history grows by one point per function per week, the surrogate evaluated in early rounds and the surrogate evaluated at round 10 are not equivalent models; the performance figures reported below are specific to round 10 rather than a fixed property of the pipeline.

## Details of the strategy evolution

Each change described below was introduced in response to a specific issue identified during that week, rather than as a general increase in sophistication. The summary is condensed from the weekly capstone discussion posts submitted throughout the project.

**Week 1.** A GP surrogate with an RBF kernel was used with three acquisition functions (UCB, PI, EI), with the candidate achieving the highest predicted mean selected each round. Probability of Improvement was found to stall in high-uncertainty regions when `z` is negative, so an Expected Improvement implementation was adopted whose additional `σ·φ(z)` term continues to reward exploration in these regions. F1 (flat, near-zero output with one outlier) and F5 (a smaller outlier) were identified as the two functions with the least clear signal.

**Week 2.** A fixed `xi`/`beta` value applied uniformly across functions was found to be inappropriate given the substantial differences in output scale between functions. `xi` (used in EI/PI) was scaled to each function's output range, and `beta` (used in UCB) was scaled to grow with `log(d·t²)` following Srinivas et al. (2009), rather than remaining constant. Dimension-sweep plots were introduced, in which all but one input dimension is held fixed while the remaining dimension is varied, to provide an interpretable two-dimensional view of a multi-dimensional function.

**Week 3.** The RBF kernel was replaced with a Matérn 5/2 kernel, since RBF's assumption of infinite differentiability was found to be a poor fit for functions exhibiting sharp local structure, such as F1 and F2. The Matérn kernel, being only twice differentiable, better represents this structure and grows uncertainty more quickly away from observed points. The UCB `beta` parameter was capped after its scaling began to diverge substantially from EI and PI. Automatic relevance determination (ARD) lengthscales were introduced as the first per-dimension importance signal, and the number of acquisition candidates sampled was increased from 2,000 to as many as 20,000 for higher-dimensional functions.

**Week 4.** An SVM classifier (using a 75th-percentile threshold) was introduced to identify promising regions per dimension, alongside a neural network (implemented in Keras) used to estimate input sensitivity via GradientTape. Acquisition optimisation was changed from random sampling to differential evolution with an L-BFGS-B local polish. GP-ARD and neural-network sensitivity scores were found to agree closely for most functions, but disagreed substantially for F4 (r = −0.96), which was interpreted as evidence that the two methods capture different aspects of the underlying structure rather than one method being incorrect.

**Week 5.** Dimension importance was formalised as a consensus of four methods: ARD lengthscales, neural-network sensitivity, random-forest feature importance, and SHAP. Leave-one-out cross-validation R² was adopted as the primary surrogate-trust diagnostic; this revealed negative R² values for F1, F2, and F3 (indicating the surrogate was fitting noise rather than signal) against values above 0.9 for F4 and F8. Planning began for a repository restructuring, moving from a single monolithic notebook toward a structure comprising a README, a strategy log (`strategy_log.md`/`DECISIONS.md`), and a requirements file.

**Week 6.** The documentation structure was finalised, comprising a README, a design-decision log (`DECISIONS.md`), and a references file (`REFERENCES.md`). The libraries used in the pipeline were catalogued, including scikit-learn (GP surrogate), SciPy (differential evolution and L-BFGS-B), SHAP, SVM, PyTorch/TensorFlow, and XGBoost. Several candidate techniques for future weeks were scoped, including TuRBO for high-dimensional local search, BoTorch's batched acquisition functions, the COCO/BBOB benchmarking framework, and PySR symbolic regression; none of these were adopted at this stage.

**Week 7.** A `WhiteKernel` noise term (lengthscale bounds 0.05–10,000) was added to prevent the GP from fitting noise with unrealistically short lengthscales. The earlier manual classification of functions ("blind", "stalled", "TuRBO") was replaced with a systematic rule based on the LOO-CV trust gate: functions with R² ≥ 0.30 were routed to the global GP ensemble, and all others to TuRBO. Neural-network sensitivity was removed from the dimension-importance panel after being found to correlate closely with the GP and random-forest methods, while being the least reliable of the four given the limited number of data points available. A log-warping trial on F5 was reverted after it was found to reduce LOO-CV R².

**Week 8.** As a reflective exercise, the acquisition and kernel choices used in the pipeline were compared to prompting strategies used with large language models (structured versus zero-shot prompting, temperature and top-p sampling, and context-window truncation). The practical outcome of this exercise was the addition of `TuRBOTracer`, a diagnostic that tracks the trust-region radius across iterations, allowing a collapsing trust region — indicating the optimiser has become stuck — to be identified explicitly rather than inferred after the fact.

**Week 9.** Most of the pipeline's current configuration originates from the audit conducted this week, ahead of the final round.

- A reproducibility issue was identified and corrected: GP restarts and TuRBO candidate draws were not passing through a seeded random-number generator, causing inconsistent R² values across repeated runs of identical code, most notably for F1 and F2. Every GP instance was subsequently fixed to `random_state=42`, with separate seeded sub-streams per function and per week.
- F7's undefined (NaN) LOO-CV R² was traced to a Yeo-Johnson warp with a fitted lambda of −2.62, which placed the transform's pole within the range of values the GP was predicting. Clipping the prediction near the pole restored a finite R² of 0.031, allowing F7 to be routed to TuRBO for a substantiated reason rather than due to an undefined value.
- Evidence was found suggesting F2 may be stochastic rather than deterministic: five adjacent points returned outputs differing by approximately 0.033 on average, while a sixth nearby point returned a value more than three standard deviations from that cluster's mean. Resubmitting an identical query confirmed this, with the repeated query differing from the original by more than 0.05. F2's strategy was revised accordingly, holding x1 near its historical peak (approximately 0.7) and widening the range explored on x2, despite x2 being classified as trivial by the dimension-importance panel.
- A sampling bias in TuRBO candidate generation was identified and corrected: approximately 17% of x2 candidate draws were found to be pinned to the lower clipping bound whenever the anchor point fell within two standard deviations of that bound. This was resolved by resampling rather than clipping.
- A `REPORT_MODE` flag was added to distinguish between a fast development mode, which fits the GP kernel once and reuses it across all LOO-CV folds (approximately 3 minutes runtime), and a full mode, which refits the kernel for every fold (approximately 15 minutes runtime) and is used for any reported R² value.
- For F1, whose output spans approximately 245 orders of magnitude and whose LOO-CV R² is consequently close to meaningless, the strategy was changed from a maximum-distance exploration approach, which had returned very low values over several consecutive rounds, to a narrow trust region centred on the best point identified so far.

## Performance

The primary evaluation metric is the raw-scale LOO-CV R² of each function's GP surrogate, used both as a reported diagnostic and as the live signal driving strategy routing. At round 10, the surrogate R² values by function are tabulated below:

| Function | Dim | Strategy           | LOO-CV R² |
| -------- | --- | ------------------ | --------- |
| F1       | 2D  | multi-basin TuRBO  | −0.12     |
| F2       | 2D  | multi-basin TuRBO  | +0.81     |
| F3       | 3D  | TuRBO              | +0.22     |
| F4       | 4D  | TuRBO              | +0.95     |
| F5       | 4D  | global (EI/UCB/PI) | +0.99     |
| F6       | 5D  | global (EI/UCB/PI) | +0.88     |
| F7       | 6D  | TuRBO              | +0.04     |
| F8       | 8D  | global (EI/UCB/PI) | +0.99     |

Five of the eight surrogates (F2, F4, F5, F6, F8) meet the R² ≥ 0.30 threshold and are routed to global exploitation. F1, F3, and F7 fall below this threshold and are routed to TuRBO instead, which is considered the more appropriate choice given that a poorly fitted surrogate would otherwise be optimised against directly. F7's surrogate remains weak (R²=0.04) even after freezing its one saturated dimension (x3), suggesting that the current kernel and warping choice does not fully capture the underlying structure. F2's surrogate is flagged as multimodal by the saturation diagnostic, which motivated its multi-basin TuRBO treatment; subsequent investigation suggests F2's behaviour may instead reflect genuine stochasticity in the function rather than multimodality.

A secondary diagnostic compares the GP-based and random-forest-based families of dimension-importance scores; disagreement between the two families is used qualitatively to flag potential dimension interactions rather than as a scored metric. F5's second dimension shows the largest disagreement observed (GP family ≈0.91 versus random-forest family ≈0.09), consistent with F5's optimum occurring in a region where dimensions interact jointly rather than acting independently.

The R² figures reported above were obtained with `REPORT_MODE` enabled, meaning each LOO-CV fold refits the GP's hyperparameters independently. A faster development mode exists in which the kernel is fitted once on the full dataset and reused across folds; this mode is documented as producing mildly optimistic R² estimates, since the shared hyperparameters have effectively already been exposed to the held-out point, and is used only for iterative development rather than for reported figures.

## Assumptions and limitations

The pipeline assumes that a stationary Matérn kernel can adequately represent each function's response surface. This assumption does not appear to hold for F1, whose output spans approximately 245 orders of magnitude; a single lengthscale per dimension cannot represent this range, which is reflected in F1's near-meaningless LOO-CV R².

For F2 and F4 specifically, the magnitude of improvement being sought may be smaller than the GP's own noise floor. For F2, this is consistent with genuine stochasticity in the function's output, identified during the Week 9 audit. For F4, the posterior is relatively flat near the current best region, meaning that Thompson sampling draws largely uniformly within it rather than identifying genuine structure. Neither the trust gate nor TuRBO currently distinguishes genuine improvement from noise interpreted as signal, which is an acknowledged limitation of the current design.

A function that persistently fails to clear the R² ≥ 0.30 threshold, such as F1 or F7, continues to be routed to local search indefinitely under the current rule, even where the underlying cause may be a kernel or warping mismatch rather than local structure. The router does not currently attempt an alternative kernel or warping strategy automatically in this situation.

The candidate-generation strategies used, differential evolution for the global branch and random candidate sampling for TuRBO, are heuristic searches over a finite candidate set rather than exact global optimisers. Suggested query points should therefore be understood as best-effort choices given the fitted surrogate, rather than provably optimal points.

## Ethical considerations

All data used by the pipeline is synthetic coursework data; no personal data, protected attributes, or real-world deployment risk arise from its use as documented here. Caveats such as the gap between fast-development-mode and full-report-mode LOO-CV figures and the likely stochasticity of F2 are documented explicitly so that a reader (a facilitator, a peer, or the author revisiting the repository later) can determine when a given round's suggestion should be trusted. This level of documentation is also intended to support adaptation of the pipeline to other black-box optimisation problems: a user reusing the pipeline is expected to check the LOO-CV trust gate and the multimodality or noise flags for a given function before relying on the global-exploitation branch, rather than assuming uniform reliability across all cases.

### Would additional detail improve this card?

Additional detail is not considered necessary beyond what is included here. The card documents one substantive methodological caveat (the difference between fast-development-mode and full-report-mode LOO-CV) rather than omitting them, and explains the rationale behind each function's routing decision rather than stating the outcome alone. A full per-function hyperparameter table would largely restate information already present in the pipeline's configuration and would not materially change how the reported R² values or routing decisions should be interpreted.


# References

- Srinivas, N., Krause, A., Kakade, S. M., & Seeger, M. "Gaussian Process Optimization in the Bandit Setting: No Regret and Experimental Design." _arXiv:0912.3995_ (2009).

- Eriksson, D., Pearce, M., Gardner, J., Turner, R. D., & Poloczek, M. "Scalable Global Optimization via Local Bayesian Optimization." _NeurIPS_ (2019).

- Chen, Y. "Lecture 23: Limited-Memory BFGS (L-BFGS)." Lecture notes, CS/ISyE/Math/Stat 726, University of Wisconsin–Madison (2023).

- Vodopija, A., Tušar, T., & Filipič, B. "Comparing Black-Box Differential Evolution and Classic Differential Evolution." _Proceedings of the Genetic and Evolutionary Computation Conference Companion_ (2018).

- Snoek, J., Larochelle, H., & Adams, R. P. "Practical Bayesian Optimization of Machine Learning Algorithms." _NeurIPS_ 25 (2012).



