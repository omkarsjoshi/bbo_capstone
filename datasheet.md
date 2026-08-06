# Datasheet: BBO Capstone Query and Evaluation Dataset

This datasheet documents the dataset produced by the Bayesian optimisation (BO) pipeline built for the ICL PCMLAI BBO Challenge capstone. The dataset is not a static, pre-existing collection; it is the accumulated record of every query point submitted to each of the eight benchmark functions and every value returned in response, built up round by round over the ten-week challenge.

## Motivation

The dataset was created as a direct consequence of running an active BO pipeline. In each round, the pipeline selects one query point per function, submits it to that function's black-box oracle, and records the value returned. The dataset therefore exists to support the BBO Challenge itself: finding the maximiser of each of the eight functions within a fixed budget of ten weekly queries per function.

While the initial datashet was provided as part of course material, it was subsequently developed by a single author, working as an individual contributor for the capstone component of the ICL PCMLAI programme. No external funding was involved beyond programme enrolment.

## Composition

The dataset consists of eight independent sub-datasets, one per benchmark function. Each sub-dataset contains a set of `(x, y)` pairs, where `x` is a query point in `[0,1]^d` and `y` is the scalar value returned by the corresponding evaluation. Dimensionality, range for outputs and the nature vary by function: 

| Function | Dimensions | Initial points | $y_{\min}$ (initial) | $y_{\max}$ (initial) |         Nature          |
| :------: | :--------: | :------------: | :------------------: | :------------------: | :---------------------: |
|    F1    |     2      |       10       |       ≈ 0.0000       |        0.0000        |  Near-degenerate, flat  |
|    F2    |     2      |       10       |       −0.3500        |        0.6112        |      Rugged, noisy      |
|    F3    |     3      |       15       |       −0.3800        |       −0.0051        |   Moderate, sharp dip   |
|    F4    |     4      |       30       |       −4.0300        |       −2.0458        | Smooth, negative range  |
|    F5    |     4      |       20       |        0.1100        |      1088.0000       | Dominant outlier, spiky |
|    F6    |     5      |       20       |       −0.9800        |       −0.2896        | Smooth, negative range  |
|    F7    |     6      |       30       |        0.3200        |        1.8861        |   Moderate, positive    |
|    F8    |     8      |       40       |        7.2000        |        9.7968        |     Smooth, high-D      |

Each sub-dataset began from an initial seed set supplied by the course, then grew by one point per week as the pipeline generated its round-by-round suggestion. The version of the dataset documented here is cumulative through round 9, and is the input used to generate round 10's suggestions.

The dataset is not a sample of a larger population; every point was deliberately selected by the pipeline, so the dataset is complete by construction with respect to what has been queried so far. Data is stored as NumPy arrays on disk, with one pair of files per function (`function_<n>/initial_inputs.npy` and `function_<n>/initial_outputs.npy`), loaded at runtime into pandas DataFrames and Series. No labels or annotations exist beyond the scalar `y` value itself; the function evaluation is the target signal, and there is no separate ground truth. No missing values exist within the raw query arrays, since every submitted query returns a value. 

Points within a function are not independently and identically distributed. Each point is temporally ordered and depends on the GP surrogate fitted to all preceding points, as well as the current trust-region state, so the dataset should not be treated as an unbiased sample of the input space. Due to nature of the objective, no formal train/test/validation split was performed, however leave-one-out cross-validation (LOO-CV) is used internally as a diagnostic on the fitted surrogate.

The eight black-box functions are supplied by the course and are not included in the dataset; only the queries submitted and the values returned are recorded. No personal, sensitive, or demographic information is present anywhere in the dataset, since all values are synthetic numerical outputs from mathematical benchmark functions.

## Collection process

Data was collected through an active, model-based process rather than through passive or random sampling. In each round, for every function, the pipeline fitted a Gaussian Process surrogate (a Matérn kernel with automatic relevance determination, using ν=1.5 for functions with sharper local structure and ν=2.5 for smoother functions) to the current query history, evaluated the surrogate's leave-one-out cross-validated R² as a trust score, ran a four-method dimension-importance panel to identify candidate dimensions for freezing, and then routed the function to one of several acquisition strategies (a global ensemble of Expected Improvement, Upper Confidence Bound, and Probability of Improvement; TuRBO-based local trust-region search; or space-filling exploration) depending on the trust score.

Sampling was deterministic given a fixed random seed, not random. An earlier version of the pipeline lacked a prescribed `random_state`, which meant that GP restarts drew from the global NumPy generator and produced inconsistent R² values across repeated runs of identical code, most noticeably for F1 and F2. This was corrected by fixing `random_state=42` on every GP instance and routing all sampling through seeded, per-function and per-week generators.

Data was collected weekly over the ten-round capstone schedule. The dataset documented here reflects the cumulative history through round 9. No ethical review or consent procedures apply, since no human subjects are involved at any stage.

## Preprocessing/cleaning/labelling

Raw `.npy` arrays are preserved unmodified on disk; all transformations are applied at model-fit time rather than to the stored data. A Yeo-Johnson power transform is fitted and applied to `y` prior to GP fitting for six of the eight functions (F3 through F8); F1 and F2 are modelled without transformation. Predictions are inverted back to the raw output scale for reporting, with numerical safeguards near the transform's singularity. This safeguard was added after an inversion failure was identified for F7, where a fitted Yeo-Johnson lambda of −2.62 placed the transform's pole inside the range the GP was predicting into, producing a NaN for LOO-CV R² that silently affected the routing decision until the issue was corrected.

Inputs are used without additional scaling, since all functions are already defined on the unit hypercube `[0,1]^d`; a small epsilon-based clip is applied near the domain boundary to avoid numerical issues. No missing values exist in the raw query data.

## Uses

The dataset is intended to be used for generating the pipeline's weekly query suggestions, for serving as a diagnostic record explaining why each round's query was chosen (via the associated LOO-CV R² values, dimension-importance panel, and routing decisions), and potentially as a small worked example of BO surrogate diagnostics for instructional purposes.

The dataset is not suitable for training general-purpose supervised models, since it contains too few points per function (tens of observations) and is not independently sampled; later points are chosen adaptively based on earlier ones, introducing selection bias that would propagate into any downstream model trained on it. The dataset is also not suitable for benchmarking other optimisers against this specific query history, since the history reflects the choices made by one particular pipeline rather than a fixed, optimiser-agnostic evaluation protocol. Use outside the F1-F8 context is not appropriate, as the benchmark functions have no meaning beyond the coursework setting.

The principal risk associated with this dataset is that the adaptively collected points may be mistaken for a representative or uniform sample of each function's input space. Because points are concentrated deliberately near promising regions once identified, treating the dataset as an unbiased sample would produce a misleading picture of a function's global behaviour.

## Distribution

The dataset is intended to be published in a public GitHub repository alongside the pipeline code and accompanying documentation, as plain files (`.npy` arrays under `initial_data_9/function_<n>/`), with no separate API or release mechanism. It will be available once the repository is made public for peer and facilitator review. Use is limited to academic and educational purposes within the ICL PCMLAI programme; no commercial licence is claimed. The underlying benchmark functions F1-F8 remain the property of the course and are not redistributed as source code; only the query and response pairs generated against them are included.

## Maintenance

The dataset is maintained by its sole author for the duration of the capstone project. It grows by one query per function per week throughout the duration of the project and is expected to be frozen as a historical record once all the rounds are complete. Versioning is implicit in the weekly folder-naming convention (for example, `initial_data_9` denotes the cumulative history through round 9); no formal semantic versioning scheme is applied. The dataset is retained within the GitHub repository, with no separate archival plan beyond standard repository history.