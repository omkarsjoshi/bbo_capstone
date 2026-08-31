"""
bbo_diagnostics.py: Computes the diagnostic metrics and visualisations

Every function takes the output of run_all()along with all_X/all_y explicitly as
arguments and produces the corresponding diagnostic scores and plots

Sectioned to match the notebook:
    1. Consensus panel          (dimension importance / frozen dimensions)
    2. R^2 narrative            (LOO predicted-vs-true, before/after warp)
    3. Acquisition comparison
    4. Interpretability         (marginals, GP surfaces, trust regions,
                                 clusters, smart scatter, parallel coords)
    5. Progress tracking
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.patches as patches
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull, QhullError
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from IPython.display import display
from bbo_pipeline import (EPS, UB, get_cfg, fit_warp, warp_fwd, warp_inv, matern_ls,
                           TURBO_STATE, make_gp, find_basins, aEI, aUCB, aPI, RF_TREES,
                           loo_cv)

from bbo_pipeline import (EPS, UB, get_cfg, fit_warp, warp_fwd, warp_inv, matern_ls,
                           TURBO_STATE, make_gp, find_basins, aEI, aUCB, aPI, RF_TREES)

ACQ_COLOURS = {'EI': '#2196F3', 'UCB': '#FF9800', 'PI': '#4CAF50', 'TS': '#9C27B0',
               'MD': '#795548', 'TuRBO': '#E91E63', 'TuRBO-mb': '#AD1457',
               'EI-needle': '#00838F', 'forced': '#607D8B', 'corner': '#8D6E63'}


# =====================================================================
# Diagnose the warping requirement
# =====================================================================
def compare_warps(all_X, all_y, fns=range(1, 9), tol=0.01, verbose=True):
    """Compares warp=None/log/yeo per function via loo_cv(), recommending
    whichever wins on raw R2"""
    warp_options = [None, "log", "yeo"]
    labels = {None: "None", "log": "log", "yeo": "yeo"}
    fns = list(fns)

    rows = []
    for fn in fns:
        X = all_X[fn-1].values; Yraw = all_y[fn-1].values
        current_warp = labels[get_cfg(fn).warp]
        if verbose:
            print(f"\n--- F{fn} (current: warp={current_warp}) ---")
        row = {"function": fn, "current_warp": current_warp}
        for w in warp_options:
            cfg_test = get_cfg(fn); cfg_test.warp = w
            _, r2, diag = loo_cv(X, Yraw, cfg_test)
            lab = labels[w]
            row[f"{lab}_R2"] = r2; row[f"{lab}_trim1"] = diag["r2_trim1"]; row[f"{lab}_rho"] = diag["rho"]
            if verbose:
                print(f"  warp={lab:6s}  R2={r2:+.3f}  trim1={diag['r2_trim1']:+.3f}  rho={diag['rho']:+.3f}")
        rows.append(row)

    df_warp = pd.DataFrame(rows).set_index("function")
    r2_cols    = [f"{labels[w]}_R2" for w in warp_options]
    trim1_cols = [f"{labels[w]}_trim1" for w in warp_options]
    rho_cols   = [f"{labels[w]}_rho" for w in warp_options]

    best_r2    = df_warp[r2_cols].idxmax(axis=1).str.replace("_R2", "", regex=False)
    best_rho   = df_warp[rho_cols].idxmax(axis=1).str.replace("_rho", "", regex=False)
    best_trim1 = df_warp[trim1_cols].idxmax(axis=1).str.replace("_trim1", "", regex=False)
    df_warp["best_by_R2"] = best_r2
    df_warp["metrics_agree"] = (best_rho == best_r2) & (best_trim1 == best_r2)

    current_r2 = df_warp.apply(lambda r: r[f"{r['current_warp']}_R2"], axis=1)
    gap = df_warp[r2_cols].max(axis=1) - current_r2

    def _verdict(best, current, row_gap, agree):
        if best == current: return f"KEEP ({current})"
        if row_gap < tol: return f"KEEP ({current} -- gap too small to act on)"
        flag = "" if agree else "  [CAUTION: rho/trim1 disagree]"
        return f"SWITCH -> {best}{flag}"

    df_warp["recommendation"] = [_verdict(b, c, g, a) for b, c, g, a in
        zip(df_warp["best_by_R2"], df_warp["current_warp"], gap, df_warp["metrics_agree"])]

    def _highlight_row_max(row):
        rounded = row.round(3); is_max = rounded == rounded.max()
        return ['background-color: #a6d96a' if v else '' for v in is_max]

    display_cols = ["function", "current_warp"] + r2_cols + ["best_by_R2", "recommendation"]
    styled = (df_warp.reset_index()[display_cols].style
        .apply(_highlight_row_max, subset=r2_cols, axis=1)
        .format({c: "{:+.3f}" for c in r2_cols})
        .hide(axis="index")
        .set_table_styles([
            {'selector': 'th, td', 'props': [('padding', '6px 14px'), ('border', 'none')]},
            {'selector': 'table', 'props': [('border-collapse', 'collapse')]},
        ]))

    if verbose:
        print("\n" + "="*70); print("SUMMARY"); print("="*70)
    display(styled)
    return df_warp

# =====================================================================
# 1. CONSENSUS PANEL — dimension importance / frozen dimensionss
# =====================================================================

def print_importance_table(diagnostics, fns=range(1, 9)):
    """Creates dimension importance table, one function at a time."""
    for fn in fns:
        d = diagnostics[fn]
        print(f"\nF{fn}  (LOO R^2 = {d['loo_r2']:+.3f})")
        print(d['importance'].to_string(index=False))


def plot_importance_bars(diagnostics, fns=range(1, 9)):
    """Plots individual importance metrics  (ard/gp_grad/rf_perm/shap) using grouped bars
    and marks the frozen diemnsions.
    """
    n = len(list(fns)); ncols = min(4, n); nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5*ncols, 4.5*nrows), squeeze=False)
    axes = axes.ravel()
    metrics = ['ard', 'gp_grad', 'rf_perm', 'shap']
    mc = {'ard': '#1565C0', 'gp_grad': '#5E9BE6', 'rf_perm': '#2E7D32', 'shap': '#81C784'}
    for i, fn in enumerate(fns):
        ax = axes[i]; imp = diagnostics[fn]['importance']; dims = imp['dim'].tolist()
        x = np.arange(len(dims)); w = 0.2
        for k, m in enumerate(metrics):
            ax.bar(x + (k-1.5)*w, imp[m].values, w, label=m, color=mc[m])
        for j, fr in enumerate(imp['frozen'].values):
            if fr: ax.text(j, 1.04, '\u2744', ha='center', fontsize=11)
        ax.set_xticks(x); ax.set_xticklabels(dims); ax.set_ylim(0, 1.15)
        ax.set_title(f'F{fn}', fontweight='bold'); ax.grid(axis='y', alpha=.3)
        if i == 0: ax.legend(fontsize=8, ncol=2)
    for k in range(len(list(fns)), len(axes)): axes[k].axis('off')
    fig.suptitle('Individual importance metrics per dimension  |  GP family (blues) vs '
                 'RF family (greens)  |  \u2744 frozen', fontweight='bold')
    plt.tight_layout(); plt.show()


def plot_importance_heatmaps(diagnostics, fns=range(1, 9), nd_max=8):
    """Plots dimension importance heatmaps for GP-family and RF-family, along with
    the absolute per-dimension disagreement between the two families' consensus
    importance scores."""
    plt.close('all')
    fig, axes = plt.subplots(1, 3, figsize=(21, 5.5))
    truncated_rdylgn = mcolors.LinearSegmentedColormap.from_list(
        'truncated_rdylgn', plt.cm.RdYlGn_r(np.linspace(0.1, 0.8, 256)))
    panels = [('gp_fam', 'GP-family (ard+gp_grad)', truncated_rdylgn),
          ('rf_fam', 'RF-family (rf_perm+shap)', truncated_rdylgn),
          ('disagree', '|GP - RF|  (bright = interaction)', truncated_rdylgn)]
    fns = list(fns)
    for ax, (key, title, cmap) in zip(axes, panels):
        M = np.full((len(fns), nd_max), np.nan)
        for i, fn in enumerate(fns):
            imp = diagnostics[fn]['importance']; M[i, :len(imp)] = imp[key].values
        im = ax.imshow(np.ma.masked_invalid(M), cmap=cmap, aspect='auto', vmin=0, vmax=1, alpha=1)
        for i, fn in enumerate(fns):
            imp = diagnostics[fn]['importance']
            for j in range(len(imp)):
                fr = imp['frozen'].iloc[j]
                lab = ('\u2744' if (fr and key != 'disagree') else f'{M[i,j]:.2f}')
                ax.text(j, i, lab, ha='center', va='center', fontsize=14,
                        color='black', fontweight='bold' if fr else 'normal')
        ax.set_xticks(range(nd_max)); ax.set_xticklabels([f'x{i+1}' for i in range(nd_max)])
        ax.set_yticks(range(len(fns))); ax.set_yticklabels([f'F{f}' for f in fns])
        ax.set_title(title, fontweight='bold')
        fig.colorbar(im, ax=ax, fraction=.046, pad=.04)
    fig.suptitle('Family-aware importance  |  \u2744 = frozen in acquisition',
                 fontweight='bold', y=1.03)
    plt.tight_layout(); plt.show()


def plot_importance_scatter(diagnostics, fns=range(1, 9)):
    """Plots dimension importance for GP-family and RF-family per dimension, highlighting
    the points off the diagonal as potentially having interaction"""
    n = len(list(fns)); ncols = min(4, n); nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows), squeeze=False)
    axes = axes.ravel()
    for i, fn in enumerate(fns):
        ax = axes[i]; imp = diagnostics[fn]['importance']
        ax.plot([0, 1], [0, 1], 'k--', alpha=.4, zorder=1)
        for _, r in imp.iterrows():
            amb = r['disagree'] > 0.5
            ax.scatter(r['gp_fam'], r['rf_fam'], s=110, zorder=3,
                       c='#d62728' if amb else '#1f77b4', edgecolors='k', linewidths=.5)
            ax.annotate(r['dim'], (r['gp_fam'], r['rf_fam']), fontsize=8,
                        xytext=(4, 4), textcoords='offset points')
        ax.set_xlim(-.05, 1.05); ax.set_ylim(-.05, 1.05)
        ax.set_xlabel('GP-family'); ax.set_ylabel('RF-family')
        ax.set_title(f'F{fn}', fontweight='bold'); ax.grid(alpha=.3)
    for k in range(len(list(fns)), len(axes)): axes[k].axis('off')
    fig.suptitle('GP-family vs RF-family per dimension  |  red = interaction candidate '
                 '(|diff|>0.5)', fontweight='bold')
    plt.tight_layout(); plt.show()


def plot_lengthscale_heatmap(all_X, gps):
    """Plots ARD lengthscale per function/dimension on a log-scale, marking the 
    saturated dimensions with "X" """
    rows = []
    for fn in range(1, len(all_X)+1):
        ls = matern_ls(gps[fn]); r = {'function': fn}
        for c, l in zip(all_X[fn-1].columns, ls): r[c] = round(float(l), 4)
        rows.append(r)
    df_ls = pd.DataFrame(rows).set_index('function')
    xcols = [c for c in df_ls.columns if c.startswith('x')]
    data = df_ls[xcols].values.astype(float)
    logd = np.log10(np.where(np.isnan(data), np.nan, data))
    fig, ax = plt.subplots(figsize=(max(8, len(xcols)*1.2), 5))
    im = ax.imshow(np.ma.masked_invalid(logd), cmap='RdYlGn_r', aspect='auto',
                    vmin=np.log10(.005), vmax=np.log10(100))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if not np.isnan(data[i, j]):
                v = data[i, j]
                ax.text(j, i, 'X' if v >= 10 else f'{v:.2f}', ha='center', va='center',
                        fontsize=9, color='white' if v >= 10 else 'black',
                        fontweight='bold' if v >= 10 else 'normal')
    ax.set_xticks(range(len(xcols))); ax.set_xticklabels(xcols)
    ax.set_yticks(range(len(df_ls))); ax.set_yticklabels([f'F{f}' for f in df_ls.index])
    ax.set_title('ARD lengthscales | green=informative | red=irrelevant (X=ceiling)',
                 fontweight='bold')
    cbar = fig.colorbar(im, ax=ax); cbar.set_ticks([np.log10(v) for v in [.01, .1, 1, 10, 100]])
    cbar.set_ticklabels(['0.01', '0.1', '1', '10', '100']); cbar.set_label('log10(lengthscale)')
    plt.tight_layout(); plt.show()
    return df_ls


def shap_values_for(all_X, all_y, fn):
    """Fits the SAME RandomForestRegressor config that dimension_importance()
    uses and on the same warped y.
    """
    import shap
    from sklearn.ensemble import RandomForestRegressor
    cfg = get_cfg(fn)
    X = all_X[fn-1].values; Yraw = all_y[fn-1].values
    Yw, wm = warp_fwd(Yraw, cfg.warp)
    rf = RandomForestRegressor(RF_TREES, random_state=42, n_jobs=-1).fit(X, Yw)
    sv = shap.TreeExplainer(rf).shap_values(X, check_additivity=False)
    names = [f'x{i+1}' for i in range(X.shape[1])]
    return X, sv, names


def plot_shap_beeswarm(all_X, all_y, fn):
    """Plots every sample's per-dimension SHAP value, coloured by that sample's 
    own feature value, with dimensions sorted by mean |SHAP|. Also captures the direction 
    in x that improves y and also how broad the importance is."""
    import shap
    X, sv, names = shap_values_for(all_X, all_y, fn)
    exp = shap.Explanation(values=sv, data=X, feature_names=names)
    plt.figure(figsize=(7, 0.5*X.shape[1] + 2))
    shap.plots.beeswarm(exp, show=False)
    plt.title(f'F{fn} — SHAP beeswarm (n={X.shape[0]})', fontweight='bold')
    plt.tight_layout(); plt.show()


def plot_all_shap_beeswarm(all_X, all_y, fns=range(1, 9)):
    """Loops plot_shap_beeswarm() over every function"""
    for fn in fns:
        plot_shap_beeswarm(all_X, all_y, fn)


# =====================================================================
# 2. R^2 narrative (trust-gate)
# =====================================================================

def plot_loo_diagnostic(diagnostics, fns=range(1, 9)):
    """Plots predicted vs true y per function from LOO-CV, with title coloured by whether
    the function creates the R2 gate"""
    n = len(list(fns)); ncols = min(4, n); nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows), squeeze=False)
    axes = axes.ravel()
    for i, fn in enumerate(fns):
        ax = axes[i]; d = diagnostics[fn]
        yt = np.asarray(d['loo_true_raw']); yp = np.asarray(d['loo_preds_raw'])
        lo = min(yt.min(), yp.min()); hi = max(yt.max(), yp.max()); pad = 0.05*(hi-lo)+1e-9
        ax.plot([lo-pad, hi+pad], [lo-pad, hi+pad], 'k--', alpha=.5, zorder=1)
        ax.scatter(yt, yp, s=45, alpha=.75, edgecolors='k', linewidths=.3, c='#1f77b4', zorder=3)
        r2 = d['loo_r2']; col = '#2E7D32' if r2 >= .3 else '#d62728'
        ax.set_title(f'F{fn}  |  LOO R2={r2:+.3f}', fontweight='bold', color=col)
        ax.set_xlabel('true y'); ax.set_ylabel('LOO predicted y'); ax.grid(alpha=.3)
    for k in range(len(list(fns)), len(axes)): axes[k].axis('off')
    fig.suptitle('Leave-one-out CV: predicted vs true (dashed = perfect)  |  '
                 'red = fails gate (R2<0.30)', fontweight='bold')
    plt.tight_layout(); plt.show()


def print_r2_story(df_cmp):
    """Prints the comparison between the R2 based on warped-scale y calculated on full-dataset, and hence
    prone to leak, vs the corrected raw space, per-fold-warp.
    
    Metric              Description
    
    r2                  Pooled raw-scale LOO R2
    r2_trim1            R2 with the single worst-error group dropped
    rho                 Spearman rank correlation, raw y vs raw LOO predicted (unscaled)
    worst_group         Group id with the largest total squared error
    amp_at_worst_fold   Mean |dy/dz| at that group (warp amplification factor)
    n_groups            Distinct LOO groups (duplicates merged)
    n_points            Total raw data points
    single_fold_driven  Flags (R2<0.30) ∧ (Rtrim1≥0.30) ∧ (ρ≥0.70)
    n_clipped           Folds whose z-prediction hit the Yeo-Johnson pole clip
    clip_dominated      Flags (R2<0.30) ∧ (n_clipped>=2) ∧ (ρ≥0.70)
    """
    pd.set_option("display.width", 200)
    cols = ["d", "warp", "nu", "loo_r2_leaky_warpedscale", "loo_r2_leakproof_rawscale", "delta",
            "loo_r2_trim1", "loo_rho", "amp_at_worst_fold", "n_clipped",
            "n_dup_groups", "route_before", "route_after", "strategy_used",
            "single_fold_driven", "clip_dominated"]
    print(df_cmp[cols].to_string())
    flagged = df_cmp[df_cmp["single_fold_driven"]]
    if len(flagged):
        print("\nsingle_fold_driven (one hard point is doing all the damage, not a "
              "broadly unreliable GP) -- worth a manual look, route() doesn't act on "
              f"this automatically: {flagged.index.tolist()}")
    clipped = df_cmp[df_cmp["clip_dominated"]]
    if len(clipped):
        print("\nclip_dominated (several LOO folds hit the Yeo-Johnson pole-safety clip -- "
              f"a metric artefact, not a GP failure; trust loo_rho instead): {clipped.index.tolist()}")


# =====================================================================
# 3. ACQUISITION COMPARISON
# =====================================================================

def plot_acquisition_comparison(df_all, df_best, fns=range(1, 9)):
    """Bar chart of EI/UCB/PI's predicted y per function, with bold bar for the chosen one."""
    n = len(list(fns)); ncols = min(4, n); nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5*ncols, 4*nrows), squeeze=False)
    axes = axes.ravel()
    for i, fn in enumerate(fns):
        ax = axes[i]; g = df_all[df_all['function'] == fn]
        winner = df_best.loc[fn, 'acquisition']; yc = df_best.loc[fn, 'y_max_so_far']
        names = g['acquisition'].tolist(); preds = g['predicted_y'].tolist()
        ax.bar(names, preds, color=[ACQ_COLOURS.get(a, 'grey') for a in names],
               edgecolor='black', linewidth=[3 if a == winner else 0.8 for a in names])
        ax.axhline(yc, color='red', ls='--', alpha=.7, label=f'best: {yc:.3g}')
        ax.set_title(f'F{fn} — {int(g["dimensions"].iloc[0])}D | {winner}',
                     fontsize=10, fontweight='bold')
        ax.legend(fontsize=7)
    for k in range(len(list(fns)), len(axes)): axes[k].axis('off')
    fig.suptitle('Acquisition comparison — bold = selected | red = current best',
                 fontweight='bold')
    plt.tight_layout(); plt.show()


def plot_acquisition_alignment(df_all, fns=range(1, 9)):
    """Plots line plot of each acquisition's suggested x across dimensions per function,
    showing if EI/UCB/PI agree on a region."""
    n = len(list(fns)); ncols = min(4, n); nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3.5*nrows), squeeze=False)
    axes = axes.ravel()
    cols_all = [f'x{i}_suggested' for i in range(1, 9)]
    for i, fn in enumerate(fns):
        ax = axes[i]; g = df_all[df_all['function'] == fn]; nd = int(g['dimensions'].iloc[0])
        cs = cols_all[:nd]
        for _, r in g.iterrows():
            ax.plot(range(nd), r[cs].astype(float).values, 'o-',
                    label=r['acquisition'], color=ACQ_COLOURS.get(r['acquisition'], 'grey'),
                    alpha=.8)
        ax.set_xticks(range(nd)); ax.set_xticklabels([c.replace('_suggested', '') for c in cs],
                                                       rotation=45)
        ax.set_ylim(-.05, 1.05); ax.set_title(f'F{fn}', fontweight='bold')
        ax.grid(alpha=.3); ax.legend(fontsize=7)
    for k in range(len(list(fns)), len(axes)): axes[k].axis('off')
    fig.suptitle('x_next across dimensions, by acquisition', fontweight='bold')
    plt.tight_layout(); plt.show()


# =====================================================================
# 4. INTERPRETABILITY
# =====================================================================

def plot_marginals(all_X, all_y, fn):
    """Plots y vs. each raw input dimension, colored by y-rank, with a linear trend line."""
    X = all_X[fn-1]; y = all_y[fn-1]; nd = X.shape[1]
    nc = min(4, nd); nr = int(np.ceil(nd/nc))
    fig, axes = plt.subplots(nr, nc, figsize=(5*nc, 4*nr), squeeze=False)
    yr = y.rank(pct=True)
    for i, c in enumerate(X.columns):
        ax = axes[i//nc][i % nc]
        ax.scatter(X[c], y, c=yr, cmap='rainbow', s=60, edgecolors='k', linewidths=.3)
        z = np.polyfit(X[c], y, 1); xl = np.linspace(X[c].min(), X[c].max(), 50)
        ax.plot(xl, np.poly1d(z)(xl), 'k--', alpha=.6)
        ax.set_title(f'{c} vs y | r={X[c].corr(y):.2f}', fontsize=10, fontweight='bold')
    for k in range(nd, nr*nc): axes[k//nc][k % nc].axis('off')
    fig.suptitle(f'F{fn} marginals', fontweight='bold'); plt.tight_layout(); plt.show()


def plot_all_marginals(all_X, all_y, fns=range(1, 9)):
    """Plot marginal plots for all functions sequentially."""
    for fn in fns:
        plot_marginals(all_X, all_y, fn)


def plot_gp_surfaces(all_X, all_y, gps, top_dims, fns=range(1, 9), n_grid=40, out_path=None):
    """ Plots interactive Plotly 3D GP-mean surface over each function's top-2 dimensions,
    with all other dimensions frozen at the incumbent.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fns = list(fns)
    ncols = min(4, len(fns)); nrows = -(-len(fns) // ncols)
    specs = [[{'type': 'surface'} for _ in range(ncols)] for _ in range(nrows)]
    titles = []
    surfaces, points = {}, {}

    for fn in fns:
        cfg = get_cfg(fn)
        X = all_X[fn-1].values; Yraw = all_y[fn-1].values
        wm = fit_warp(Yraw, cfg.warp)
        gp = gps[fn]
        d1n, d2n = top_dims[fn][0], top_dims[fn][1]
        j1, j2 = int(d1n[1:]) - 1, int(d2n[1:]) - 1
        xb = X[np.argmax(Yraw)].copy()

        g1 = np.linspace(EPS, UB, n_grid); g2 = np.linspace(EPS, UB, n_grid)
        G1, G2 = np.meshgrid(g1, g2)
        grid = np.tile(xb, (n_grid*n_grid, 1))
        grid[:, j1] = G1.ravel(); grid[:, j2] = G2.ravel()
        predw = gp.predict(grid)
        Z = warp_inv(predw, wm).reshape(n_grid, n_grid)

        surfaces[fn] = (g1, g2, Z, d1n, d2n)
        points[fn] = (X[:, j1], X[:, j2], Yraw)
        titles.append(f"F{fn} — ({d1n},{d2n})")

    fig = make_subplots(rows=nrows, cols=ncols, specs=specs, subplot_titles=titles,
                         horizontal_spacing=0.02, vertical_spacing=0.08)
    positions = [(r+1, c+1) for r in range(nrows) for c in range(ncols)][:len(fns)]
    for fn, (row, col) in zip(fns, positions):
        g1, g2, Z, d1n, d2n = surfaces[fn]
        fig.add_trace(go.Surface(x=g1, y=g2, z=Z, colorscale='rainbow', showscale=False,
                                  opacity=0.85, name=f'F{fn} surface',
                                  hovertemplate=f'{d1n}=%{{x:.3f}}<br>{d2n}=%{{y:.3f}}<br>'
                                                'pred y=%{z:.4g}<extra></extra>'),
                      row=row, col=col)
        xo, yo, yr = points[fn]
        fig.add_trace(go.Scatter3d(x=xo, y=yo, z=yr, mode='markers',
                                    marker=dict(size=3, color=yr, colorscale='Rainbow',
                                                line=dict(width=0.5, color='black')),
                                    name=f'F{fn} observed', showlegend=False,
                                    hovertemplate=f'{d1n}=%{{x:.3f}}<br>{d2n}=%{{y:.3f}}<br>'
                                                  'true y=%{z:.4g}<extra></extra>'),
                      row=row, col=col)

    for i, fn in enumerate(fns):
        _, _, _, d1n, d2n = surfaces[fn]
        scene_key = 'scene' if i == 0 else f'scene{i+1}'
        fig.update_layout(**{scene_key: dict(
            xaxis_title=d1n, yaxis_title=d2n, zaxis_title='y', aspectmode='cube',
            camera=dict(projection=dict(type='orthographic')))})  # true height comparison,
                                                                    # no perspective foreshortening

    fig.update_layout(
        title="GP mean-surface slices — top-2 dims per function, rest fixed at incumbent "
              "(drag to rotate, scroll to zoom, hover for values)",
        height=420*nrows, margin=dict(t=90, b=10))

    if out_path:
        fig.write_html(out_path, include_plotlyjs='cdn')
        print(f"Saved: {out_path}")
    return fig


def plot_gp_surfaces_interactive(all_X, all_y, gps, top_dims, fns=(3, 4, 5, 6, 7, 8),
                                  n_grid=32, n_slider=21, out_path="gp_surfaces_interactive.html"):
    """Plots interactive Plotly 3D GP-mean surface over each function's top-2 dimensions
    for functions with 3+ dimensions, with the third most important dimension becoming
    a slider instead of being frozen at the incumbent. Moving it re-renders the GP
    surface for every function at once.

    For 2D functions (F1, F2) that have no third dimension to slide, use
    plot_gp_surfaces for those instead. 
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fns = list(fns)
    ncols = min(3, len(fns)); nrows = -(-len(fns) // ncols)
    specs = [[{'type': 'surface'} for _ in range(ncols)] for _ in range(nrows)]
    titles, meta = [], []

    grids, warps, incumbents, obs = {}, {}, {}, {}
    for fn in fns:
        cfg = get_cfg(fn)
        X = all_X[fn-1].values; Yraw = all_y[fn-1].values
        wm = fit_warp(Yraw, cfg.warp)
        d1n, d2n, d3n = top_dims[fn][0], top_dims[fn][1], top_dims[fn][2]
        j1, j2, j3 = int(d1n[1:])-1, int(d2n[1:])-1, int(d3n[1:])-1
        xb = X[np.argmax(Yraw)].copy()
        grids[fn] = (j1, j2, j3, d1n, d2n, d3n)
        warps[fn] = wm
        incumbents[fn] = xb
        obs[fn] = (X, Yraw)
        titles.append(f"F{fn} — ({d1n},{d2n}) | sliding {d3n}")

    fig = make_subplots(rows=nrows, cols=ncols, specs=specs,
                         subplot_titles=titles,
                         horizontal_spacing=0.02, vertical_spacing=0.06)

    g = np.linspace(EPS, UB, n_grid)
    G1, G2 = np.meshgrid(g, g)
    slider_vals = np.linspace(EPS, UB, n_slider)

    def compute_Z(fn, d3_val):
        cfg = get_cfg(fn)
        j1, j2, j3, *_ = grids[fn]
        xb = incumbents[fn]
        grid = np.tile(xb, (n_grid*n_grid, 1))
        grid[:, j1] = G1.ravel(); grid[:, j2] = G2.ravel(); grid[:, j3] = d3_val
        predw = gps[fn].predict(grid)
        pred = warp_inv(predw, warps[fn])
        return pred.reshape(n_grid, n_grid)

    # initial surfaces (slider at its first position) + fixed observed-point scatter
    positions = [(r+1, c+1) for r in range(nrows) for c in range(ncols)][:len(fns)]
    for (fn, (row, col)) in zip(fns, positions):
        Z0 = compute_Z(fn, slider_vals[0])
        j1, j2, j3, d1n, d2n, d3n = grids[fn]
        fig.add_trace(go.Surface(x=g, y=g, z=Z0, colorscale='rainbow', showscale=False,
                                  opacity=0.85, name=f'F{fn} surface'), row=row, col=col)
    for (fn, (row, col)) in zip(fns, positions):
        X, Yraw = obs[fn]; j1, j2, *_ = grids[fn]
        fig.add_trace(go.Scatter3d(x=X[:, j1], y=X[:, j2], z=Yraw, mode='markers',
                                    marker=dict(size=3, color=Yraw, colorscale='Rainbow',
                                                line=dict(width=0.5, color='black')),
                                    name=f'F{fn} observed', showlegend=False),
                      row=row, col=col)

    n_fns = len(fns)
    frames = []
    for s_i, v in enumerate(slider_vals):
        frame_data = [go.Surface(z=compute_Z(fn, v)) for fn in fns]
        frames.append(go.Frame(data=frame_data, traces=list(range(n_fns)), name=f"{v:.3f}"))
    fig.frames = frames

    sliders = [dict(
        active=0,
        currentvalue=dict(prefix="normalized 3rd-dim position: "),
        pad=dict(t=40),
        steps=[dict(method="animate", label=f"{v:.2f}",
                     args=[[f"{v:.3f}"], dict(mode="immediate",
                                               frame=dict(duration=0, redraw=True),
                                               transition=dict(duration=0))])
               for v in slider_vals],
    )]

    for i, fn in enumerate(fns):
        j1, j2, j3, d1n, d2n, d3n = grids[fn]
        scene_key = 'scene' if i == 0 else f'scene{i+1}'
        fig.update_layout(**{scene_key: dict(
            xaxis_title=d1n, yaxis_title=d2n, zaxis_title='y',
            aspectmode='cube', camera=dict(projection=dict(type='orthographic')))})

    fig.update_layout(
        title="GP surfaces with a shared slider on each function's 3rd most "
              "important dimension (position is normalized 0-1; it maps onto "
              "a DIFFERENT physical dimension per function)",
        sliders=sliders, height=420*nrows, margin=dict(t=90, b=10))

    if out_path:
        fig.write_html(out_path, include_plotlyjs='cdn')
        print(f"Saved: {out_path}")
    return fig


def turbo_tracer(all_X, all_y, fn, ni=(10, 10, 15, 30, 20, 20, 30, 40), st_=3, ft=3):
    """
    Computes the TuRBO trust-region length trajectory week-by-week under today's config for
    the function that currently uses TuRBO strategy.
    """
    cfg = get_cfg(fn)
    X_full = all_X[fn-1].values; Y_full = all_y[fn-1].values
    n0 = ni[fn-1]
    n_rounds = len(Y_full) - n0
    if n_rounds < 1:
        raise ValueError(f"F{fn}: no post-initial-sampling rounds to trace (n={len(Y_full)}, ni={n0})")

    state = {}   # keyed by basin row-index, matching TURBO_STATE's (fn,b) keying
    records = []
    for week in range(1, n_rounds + 1):
        X_hist = X_full[:n0 + week]; Y_hist = Y_full[:n0 + week]
        Yw, wm = warp_fwd(Y_hist, cfg.warp)
        gp = make_gp(X_hist, Yw, cfg)
        ls = matern_ls(gp)
        anchors = find_basins(X_hist, Yw, min_sep=cfg.min_sep, rank=cfg.basin_rank,
                               max_basins=cfg.max_basins, ls=ls)
        b = anchors[(week + cfg.basin_offset) % len(anchors)]
        ym = float(Yw[b])
        s = state.setdefault(b, {"length": cfg.turbo_ti, "succ": 0, "fail": 0, "last": ym})
        if ym > s["last"] + 1e-9:
            s["succ"] += 1; s["fail"] = 0
        else:
            s["fail"] += 1; s["succ"] = 0
        event = "none"
        if s["succ"] >= st_:
            s["length"] = min(0.8, s["length"]*2); s["succ"] = 0; event = "grow"
        elif s["fail"] >= ft:
            s["length"] = max(0.05, s["length"]/2); s["fail"] = 0; event = "shrink"
        s["last"] = ym
        records.append(dict(week=week, basin_row=b, n_basins=len(anchors), length=s["length"],
                             succ_streak=s["succ"], fail_streak=s["fail"], event=event,
                             y_incumbent_raw=float(Y_hist.max())))
    return pd.DataFrame(records)


def plot_turbo_tracer(all_X, all_y, fns=None, ni=(10, 10, 15, 30, 20, 20, 30, 40)):
    """Reports the CURRENT trust-region state for every turbo/turbo_mb
    function, read directly from TURBO_STATE. Replaces the earlier
    week-by-week replay: that reconstruction requires re-deriving
    find_basins() from scratch at every past week under TODAY'S CONFIG,
    which produces a shifting set of basin anchors as new data arrives
    and is unreliable for a function with no real local structure (e.g.
    F1). The live state answers the question that actually matters --
    has the search collapsed -- directly and without a counterfactual
    replay."""
    if fns is None:
        fns = [fn for fn in range(1, len(all_X)+1) if get_cfg(fn).strategy in ('turbo', 'turbo_mb')]
    if not fns:
        print("No function is currently configured for turbo/turbo_mb -- nothing to report.")
        return {}

    status = {}
    for fn in fns:
        s = TURBO_STATE[fn]
        floor = DEFAULTS.get("ls_floor", 0.05)  # trust-region floor, same constant used in _opt_active's clamp
        collapsed = s["length"] <= floor
        streak = f"{s['succ']} success" if s["succ"] > 0 else (f"{s['fail']} failure" if s["fail"] > 0 else "0")
        state_str = "COLLAPSED (at floor)" if collapsed else "active"
        print(f"F{fn}: length={s['length']:.3f}  (floor={floor:.3f})  "
              f"current streak={streak}  -- {state_str}")
        status[fn] = dict(length=s["length"], succ=s["succ"], fail=s["fail"], collapsed=collapsed)
    return status


def de_tracer(score_fn, active_dims, anchor, popsize=20, maxiter=80, rng_seed=42):
    """
    Computes differential_evolution call, recording the full population every generation
    for the 'global' strategy, until the final L-BFGS-B refinement step"""
    from scipy.optimize._differentialevolution import DifferentialEvolutionSolver
    active_dims = list(active_dims)

    def neg(x):
        x = np.atleast_2d(x)
        if x.shape[0] == len(active_dims) and x.shape[1] != len(active_dims):
            x = x.T
        full = np.tile(anchor, (x.shape[0], 1)); full[:, active_dims] = x
        s = -np.atleast_1d(score_fn(full))
        return s if s.shape[0] > 1 else s[0]

    solver = DifferentialEvolutionSolver(
        neg, [(EPS, UB)] * len(active_dims), popsize=popsize, maxiter=maxiter,
        tol=1e-7, rng=rng_seed, vectorized=True, updating='deferred', polish=True)

    history = []
    for gen, (xk, conv) in enumerate(solver, start=1):
        history.append(dict(gen=gen, best_x=xk.copy(), convergence=float(conv),
                             best_score=-float(solver.population_energies.min()),
                             population=solver.population.copy(),
                             energies=solver.population_energies.copy()))
        if solver.converged() or gen >= maxiter:
            break
    return history


def trace_de_for_function(all_X, all_y, gps, diagnostics, fn, acquisition='EI'):
    """Sets up the exact score function suggest_global builds for one
    acquisition of one 'global'-strategy function (using the already-fit
    gp/importance, then traces its DE run."""
    cfg = get_cfg(fn)
    X = all_X[fn-1].values; Yraw = all_y[fn-1].values
    Yw, wm = warp_fwd(Yraw, cfg.warp)
    gp = gps[fn]; imp = diagnostics[fn]['importance']
    nd = X.shape[1]; ys = Yw.std(); ym = Yw.max(); xb = X[np.argmax(Yw)]
    active = [j for j in range(nd) if not imp['frozen'].iloc[j]]

    def sc(f):
        m, s = gp.predict(f, return_std=True)
        if acquisition == 'EI':
            return aEI(m, s, ym, ys)
        elif acquisition == 'UCB':
            return aUCB(m, s, len(Yw), max(len(active), 1))
        return aPI(m, s, ym, ys)

    hist = de_tracer(sc, active, xb)
    return hist, active


def plot_de_convergence(all_X, all_y, gps, diagnostics, fns=None, acquisition='EI'):
    """Plots historical best score vs generation, along with the population's energy
    spread (5th-95th percentile band) at each generation, for every
    'global'-strategy function"""
    if fns is None:
        fns = [fn for fn in range(1, len(all_X)+1) if bp_route_is_global(gps, diagnostics, fn)]
    if not fns:
        print("No function is currently routed to 'global' -- nothing to trace.")
        return {}
    traces = {fn: trace_de_for_function(all_X, all_y, gps, diagnostics, fn, acquisition)[0]
              for fn in fns}

    n = len(fns); ncols = min(3, n); nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows), squeeze=False)
    axes = axes.ravel()
    for i, fn in enumerate(fns):
        ax = axes[i]; hist = traces[fn]
        gens = [h['gen'] for h in hist]
        best = [h['best_score'] for h in hist]
        p5 = [np.percentile(-h['energies'], 5) for h in hist]
        p95 = [np.percentile(-h['energies'], 95) for h in hist]
        ax.plot(gens, best, color='#1565C0', lw=2, label='population best')
        ax.fill_between(gens, p5, p95, color='#1565C0', alpha=.15, label='pop 5th-95th pct')
        ax.set_xlabel('DE generation'); ax.set_ylabel(f'{acquisition} score')
        ax.set_title(f'F{fn} — {len(gens)} generations to converge', fontweight='bold', fontsize=10)
        ax.grid(alpha=.3); ax.legend(fontsize=7)
    for k in range(len(fns), len(axes)): axes[k].axis('off')
    fig.suptitle(f'Differential evolution convergence ({acquisition} acquisition)', fontweight='bold')
    plt.tight_layout(); plt.show()
    return traces


def bp_route_is_global(gps, diagnostics, fn):
    """A helper function that returns True if the function's current effective strategy
    is 'global'"""
    cfg = get_cfg(fn)
    if cfg.strategy == 'global':
        return True
    if cfg.strategy != 'auto':
        return False
    from bbo_pipeline import route
    return route(diagnostics[fn]['loo_r2'], diagnostics[fn]['importance']) == 'global'



def plot_de_climbing_interactive(all_X, all_y, gps, diagnostics, fns=None, acquisition='EI',
                                  n_grid=35, out_path=None):
    """Plots an interactive plot of DE population climbing the surface 
    of acquisition surface it's optimizing in 3D, against the function's
    top-2 active dimensions. All other active dimensions fixed at the
    incumbent.
    
    For a function with more than 2 active (unfrozen) dimensions, an individual's 
    TRUE score depends on where it sits in ALL of them, but the background surface 
    only shows 2, holding every other active dim at the incumbent. So a population 
    point can legitimately float above or below the surface. This gap is due to the
    dimensions this view can't show. 
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if fns is None:
        fns = [fn for fn in range(1, len(all_X)+1) if bp_route_is_global(gps, diagnostics, fn)]
    if not fns:
        print("No function is currently routed to 'global' -- nothing to trace.")
        return None
    fns = list(fns)

    traces, dims, surfaces = {}, {}, {}
    for fn in fns:
        cfg = get_cfg(fn)
        X = all_X[fn-1].values; Yraw = all_y[fn-1].values
        Yw, wm = warp_fwd(Yraw, cfg.warp)
        gp = gps[fn]; imp = diagnostics[fn]['importance']
        active = [j for j in range(X.shape[1]) if not imp['frozen'].iloc[j]]
        active_ranked = sorted(active, key=lambda j: -imp['consensus'].iloc[j])
        if len(active_ranked) < 2:
            print(f"F{fn}: fewer than 2 active dims -- skipping.")
            continue
        j1, j2 = active_ranked[0], active_ranked[1]
        ai1, ai2 = active.index(j1), active.index(j2)
        ys = Yw.std(); ym = Yw.max(); xb = X[np.argmax(Yw)]

        def sc(f, acquisition=acquisition, gp=gp, ym=ym, ys=ys, active=active):
            m, s = gp.predict(f, return_std=True)
            if acquisition == 'EI':
                return aEI(m, s, ym, ys)
            elif acquisition == 'UCB':
                return aUCB(m, s, len(Yw), max(len(active), 1))
            return aPI(m, s, ym, ys)

        hist = de_tracer(sc, active, xb)
        traces[fn] = hist
        dims[fn] = (ai1, ai2, f'x{j1+1}', f'x{j2+1}')

        g1 = np.linspace(EPS, UB, n_grid); g2 = np.linspace(EPS, UB, n_grid)
        G1, G2 = np.meshgrid(g1, g2)
        grid = np.tile(xb, (n_grid*n_grid, 1))
        grid[:, j1] = G1.ravel(); grid[:, j2] = G2.ravel()
        Z = sc(grid).reshape(n_grid, n_grid)
        surfaces[fn] = (g1, g2, Z)

    fns = [fn for fn in fns if fn in traces]
    if not fns:
        return None

    n_gen_max = max(len(traces[fn]) for fn in fns)
    ncols = min(3, len(fns)); nrows = -(-len(fns) // ncols)
    specs = [[{'type': 'surface'} for _ in range(ncols)] for _ in range(nrows)]
    titles = [f"F{fn} — ({dims[fn][2]},{dims[fn][3]}), {len(traces[fn])} gens" for fn in fns]
    fig = make_subplots(rows=nrows, cols=ncols, specs=specs, subplot_titles=titles,
                         horizontal_spacing=0.02, vertical_spacing=0.08)
    positions = [(r+1, c+1) for r in range(nrows) for c in range(ncols)][:len(fns)]

    def frame_at(fn, gen_idx):
        h = traces[fn][min(gen_idx, len(traces[fn]) - 1)]
        ai1, ai2, *_ = dims[fn]
        return h['population'][:, ai1], h['population'][:, ai2], -h['energies']

    for fn, (row, col) in zip(fns, positions):
        g1, g2, Z = surfaces[fn]
        fig.add_trace(go.Surface(x=g1, y=g2, z=Z, colorscale='rainbow', showscale=False,
                                  opacity=0.55, name=f'F{fn} acquisition surface'),
                      row=row, col=col)
        x0, y0, z0 = frame_at(fn, 0)
        fig.add_trace(go.Scatter3d(x=x0, y=y0, z=z0, mode='markers',
                                    marker=dict(size=4, color='#E91E63',
                                                line=dict(width=0.4, color='black')),
                                    name=f'F{fn} population', showlegend=False),
                      row=row, col=col)

    n_fns = len(fns)
    frames = []
    for g in range(n_gen_max):
        frame_data = []
        for fn in fns:
            x, y, z = frame_at(fn, g)
            frame_data.append(go.Scatter3d(x=x, y=y, z=z))
        # trace indices: surface, pop, surface, pop, ... -> population traces are odd indices
        pop_trace_idx = [2*i + 1 for i in range(n_fns)]
        frames.append(go.Frame(data=frame_data, traces=pop_trace_idx, name=str(g+1)))
    fig.frames = frames

    slider = [dict(active=0, currentvalue=dict(prefix="generation: "),
                   steps=[dict(method="animate", label=str(g+1),
                                args=[[str(g+1)], dict(mode="immediate",
                                                        frame=dict(duration=0, redraw=True),
                                                        transition=dict(duration=0))])
                          for g in range(n_gen_max)])]

    for i, fn in enumerate(fns):
        _, _, d1n, d2n = dims[fn]
        scene_key = 'scene' if i == 0 else f'scene{i+1}'
        fig.update_layout(**{scene_key: dict(
            xaxis_title=d1n, yaxis_title=d2n, zaxis_title=f'{acquisition} score',
            aspectmode='cube', camera=dict(projection=dict(type='orthographic')))})

    fig.update_layout(
        title=f"DE climbing the {acquisition} acquisition surface — shared slider. "
              "F2/F3 (exactly 2 active dims) sit exactly ON the surface; others can "
              "float off it (they're moving in dims this view can't show — see docstring)",
        sliders=slider, height=460*nrows, margin=dict(t=110, b=10))

    if out_path:
        fig.write_html(out_path, include_plotlyjs='cdn')
        print(f"Saved: {out_path}")
    return fig


def plot_trust_regions(all_X, df_best, fns=None):
    """Plots trust-region bars for for every dimension in the function currently 
    using turbo/turbo_mb"""
    if fns is None:
        fns = [fn for fn in df_best.index if df_best.loc[fn, 'strategy'] in ('turbo', 'turbo_mb')]
    if not fns:
        print("No function is currently using turbo/turbo_mb -- nothing to plot.")
        return
    fig, axes = plt.subplots(1, len(fns), figsize=(5*len(fns), 4), squeeze=False)
    axes = axes[0]
    for ax, fn in zip(axes, fns):
        lo, hi = TURBO_STATE[fn]["lo"], TURBO_STATE[fn]["hi"]
        xb = TURBO_STATE[fn]["xb"]; nd = len(xb)
        dims = [f"x{i+1}" for i in range(nd)]
        X = all_X[fn-1].values
        ax.barh(dims, hi-lo, left=lo, color='#2196F3', alpha=.4, edgecolor='k', zorder=1)
        for j in range(nd):
            ax.scatter(X[:, j], [dims[j]]*len(X), color='grey', s=25, alpha=.6, zorder=3,
                       label='observed' if j == 0 else None)
        for j in range(nd):
            ax.scatter([xb[j]], [dims[j]], color='red', s=70, zorder=5,
                       label='incumbent $x_b$' if j == 0 else None, edgecolors='k', linewidths=.5)
        ax.set_xlim(0, 1); ax.set_title(f'F{fn} trust region  (L={TURBO_STATE[fn]["length"]:.3f})')
        ax.legend(fontsize=8)
    plt.tight_layout(); plt.show()


def force_front(collection):
    """Forces the best point on the trop in a 3D scatter collection."""
    orig_proj = collection.do_3d_projection
    def patched():
        orig_proj(); return -1e9
    collection.do_3d_projection = patched


def safe_convex_hull(pts):
    """Wraps ConvexHull with a fallback (QJ option) so near-degenerate point sets don't crash it."""
    try:
        return ConvexHull(pts)
    except QhullError:
        try:
            return ConvexHull(pts, qhull_options="QJ")
        except QhullError:
            return None


def plot_clusters(all_X, all_y, out_path=None, k_max=4):
    """Plots k-means clusters per function shown as shaded convex-hullls (clouds)
    -point clusters (k-means) per function as shaded 'cloud', coloured by
    cluster-mean y. The points are coloured by ther individual y-values."""
    n_funcs = len(all_X)
    n_cols = min(4, n_funcs); n_rows = -(-n_funcs // n_cols)
    cmap = plt.cm.rainbow
    fig = plt.figure(figsize=(5.5*n_cols, 5*n_rows))
    for i in range(n_funcs):
        fn_num = i + 1
        X = np.asarray(all_X[i]); Y = np.asarray(all_y[i]).ravel()
        n, d = X.shape
        Xs = StandardScaler().fit_transform(X)
        k = max(2, min(k_max, n // 8))
        labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(Xs)
        ymin, ymax = Y.min(), Y.max(); norm = plt.Normalize(vmin=ymin, vmax=ymax)
        is_2d = d == 2
        if is_2d:
            dim_idx = [0, 1]; ax = fig.add_subplot(n_rows, n_cols, i+1)
        elif d == 3:
            dim_idx = [0, 1, 2]; ax = fig.add_subplot(n_rows, n_cols, i+1, projection="3d")
        else:
            var = Xs.var(axis=0); dim_idx = list(np.argsort(var)[::-1][:3])
            ax = fig.add_subplot(n_rows, n_cols, i+1, projection="3d")
        if not is_2d:
            ax.xaxis.pane.set_facecolor((0.95, 0.95, 0.95, 1.0))
            ax.yaxis.pane.set_facecolor((0.95, 0.95, 0.95, 1.0))
            ax.zaxis.pane.set_facecolor((0.92, 0.92, 0.92, 1.0))
            ax.xaxis.pane.set_edgecolor("gray"); ax.yaxis.pane.set_edgecolor("gray")
            ax.zaxis.pane.set_edgecolor("gray"); ax.grid(False)
        X0 = X[:, dim_idx[0]]; Y0 = X[:, dim_idx[1]]
        Z0 = X[:, dim_idx[2]] if not is_2d else None
        best_idx = np.argmax(Y)
        for c in range(k):
            mask = labels == c; cluster_mean_y = Y[mask].mean(); color = cmap(norm(cluster_mean_y))
            if is_2d:
                pts = np.column_stack([X0[mask], Y0[mask]])
                if mask.sum() >= 3:
                    hull = safe_convex_hull(pts)
                    if hull is not None:
                        ax.add_patch(Polygon(pts[hull.vertices], closed=True, facecolor=color,
                                              alpha=0.35, edgecolor=color, linewidth=1.8, zorder=1))
                    else:
                        ax.plot(pts[:, 0], pts[:, 1], color=color, alpha=0.5, lw=6, zorder=1)
                elif mask.sum() == 2:
                    ax.plot(pts[:, 0], pts[:, 1], color=color, alpha=0.5, lw=6, zorder=1)
            else:
                pts = np.column_stack([X0[mask], Y0[mask], Z0[mask]])
                if mask.sum() >= 4:
                    hull = safe_convex_hull(pts)
                    if hull is not None:
                        faces = [pts[simplex] for simplex in hull.simplices]
                        ax.add_collection3d(Poly3DCollection(faces, facecolor=color, edgecolor=color,
                                                              alpha=0.28, linewidths=0.4))
                    else:
                        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color, alpha=0.5, lw=5)
                elif mask.sum() >= 2:
                    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color, alpha=0.5, lw=5)
        for c in range(k):
            mask = labels == c; idxs = np.where(mask)[0]; rest = idxs[idxs != best_idx]
            if len(rest) == 0: continue
            common_kwargs = dict(cmap="rainbow", edgecolors="k", vmin=ymin, vmax=ymax)
            if is_2d:
                sc = ax.scatter(X0[rest], Y0[rest], s=140, c=Y[rest], marker="o",
                                 linewidths=0.6, alpha=1.0, zorder=3, **common_kwargs)
            else:
                sc = ax.scatter(X0[rest], Y0[rest], Z0[rest], s=140, c=Y[rest], marker="o",
                                 linewidths=0.6, alpha=1.0, depthshade=False, zorder=3, **common_kwargs)
        common_kwargs = dict(cmap="rainbow", edgecolors="black", vmin=ymin, vmax=ymax)
        if is_2d:
            ax.scatter(X0[best_idx], Y0[best_idx], s=140, c=[Y[best_idx]], marker="o",
                       linewidths=2.5, alpha=1.0, zorder=10, **common_kwargs)
        else:
            sc_best = ax.scatter(X0[best_idx], Y0[best_idx], Z0[best_idx], s=140, c=[Y[best_idx]],
                                  marker="o", linewidths=2.5, alpha=1.0, depthshade=False,
                                  zorder=10, **common_kwargs)
            force_front(sc_best)
        dim_note = f"x{dim_idx[0]+1}, x{dim_idx[1]+1}" + ("" if is_2d else f", x{dim_idx[2]+1}")
        hd_note = " (top-var dims)" if d > 3 else ""
        ax.set_title(f"F{fn_num} (d={d}, n={n})\n{dim_note}{hd_note}", fontsize=10, fontweight="bold")
        ax.set_xlabel(f"x{dim_idx[0]+1} (raw)", fontsize=8); ax.set_ylabel(f"x{dim_idx[1]+1} (raw)", fontsize=8)
        if not is_2d: ax.set_zlabel(f"x{dim_idx[2]+1} (raw)", fontsize=8)
        cbar = plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.1); cbar.set_label("y", fontsize=8)
    fig.suptitle("Cloud (hull) shade = cluster MEAN y  |  point color = that point's own y  |  "
                 "black outline = best point\n(same rainbow scale for both, so cloud vs. point "
                 "color can be compared directly)", fontsize=12, y=1.02)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=140, bbox_inches="tight"); print(f"Saved: {out_path}")
    plt.show()


def plot_smart(all_X, all_y, df_best, top_dims, fn):
    """Plots 2D/3D/4D projection onto the function's top consensus dimensions,
    coloured by y, with x_next marked as a black star."""
    X = all_X[fn-1]; y = all_y[fn-1].values; nd = X.shape[1]
    dims = top_dims[fn]; pd_ = dims if nd <= 4 else dims[:4]; npl = len(pd_)
    norm = plt.Normalize(y.min(), y.max()); cmap = plt.cm.rainbow
    br = df_best.loc[fn]; cm_ = dict(zip(X.columns.tolist(), br['x_next']))
    sc_ = {c: cm_[c] for c in pd_}
    lbl = f"x_next/{br['acquisition']} (pred={br['predicted_y']:.3g})"
    fig = plt.figure(figsize=(8, 6))
    if npl <= 2:
        d1, d2 = pd_[0], pd_[1]; ax = fig.add_subplot(111)
        s = ax.scatter(X[d1], X[d2], c=y, cmap=cmap, norm=norm, s=80, edgecolors='k', linewidths=.3)
        ax.scatter(sc_[d1], sc_[d2], marker='*', s=340, color='k', zorder=6, label=lbl)
        ax.set_xlabel(d1); ax.set_ylabel(d2); fig.colorbar(s, ax=ax, label='y')
    elif npl == 3:
        d1, d2, d3 = pd_; ax = fig.add_subplot(111, projection='3d')
        s = ax.scatter(X[d1], X[d2], X[d3], c=y, cmap=cmap, norm=norm, s=80, edgecolors='k', linewidths=.3)
        ax.scatter(sc_[d1], sc_[d2], sc_[d3], marker='*', s=340, color='k', zorder=6, label=lbl)
        ax.set_xlabel(d1); ax.set_ylabel(d2); ax.set_zlabel(d3); fig.colorbar(s, ax=ax, label='y', pad=.1)
    else:
        d1, d2, d3, d4 = pd_; ax = fig.add_subplot(111, projection='3d')
        sz = 60 + 200*(X[d4]-X[d4].min())/(X[d4].max()-X[d4].min()+1e-9)
        s = ax.scatter(X[d1], X[d2], X[d3], c=y, cmap=cmap, norm=norm, s=sz, edgecolors='k', linewidths=.3)
        ax.scatter(sc_[d1], sc_[d2], sc_[d3], marker='*', s=340, color='k', zorder=6, label=lbl)
        ax.set_xlabel(d1); ax.set_ylabel(d2); ax.set_zlabel(d3)
        ax.set_title(f'F{fn} — size→{d4}', fontsize=9); fig.colorbar(s, ax=ax, label='y', pad=.1)
    ax.legend(fontsize=8); fig.suptitle(f'F{fn} — {nd}D (top dims {pd_})', fontweight='bold')
    plt.tight_layout(); plt.show()


def plot_all_smart(all_X, all_y, df_best, top_dims, fns=range(1, 9)):
    """Loops plot_smart()) over every function"""
    for fn in fns:
        plot_smart(all_X, all_y, df_best, top_dims, fn)


def plot_parallel(all_X, all_y, df_best, fn, clip_bottom=False):
    """Plots parallel-coordinates plot of all observed points for one function, with 
    x_next overlaid as a bold line"""
    X = all_X[fn-1]; y = all_y[fn-1].values; dims = X.columns.tolist(); nd = len(dims)
    yfn = np.delete(y, np.argmin(y)) if (clip_bottom and len(y) > 1) else y
    norm = mcolors.Normalize(yfn.min(), yfn.max()); cmap = cm.rainbow
    br = df_best.loc[fn]; xn = np.array([float(v) for v in br['submission'].split('-')])
    fig, ax = plt.subplots(figsize=(max(8, nd*1.8), 5)); xp = np.arange(nd)
    for i in range(len(X)):
        ax.plot(xp, X.iloc[i].values, color=cmap(norm(np.clip(y[i], yfn.min(), yfn.max()))), lw=1.2)
    ax.plot(xp, xn, 'k-', lw=3, zorder=5, label=f"x_next/{br['acquisition']} (pred={br['predicted_y']:.3g})")
    ax.set_xticks(xp); ax.set_xticklabels(dims); ax.set_ylim(-.05, 1.05); ax.legend(fontsize=9)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([]); fig.colorbar(sm, ax=ax, label='y')
    ax.set_title(f'F{fn} — parallel coords | bold=x_next ({br["acquisition"]})', fontweight='bold')
    plt.tight_layout(); plt.show()


def plot_all_parallel(all_X, all_y, df_best, fns=range(1, 9), clip_bottom_fns=(3,)):
    """Loops plot_parallel over every function."""
    for fn in fns:
        plot_parallel(all_X, all_y, df_best, fn, clip_bottom=(fn in clip_bottom_fns))


# =====================================================================
# 5. PROGRESS TRACKING
# =====================================================================

def plot_progress_grid(all_y, ni=(10, 10, 15, 30, 20, 20, 30, 40)):
    """Plots a curave for the "best-so-far" result from the full history since week."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    axes = axes.ravel()
    for i, ax in enumerate(axes):
        fn = i + 1; y = all_y[fn-1].values; n0 = ni[fn-1]
        rm_full = np.maximum.accumulate(y)
        it = np.arange(len(y)); sel = it >= n0; x_rel = it[sel] - n0
        ax.scatter(x_rel, y[sel], color='grey', s=20, alpha=.5, label='y')
        ax.plot(x_rel, rm_full[sel], color='blue', lw=2, label='best so far')
        ax.set_xlabel('BO round'); ax.set_ylabel('y')
        ax.set_title(f'F{fn} progress', fontweight='bold')
        ax.legend(fontsize=7); ax.grid(alpha=.3)
    fig.suptitle('BO progress (initial sampling cropped)', fontweight='bold')
    plt.tight_layout(); plt.show()


def print_best_summary(all_y, fns=range(1, 9)):
    """Prints best y for all functions"""
    print(f"{'Function':<10}{'Best y':>15}{'At idx':>10}")
    for fn in fns:
        y = all_y[fn-1].values
        print(f"F{fn:<9}{y.max():>15.4f}{int(np.argmax(y)):>10}")