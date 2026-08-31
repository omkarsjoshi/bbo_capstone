"""
bbo_pipeline.py — the BO pipeline engine for GP surrogate, warps, LOO-CV,
dimension importance/freezing, and the six suggestion strategies
(global / turbo / turbo_mb / needle / explore / corners) used in 
by run_all(). CONFIG section below dictates the strategy for each function
"""

import numpy as np, pandas as pd, itertools
from types import SimpleNamespace
from scipy.stats import norm, qmc, spearmanr
from scipy.optimize import differential_evolution
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import PowerTransformer

RNG = np.random.default_rng(42)

def fn_rng(fn, week=None):
    """Documents the configuration of settings for each function to decide how it 
    is routed to specific strategies."""
    return np.random.default_rng([42, int(fn), int(WEEK if week is None else week)])

EPS, UB, LS_UPPER = 1e-6, 1-1e-6, 20.0      
WEEK = 13                                   
REPORT_MODE   = True
REFIT_HYPERS  = REPORT_MODE   # refit GP hypers on every LOO fold (~12x slower)
COMPARE_LOO   = REPORT_MODE   # also compute the old warped-scale BEFORE column
PERM_REPEATS  = 30 if REPORT_MODE else 10
RF_TREES      = 300 if REPORT_MODE else 150
GP_RESTARTS   = None          # None -> max(5, 2*d);  int -> fixed
# ---------------------------------------------------------------


# =====================================================================
#  CONFIG  —  plain dicts. Anything you omit falls back to DEFAULTS.
# =====================================================================
DEFAULTS = dict(
    strategy      = "auto",   # auto | global | turbo | turbo_mb | needle | explore | corners
    warp          = None,     # None | "log" | "yeo"
    ls_floor      = 0.05,     # Matern lengthscale lower bound
    noise         = True,     # add WhiteKernel
    freeze        = True,     # allow dims to be frozen at all
    n_candidates  = 4000,
    nu            = 2.5,      # Matern smoothness
    override_frozen = (),     # 0-BASED dim indices forced frozen, e.g. (0, 2) = x1, x3
    # --- P2 multi-basin ---
    min_sep       = 1.0,      # ARD LENGTHSCALE UNITS since P4 (was raw, 0.25)
    basin_rank    = "y",      # "y" (signed) | "abs" (envelope)
    max_basins    = 3,
    basin_offset  = 0,        # shifts which anchor this WEEK lands on
    turbo_ti      = 0.4,      # initial trust-region length
    # --- Needle for F4 ---
    radius        = 0.20,     # local-GP data radius
    shrink        = 0.06,     # half-width of the needle box
    xi            = 0.001,    # EI exploration margin
    # --- Manual override ---
    force_x       = None,     # if set, skip strategy dispatch entirely and submit
                               # this point as-is 
)

CONFIG = {

    1: dict(strategy="turbo_mb", warp="yeo", nu=1.5,
            basin_rank="abs", min_sep=1.0, turbo_ti=0.09, max_basins=2),

    2: dict(warp="log", nu=2.5),

    3: dict(strategy="global", warp="yeo", nu=1.5),

    4: dict(strategy="needle", warp=None, nu=1.5,
            shrink=0.045, radius=0.10, xi=0.0),

    5: dict(strategy="corners", warp=None, nu=2.5,
            force_x=(EPS, UB, UB, UB)),               

    6: dict(warp=None, nu=1.5),

    7: dict(warp=None, nu=1.5, override_frozen=(0, 2)),

    8: dict(strategy="auto", warp="yeo", nu=2.5),
}


def get_cfg(fn):
    """Merges functions' overrides onto DEFAULTS and return attributes.
    """
    over = CONFIG.get(fn, {})
    bad = set(over) - set(DEFAULTS)
    if bad:
        raise KeyError(f"F{fn}: unknown config key(s) {sorted(bad)} — check spelling")
    return SimpleNamespace(**{**DEFAULTS, **over})


def show_config(fns=range(1, 9)):
    """Returns a dataframe of the configuration for the functions"""
    return pd.DataFrame([vars(get_cfg(f)) for f in fns],
                        index=pd.Index(list(fns), name="function"))


def build_kernel(nd, cfg):
    """ Builds Matern kernel and optional WhiteKernel for noise"""
    k = Matern(np.ones(nd), (cfg.ls_floor, LS_UPPER), nu=cfg.nu)
    return k + WhiteKernel(1e-3, (1e-6, 1.0)) if cfg.noise else k


def make_gp(X, Y, cfg):
    """Fits GP using the specified kernel"""
    gp = GaussianProcessRegressor(build_kernel(X.shape[1], cfg),
        alpha=1e-10 if cfg.noise else 1e-6,
        n_restarts_optimizer=GP_RESTARTS or max(5, 2*X.shape[1]),
        normalize_y=True, random_state=42)
    return gp.fit(X, Y)


# ---------- Warping ----------
def fit_warp(Ytr, w):
    """Fits a warp transform (log/yeo or None) on training y-values to obtain parameters"""
    if w == "log":
        s = 0.0 if Ytr.min() > 0 else 1 - Ytr.min(); return ("log", s)
    if w == "yeo":
        return ("yeo", PowerTransformer("yeo-johnson").fit(Ytr.reshape(-1, 1)))
    return (None, None)


def warp_apply(Y, m):
    """Applies fitted warp"""
    k, o = m
    if k == "log": return np.log(Y + o)
    if k == "yeo": return o.transform(Y.reshape(-1, 1)).ravel()
    return Y

def warp_fwd(Y, w):
    """Combines fitting and warping sequentially, called from Notebook"""
    m = fit_warp(Y, w); return warp_apply(Y, m), m

def warp_inv(y, m):
    """Inverts the warp before reporting"""
    k, o = m
    y = np.atleast_1d(np.asarray(y, float))
    if k == "log":
        return np.exp(np.clip(y, -700, 700)) - o
    if k == "yeo":
        lam = float(o.lambdas_[0])
        pole = _yeo_std_pole(o, lam)
        if pole is not None:
            if lam < 0:
                y = np.minimum(y, pole - 1e-8)
            elif lam > 2:
                y = np.maximum(y, pole + 1e-8)
        out = o.inverse_transform(y.reshape(-1, 1)).ravel()
        if not np.isfinite(out).all():
            fill = np.nanmin(out[np.isfinite(out)]) if np.isfinite(out).any() else 0.0
            out = np.nan_to_num(out, nan=fill, posinf=fill, neginf=fill)
        return out
    return y

def _yeo_std_pole(o, lam):
    """Computes Yeo-Johnson domain pole in standardised z-space that GP works with"""
    try:
        mean_, scale_ = float(o._scaler.mean_[0]), float(o._scaler.scale_[0])
    except Exception:
        mean_, scale_ = 0.0, 1.0
    if lam < 0:
        return ((-1.0 / lam) - mean_) / scale_
    if lam > 2:
        return ((1.0 / (2.0 - lam)) - mean_) / scale_
    return None


def matern_ls(gp):
    """Extracts fitted Matern lengthscales from GP"""
    k = gp.kernel_
    return np.atleast_1d(k.k1.length_scale if hasattr(k, "k1") else k.length_scale)


# ---------- Q1: group identical/near-identical rows for LOO ----------
def _loo_groups(X, decimals=9):
    """Assigns group id per row so near-duplicate rows are held out in LOO and 
    not split across teh folds"""
    _, inv = np.unique(np.round(X, decimals), axis=0, return_inverse=True)
    return inv.ravel()


def _warp_local_slope(m, z0, h=1e-3):
    """Calculates slope |dy/dz| at z0 for every left out fold on the fitted warp.
    A high value may indicate that a decently performing GP in warp space may show
    high residuals in unwarped space. The slope for the LOO fold that had the lowest R2
    is used for diagnostic purposes"""
    k, _ = m
    if k is None:
        return 1.0
    y_plus = warp_inv(z0 + h, m)[0]
    y_minus = warp_inv(z0 - h, m)[0]
    return abs(float(y_plus - y_minus) / (2 * h))


def _r2(y_true, y_pred):
    """Calculates r2"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2) + 1e-12
    return 1 - ss_res / ss_tot


def _warp_clipped(pw, m):
    """Flags when the GP's z-space output fell past the Yeo-Johnson domain pole 
    and got truncated to the boundary instead of genuinely inverted"""
    k, o = m
    pw = np.atleast_1d(pw)
    if k != "yeo":
        return np.zeros(len(pw), dtype=bool)
    lam = float(o.lambdas_[0])
    pole = _yeo_std_pole(o, lam)
    if pole is None:
        return np.zeros(len(pw), dtype=bool)
    if lam < 0:
        return pw >= pole - 1e-8
    if lam > 2:
        return pw <= pole + 1e-8
    return np.zeros(len(pw), dtype=bool)


def _robust_loo_diag(Yraw, p_raw, groups, ok, amp, clip, r2_gate=0.30, rho_gate=0.70):
    """Computes the diagnostics metrics that indicate whether GP is trustworthy
    """
    yt, yp = Yraw[ok], p_raw[ok]
    r2 = _r2(yt, yp) # R2

    sq_err = np.where(ok, (Yraw - p_raw) ** 2, -np.inf)
    grp_ids = np.unique(groups[ok])
    grp_err = {g: sq_err[(groups == g) & ok].sum() for g in grp_ids}
    worst_g, r2_trim1, amp_worst = None, np.nan, np.nan
    if len(grp_ids) > 1:
        worst_g = max(grp_err, key=grp_err.get)
        keep = ok & (groups != worst_g)
        if keep.sum() >= 3:
            r2_trim1 = _r2(Yraw[keep], p_raw[keep]) # R2 with worst fold removed
        amp_worst = float(np.nanmean(amp[groups == worst_g])) # dy/dz for worst forld

    rho = float(spearmanr(yt, yp).correlation) if len(yt) >= 3 else np.nan # ρ is Spearman's r between
                                                                           # true and LOO-predicted y
    single_fold_driven = bool(
        r2 < r2_gate and np.isfinite(r2_trim1) and r2_trim1 >= r2_gate # Flags when (R2<0.30) ∧ (Rtrim1≥0.30) ∧ (ρ≥0.70)
        and np.isfinite(rho) and rho >= rho_gate
    )
    n_clipped = int(clip[ok].sum())  # If z falls beyond the pole, where y blows up
    clip_dominated = bool(
        r2 < r2_gate and n_clipped >= 2 and np.isfinite(rho) and rho >= rho_gate
    )
    return {"r2": r2, "r2_trim1": r2_trim1, "rho": rho, "worst_group": worst_g,
            "amp_at_worst_fold": amp_worst, "n_groups": int(len(np.unique(groups))),
            "n_points": int(len(groups)), "single_fold_driven": single_fold_driven,
            "n_clipped": n_clipped, "clip_dominated": clip_dominated}


# ---------- LOO CV: raw-scale score ----------
def loo_cv(X, Yraw, cfg, refit_hypers=None):
    """Clculates raw-scale LOO R2, per-fold warp
    """
    if refit_hypers is None:
        refit_hypers = REFIT_HYPERS

    ktuned = None
    if not refit_hypers:
        m0 = fit_warp(Yraw, cfg.warp)
        ktuned = GaussianProcessRegressor(build_kernel(X.shape[1], cfg),
            alpha=1e-10 if cfg.noise else 1e-6, n_restarts_optimizer=3,
            normalize_y=True, random_state=42).fit(X, warp_apply(Yraw, m0)).kernel_

    groups = _loo_groups(X)                          
    p_raw = np.full(len(Yraw), np.nan) # Raw outputs
    amp = np.full(len(Yraw), np.nan)
    clip = np.zeros(len(Yraw), dtype=bool)            
    for gid in np.unique(groups):
        te = np.where(groups == gid)[0]
        tr = np.where(groups != gid)[0]
        m = fit_warp(Yraw[tr], cfg.warp)          
        g = (GaussianProcessRegressor(build_kernel(X.shape[1], cfg),
                 alpha=1e-10 if cfg.noise else 1e-6,
                 n_restarts_optimizer=3, normalize_y=True, random_state=42)
             if refit_hypers else
             GaussianProcessRegressor(ktuned, optimizer=None,
                 alpha=1e-10 if cfg.noise else 1e-6, normalize_y=True,
                 random_state=42))
        g.fit(X[tr], warp_apply(Yraw[tr], m))
        pw = np.atleast_1d(g.predict(X[te]))
        p_raw[te] = warp_inv(pw, m) if cfg.warp else pw
        amp[te] = [_warp_local_slope(m, z) for z in pw] if cfg.warp else 1.0
        clip[te] = _warp_clipped(pw, m) if cfg.warp else False

    ok = np.isfinite(p_raw)
    if ok.sum() < 3:
        diag = {"r2": -np.inf, "r2_trim1": np.nan, "rho": np.nan, "worst_group": None,
                "amp_at_worst_fold": np.nan, "n_groups": int(len(np.unique(groups))),
                "n_points": int(len(groups)), "single_fold_driven": False,
                "n_clipped": int(clip.sum()), "clip_dominated": False}
        return p_raw, -np.inf, diag
    if ok.sum() < len(Yraw):
        print(f"    [loo_cv] dropped {(~ok).sum()}/{len(Yraw)} non-finite folds")
    diag = _robust_loo_diag(Yraw, p_raw, groups, ok, amp, clip)
    return p_raw, diag["r2"], diag


def loo_cv_warped(X, Yw, cfg):
    """R2 in warped space used for comparison in "BEFORE" column in the diagnostics"""
    p = np.zeros(len(Yw))
    for tr, te in LeaveOneOut().split(X):
        g = GaussianProcessRegressor(build_kernel(X.shape[1], cfg),
            alpha=1e-10 if cfg.noise else 1e-6, n_restarts_optimizer=3,
            normalize_y=True, random_state=42).fit(X[tr], Yw[tr])
        p[te] = g.predict(X[te])
    r2 = 1 - np.sum((Yw-p)**2) / (np.sum((Yw-Yw.mean())**2) + 1e-12)
    return p, r2


def _n01(v):
    """Used in dimension importance to normalise raw importance signals"""
    v = np.clip(np.asarray(v, float), 0, None); return v/v.max() if v.max() > 0 else v


def gp_grad_sens(gp, nd, ns=512, h=1e-3):
    """Estimates each dimension's sensitivity through GP's gradient"""
    Xs = np.clip(qmc.Sobol(d=nd, seed=42).random(ns), EPS, UB)
    base = gp.predict(Xs); g = np.zeros(nd)
    for j in range(nd):
        Xp = Xs.copy(); Xp[:, j] = np.clip(Xp[:, j]+h, EPS, UB)
        g[j] = np.sqrt(np.mean(((gp.predict(Xp)-base)/h)**2))
    return g


def diagnose_saturation(gp, noise_thresh=0.02):
    """Flags a dimension as multimodal when the lengthscale saturates at upper bound
    and the noise through the WhiteKernel term is higher than 0.02 threshold"""
    k = gp.kernel_
    ls = np.atleast_1d(k.k1.length_scale if hasattr(k, "k1") else k.length_scale)
    noise = k.k2.noise_level if hasattr(k, "k2") else 0.0
    saturated = ls >= 0.5 * LS_UPPER
    return {"lengthscales": ls, "noise": float(noise),
            "saturated_dims": np.where(saturated)[0],
            "multimodal_flag": bool(saturated.any() and noise > noise_thresh)}


def guard_freeze(frozen, sat):
    """Never freeze a dimension flagged as multimodal even if it has with high lengthscale."""
    frozen = np.asarray(frozen).copy()
    if sat["multimodal_flag"]:
        frozen[sat["saturated_dims"]] = False
    return frozen


def dimension_importance(X, Y, gp, cfg):
    """Blends ARD/gradient (GP family) with permutation/SHAP (RF family) into a consensus 
    vs disagreement score, and decides which dimensions to freeze."""
    nd = X.shape[1]; ls = matern_ls(gp)
    ard = _n01(1/np.clip(ls, 1e-6, None)); grad = _n01(gp_grad_sens(gp, nd))
    rf = RandomForestRegressor(RF_TREES, random_state=42, n_jobs=-1).fit(X, Y)
    perm = _n01(permutation_importance(rf, X, Y, n_repeats=PERM_REPEATS,
                                       random_state=42, n_jobs=-1).importances_mean)
    try:
        import shap
        sv = shap.TreeExplainer(rf).shap_values(X, check_additivity=False)
        shp = _n01(np.abs(sv).mean(0))
    except Exception:
        shp = np.full(nd, np.nan)
    gp_fam = np.nanmean(np.vstack([ard, grad]), 0)
    rf_fam = np.nanmean(np.vstack([perm, shp]), 0)
    cons = np.nanmean(np.vstack([gp_fam, rf_fam]), 0)
    disagree = np.abs(gp_fam - rf_fam)

    sat = diagnose_saturation(gp)
    saturated = ls >= 0.5*LS_UPPER
    frozen = saturated & (gp_fam < 0.15) & (rf_fam < 0.15)
    for j in cfg.override_frozen: frozen[j] = True
    if not cfg.freeze: frozen[:] = False
    frozen = guard_freeze(frozen, sat)                      # P1

    return pd.DataFrame({"dim":[f"x{j+1}" for j in range(nd)], "ard":np.round(ard,3),
        "gp_grad":np.round(grad,3), "rf_perm":np.round(perm,3), "shap":np.round(shp,3),
        "gp_fam":np.round(gp_fam,3), "rf_fam":np.round(rf_fam,3),
        "consensus":np.round(cons,3), "disagree":np.round(disagree,3),
        "lengthscale":np.round(ls,2), "saturated":saturated, "frozen":frozen})


def aEI(m,s,ym,ys,xi=0.01): # Expected Improvement
    x=xi*max(ys,1e-6); z=(m-ym-x)/(s+1e-9) 
    e=(m-ym-x)*norm.cdf(z)+s*norm.pdf(z); e[s<1e-10]=0; return e
    
def aUCB(m,s,t,d): # Upper Confidence Bound
    return m+min(np.sqrt(2*np.log(d*t**2*np.pi**2/3.0)),3.0)*s

def aPI(m,s,ym,ys,xi=0.01): # Probability of Improvement
    x=xi*max(ys,1e-6); return norm.cdf((m-ym-x)/(s+1e-9))


def _opt_active(score, anchor, active, ps=20, mi=50):
    """Maximises an acquisition function over only the "active" (non-frozen) dimensions
    via differential evolution, holding frozen dims at the incumbent level (anchor)"""
    nd = len(anchor)
    if not active: return anchor.copy()
    def neg(x):
        x = np.atleast_2d(x)
        if x.shape[0] == len(active) and x.shape[1] != len(active): x = x.T
        full = np.tile(anchor, (x.shape[0], 1)); full[:, active] = x
        s = -np.atleast_1d(score(full)); return s if s.shape[0] > 1 else s[0]
    r = differential_evolution(neg, [(EPS,UB)]*len(active), vectorized=True,
        updating="deferred", popsize=ps, maxiter=mi, tol=1e-7, seed=42, polish=True)
    out = anchor.copy(); out[active] = np.clip(r.x, EPS, UB); return out


def suggest_global(X, Y, gp, imp, cfg):
    nd=X.shape[1]; ys=Y.std(); ym=Y.max(); xb=X[np.argmax(Y)]
    active=[j for j in range(nd) if not imp["frozen"].iloc[j]]
    def mk(n):
        def sc(f):
            m,s=gp.predict(f,return_std=True)
            return aEI(m,s,ym,ys) if n=="EI" else (aUCB(m,s,len(Y),max(len(active),1)) if n=="UCB" else aPI(m,s,ym,ys))
        return sc
    return [(n, _opt_active(mk(n), xb, active), None) for n in ("EI","UCB","PI")]


def suggest_explore(X, Y, gp, imp, cfg, rng=None):
    """Optimises EI/UCB/PI globally over active dimensions and returns all three candidates"""
    rng = RNG if rng is None else rng
    nd=X.shape[1]; xb=X[np.argmax(Y)]
    C = rng.uniform(EPS, UB, (cfg.n_candidates, nd))
    for j in range(nd):
        if imp["frozen"].iloc[j]: C[:,j]=xb[j]
    d=np.min(np.linalg.norm(C[:,None,:]-X[None,:,:],axis=2),axis=1)
    return [("MD", np.clip(C[np.argmax(d)],EPS,UB), None)], None


def suggest_corners(fn, X, Y, cfg, rng=None):
    """Picks the box corner farthest from all observed points for vertex-optimum functions (F5)"""
    nd = X.shape[1]
    V = np.array(list(itertools.product([EPS, UB], repeat=nd)))
    d = np.min(np.linalg.norm(V[:, None, :] - X[None, :, :], axis=2), axis=1)
    return [("corner", V[np.argmax(d)], None)], None


TURBO_STATE = {}
NEEDLE_STATE = {}


def suggest_turbo(fn, X, Y, gp, imp, cfg, ti=None, tmin=0.05, tmax=0.8, st_=3, ft=3,
                  rng=None):
    """Fits single-basin TuRBO, with a trust region around the incumbent best that 
    grows/shrinks based on recent success/failure."""
    rng = RNG if rng is None else rng
    ti = cfg.turbo_ti if ti is None else ti
    tmin = min(tmin, ti)
    nd=X.shape[1]; ym=Y.max(); xb=X[np.argmax(Y)]
    s=TURBO_STATE.setdefault(fn,{"length":ti,"succ":0,"fail":0,"last":ym})
    if ym>s["last"]: s["succ"]+=1; s["fail"]=0
    else: s["fail"]+=1; s["succ"]=0
    if s["succ"]>=st_: s["length"]=min(tmax,s["length"]*2); s["succ"]=0
    elif s["fail"]>=ft: s["length"]=max(tmin,s["length"]/2); s["fail"]=0
    s["last"]=ym; L=s["length"]
    ls=np.clip(matern_ls(gp),1e-3,1.0)          # P1: cap so a saturated dim
    active_mask=~imp["frozen"].values           #     can't distort the box
    ls_ref=ls[active_mask].mean() if active_mask.any() else ls.mean()
    half=np.clip((L/2)*(ls/ls_ref),1e-6,0.5)
    lo=np.clip(xb-half,EPS,UB); hi=np.clip(xb+half,EPS,UB); sig=(hi-lo)/4
    s["lo"], s["hi"], s["xb"], s["half"] = lo, hi, xb, half
    s["basin"], s["n_basins"] = int(np.argmax(Y)), 1
    C=np.clip(xb+sig*rng.standard_normal((cfg.n_candidates,nd)),lo,hi)
    for j in range(nd):
        if imp["frozen"].iloc[j]: C[:,j]=xb[j]
    m,sd=gp.predict(C,return_std=True); draw=m+sd*rng.standard_normal(len(m))
    return [("TuRBO", np.clip(C[np.argmax(draw)],EPS,UB), None)], L



def find_basins(X, Y, min_sep=1.0, top_frac=0.5, max_basins=3, rank="y", ls=None):
    """Greedily finds separated high-value anchor points, measuring separation 
    weighted by ARD-lengthscale to identify the basins"""
    w = np.ones(X.shape[1]) if ls is None else np.clip(ls, 1e-3, LS_UPPER)
    score = np.abs(Y) if rank == "abs" else Y
    order = np.argsort(-score)
    n_keep = max(3, int(len(Y) * top_frac))
    anchors = []
    for i in order[:n_keep]:
        if all(np.linalg.norm((X[i] - X[a]) / w) >= min_sep for a in anchors):
            anchors.append(int(i))
        if len(anchors) >= max_basins:
            break
    return anchors


def suggest_turbo_multibasin(fn, X, Y, gp, imp, cfg, week=0,
                             ti=None, tmin=0.05, tmax=0.8, st_=3, ft=3,
                             rng=None):
    """TuRBO round-robin on the trust regions across basins identified"""
    rng = RNG if rng is None else rng
    ti = cfg.turbo_ti if ti is None else ti
    tmin = min(tmin, ti)
    nd = X.shape[1]
    anchors = find_basins(X, Y, min_sep=cfg.min_sep, rank=cfg.basin_rank,
                          max_basins=cfg.max_basins, ls=matern_ls(gp))   
    b = anchors[(week + cfg.basin_offset) % len(anchors)]                
    xb, ym = X[b], Y[b]

    s = TURBO_STATE.setdefault((fn, b),
                               {"length": ti, "succ": 0, "fail": 0, "last": ym})
    if ym > s["last"] + 1e-9: s["succ"] += 1; s["fail"] = 0
    else:                     s["fail"] += 1; s["succ"] = 0
    if s["succ"] >= st_:   s["length"] = min(tmax, s["length"]*2); s["succ"] = 0
    elif s["fail"] >= ft:  s["length"] = max(tmin, s["length"]/2); s["fail"] = 0
    s["last"] = ym; L = s["length"]

    ls = np.clip(matern_ls(gp), 1e-3, 1.0)           
    active = ~imp["frozen"].values
    ls_ref = ls[active].mean() if active.any() else ls.mean()
    half = np.clip((L/2)*(ls/ls_ref), 1e-6, 0.5)

    lo = np.clip(xb-half, EPS, UB); hi = np.clip(xb+half, EPS, UB)
    s["lo"], s["hi"], s["xb"], s["half"] = lo, hi, xb, half
    s["basin"], s["n_basins"] = b, len(anchors)
    TURBO_STATE[fn] = s                              

    sig = (hi-lo)/4
    C = np.clip(xb + sig*rng.standard_normal((cfg.n_candidates, nd)), lo, hi)
    for j in range(nd):
        if imp["frozen"].iloc[j]: C[:, j] = xb[j]
    m, sd = gp.predict(C, return_std=True)
    draw = m + sd*rng.standard_normal(len(m))
    return [("TuRBO-mb", np.clip(C[np.argmax(draw)], EPS, UB), None)], L


def local_gp(X, Y, centre, radius=0.20, min_pts=12, nu=2.5, ls_cap=0.25):
    """Fits a second, local GP on just the nearby points, with a much shorter lengthscale 
    cap than the global fit for needle refinement (F4)"""
    d = np.linalg.norm(X - centre, axis=1)
    m = d <= radius
    if m.sum() < min_pts:
        m = d <= np.sort(d)[min(min_pts, len(d)) - 1]
    Xl, Yl = X[m], Y[m]
    k = Matern(np.full(X.shape[1], 0.05), (1e-3, ls_cap), nu=nu) \
        + WhiteKernel(1e-4, (1e-8, 1e-1))
    gp = GaussianProcessRegressor(k, alpha=1e-10, n_restarts_optimizer=12,
                                  normalize_y=True, random_state=42).fit(Xl, Yl)
    return gp, int(m.sum())


def suggest_needle(fn, X, Y, cfg, centre=None, n_cand=20000, rng=None):
    """Tight EI inside a small box around the incumbent, on a LOCAL GP (F4)"""
    rng = RNG if rng is None else rng
    nd = X.shape[1]
    xb = X[np.argmax(Y)] if centre is None else np.asarray(centre, float)
    lgp, npts = local_gp(X, Y, xb, radius=cfg.radius, nu=cfg.nu)
    lo = np.clip(xb - cfg.shrink, EPS, UB); hi = np.clip(xb + cfg.shrink, EPS, UB)
    C = rng.uniform(lo, hi, size=(n_cand, nd))
    m, s = lgp.predict(C, return_std=True)
    ym, ys = Y.max(), Y.std()
    x_ = cfg.xi * max(ys, 1e-6)
    z = (m - ym - x_) / (s + 1e-9)
    ei = (m - ym - x_)*norm.cdf(z) + s*norm.pdf(z)
    ei[s < 1e-10] = 0
    NEEDLE_STATE[fn] = {"lo": lo, "hi": hi, "xb": xb, "n_local": npts,
                        "local_ls": np.atleast_1d(lgp.kernel_.k1.length_scale),
                        "length": float(cfg.shrink*2)}
    return [("EI-needle", np.clip(C[np.argmax(ei)], EPS, UB), lgp)], npts


def route(r2, imp):
    """Decides strategy from LOO R²: "explore" if R² is unusable, "global" if R² ≥ 0.30, else "turbo" """
    if not np.isfinite(r2):
        return "explore"
    if not (imp["consensus"] > 0.15).any():
        return "explore"
    return "global" if r2 >= 0.30 else "turbo"


def run_all(all_X, all_y, fns=range(1, 9), verbose=True, compare_loo=None):
    """
    Top-level orchestrator: 
    1) Fits GP 
    2) Warps output
    3) Runs LOO-CV
    4) Computes dimension importance
    5) Routes functions to appropriate strategy
    6) Displays diagnostic scores
    7) Records the suggested next point(s)
    """
    if compare_loo is None: compare_loo = COMPARE_LOO
    df_all_rows=[]; best_rows=[]; gps={}; diagnostics={}; top_dims={}; cmp_rows=[]
    for fn in fns:
        cfg=get_cfg(fn); Xdf,Ydf=all_X[fn-1],all_y[fn-1]          # P6
        X=Xdf.values; Yraw=Ydf.values; cols=Xdf.columns.tolist()
        Yw,wm=warp_fwd(Yraw,cfg.warp)
        gp=make_gp(X,Yw,cfg); gps[fn]=gp

        r2_before = loo_cv_warped(X,Yw,cfg)[1] if compare_loo else np.nan
        loo_p,r2,loo_diag = loo_cv(X,Yraw,cfg)

        imp=dimension_importance(X,Yw,gp,cfg)
        sat=diagnose_saturation(gp)
        diagnostics[fn]={"loo_r2":r2,"loo_r2_leaky_warpedscale":r2_before,
                         "loo_preds_raw":loo_p,"loo_true_raw":Yraw,
                         "loo_diag":loo_diag,
                         "importance":imp,"saturation":sat}
        top_dims[fn]=imp.sort_values("consensus",ascending=False)["dim"].tolist()

        strat=route(r2,imp) if cfg.strategy=="auto" else cfg.strategy
        would=route(r2,imp)
        cmp_rows.append({"function":fn,"d":X.shape[1],"warp":cfg.warp or "-","nu":cfg.nu,
            "loo_r2_leaky_warpedscale":round(r2_before,3),"loo_r2_leakproof_rawscale":round(r2,3),
            "delta":round(r2-r2_before,3),
            "loo_r2_trim1":round(loo_diag["r2_trim1"],3) if np.isfinite(loo_diag["r2_trim1"]) else "-",
            "loo_rho":round(loo_diag["rho"],3) if np.isfinite(loo_diag["rho"]) else "-",
            "amp_at_worst_fold":round(loo_diag["amp_at_worst_fold"],2) if np.isfinite(loo_diag["amp_at_worst_fold"]) else "-",
            "n_dup_groups":X.shape[0]-loo_diag["n_groups"],
            "n_clipped":f"{loo_diag['n_clipped']}/{loo_diag['n_points']}",
            "single_fold_driven":loo_diag["single_fold_driven"],
            "clip_dominated":loo_diag["clip_dominated"],
            "route_before":route(r2_before,imp) if np.isfinite(r2_before) else "-",
            "route_after":would,"strategy_used":strat,
            "multimodal":sat["multimodal_flag"]})

        tr=None; tr_bounds=None; note=""
        rng = fn_rng(fn)                       # independent stream per function
        if cfg.force_x is not None:
            rows=[("forced", np.clip(np.asarray(cfg.force_x,dtype=float),EPS,UB), None)]
            note="H1: manual override — see CONFIG comment"
        elif strat=="turbo":
            rows,tr=suggest_turbo(fn,X,Yw,gp,imp,cfg,rng=rng)
            tr_bounds=(TURBO_STATE[fn]["lo"].copy(), TURBO_STATE[fn]["hi"].copy())
        elif strat=="turbo_mb":
            rows,tr=suggest_turbo_multibasin(fn,X,Yw,gp,imp,cfg,week=WEEK,rng=rng)
            tr_bounds=(TURBO_STATE[fn]["lo"].copy(), TURBO_STATE[fn]["hi"].copy())
            note=f"basin idx {TURBO_STATE[fn]['basin']} of {TURBO_STATE[fn]['n_basins']}"
        elif strat=="needle":
            rows,npts=suggest_needle(fn,X,Yw,cfg,rng=rng)
            tr_bounds=(NEEDLE_STATE[fn]["lo"].copy(), NEEDLE_STATE[fn]["hi"].copy())
            note=f"{npts} local pts"
        elif strat=="explore":
            rows,tr=suggest_explore(X,Yw,gp,imp,cfg,rng=rng)
        elif strat=="corners":
            rows,tr=suggest_corners(fn,X,Yw,cfg,rng=rng)          # P5
            note="unvisited vertex"
        elif strat=="global":
            rows=suggest_global(X,Yw,gp,imp,cfg)
        else:
            raise ValueError(f"F{fn}: unknown strategy {strat!r}")

        frozen=imp.loc[imp["frozen"],"dim"].tolist()
        for acq,x,report_gp in rows:
            # Q7: report with whichever model actually chose the point --
            # needle uses a LOCAL gp internally; every other strategy
            # scores with the global one, so report_gp is None and this
            # falls back to gp exactly as before.
            m_rep = report_gp if report_gp is not None else gp
            predw=float(m_rep.predict(x[None])[0])
            pred=float(warp_inv(predw,wm)[0]) if cfg.warp else predw
            r={"function":fn,"dimensions":X.shape[1],"acquisition":acq,
               "predicted_y":round(pred,6),"y_max_so_far":round(float(Yraw.max()),6),
               "submission":"-".join(f"{v:.6f}" for v in x),
               "strategy":strat,"loo_r2":round(r2,3)}
            for i,c in enumerate(cols): r[f"{c}_suggested"]=round(float(x[i]),6)
            df_all_rows.append(r)

        def _rank_pred(row):
            acq,x,report_gp=row
            m_rep = report_gp if report_gp is not None else gp
            return float(m_rep.predict(x[None])[0])
        best=max(rows,key=_rank_pred)
        acq,x,report_gp=best
        m_rep = report_gp if report_gp is not None else gp
        predw=float(m_rep.predict(x[None])[0])
        pred=float(warp_inv(predw,wm)[0]) if cfg.warp else predw
        best_rows.append({"function":fn,"dimensions":X.shape[1],"acquisition":acq,
            "y_max_so_far":round(float(Yraw.max()),6),"predicted_y":round(pred,6),
            "submission":"-".join(f"{v:.6f}" for v in x),"x_next":[float(v) for v in x],
            "strategy":strat,"loo_r2":round(r2,3),
            "frozen_dims":frozen or "-","tr_length":round(tr,3) if tr else None,
            "tr_bounds":tr_bounds,"note":note})
        if verbose:
            mm = "MULTIMODAL" if sat["multimodal_flag"] else ""
            bf = f"{r2_before:+.3f} -> " if np.isfinite(r2_before) else ""
            dup = f"dup_x{X.shape[0]-loo_diag['n_groups']}" if loo_diag["n_groups"]<X.shape[0] else ""
            flag = ""
            if loo_diag["single_fold_driven"]:
                flag += "  [SINGLE-FOLD-DRIVEN — see loo_r2_trim1/loo_rho]"
            if loo_diag["clip_dominated"]:
                flag += f"  [POLE-CLIP ARTEFACT — {loo_diag['n_clipped']}/{loo_diag['n_points']} folds clipped, trust loo_rho instead]"
            print(f"F{fn}: {strat:9s} {acq:10s} "
                  f"LOO_R2 {bf}{r2:+.3f}  "
                  f"(trim1={loo_diag['r2_trim1']:+.3f} rho={loo_diag['rho']:+.3f} amp_worst={loo_diag['amp_at_worst_fold']:.1f}x clip={loo_diag['n_clipped']}/{loo_diag['n_points']})  "
                  f"frozen={str(frozen or '-'):10s} {mm:11s} {dup:7s} {note}{flag}")

    df_all=pd.DataFrame(df_all_rows)
    df_best=pd.DataFrame(best_rows); df_best.index=range(1,len(list(fns))+1)
    df_best.index.name="function"
    df_cmp=pd.DataFrame(cmp_rows).set_index("function")
    return all_X, all_y, df_all, df_best, gps, top_dims, diagnostics, df_cmp


import pickle, os

CACHE_PATH = f"w{WEEK}_run.pkl"                

def run_cached(force=False, path=CACHE_PATH, **kw):
    """Saves the results to a file and reuses it next time instead of recomputing if 
    "force=False". The execution of full pipelien is forced through "force=True" """
    if not force and os.path.exists(path):
        print(f"[cache] loaded {path}  (run_cached(force=True) to recompute)")
        with open(path, "rb") as f:
            return pickle.load(f)
    res = run_all(**kw)
    try:
        with open(path, "wb") as f:
            pickle.dump(res, f)
        print(f"[cache] wrote {path}")
    except OSError as e:
        print(f"[cache] WARNING: could not write {path} ({e}) -- "
              f"continuing without caching. Your result below is complete "
              f"and correct either way; it just won't be reloadable via "
              f"run_cached() without recomputing next time. If this keeps "
              f"happening: check whether {path} already exists and is "
              f"read-only/locked (common in OneDrive-synced folders), or "
              f"pass an explicit path= somewhere you have write access.")
    return res