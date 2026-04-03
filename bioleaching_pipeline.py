"""
Kinetic Modelling and Machine Learning Optimisation
of Cyanide-Free Gold Bioleaching with Pseudomonas fluorescens

Two-stage pipeline:
  Stage 1 — Fit a logistic kinetic model to each experimental time-series curve.
             Extract biologically meaningful parameters per condition.
  Stage 2 — Train ML regressors on those kinetic parameters to predict and
             optimise gold extraction under unseen process conditions.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, cross_val_score, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance
import shap
import os

# ── output directory ──────────────────────────────────────────────────────────
os.makedirs("outputs", exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})

TEAL   = "#1D9E75"
CORAL  = "#D85A30"
PURPLE = "#7F77DD"
AMBER  = "#EF9F27"
GRAY   = "#888780"
BLUE   = "#378ADD"

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 0 — DATA INGESTION AND RESHAPING
# ══════════════════════════════════════════════════════════════════════════════

def load_and_reshape(path: str) -> pd.DataFrame:
    """
    Read all five experimental sheets and convert the wide pivot-table format
    into a tidy long-form dataframe where every row is one (condition, timepoint)
    observation. Controls are excluded — they contain no bacterial activity.

    Fixed defaults for each OFAT sheet are inferred from the experimental design:
      pH sheet      → inoculum 6.67 %v/v, pulp 1.0 %w/v, glycine 0.133 mol/L, ratio 0.25
      inoculum sheet → pH 9, pulp 1.0 %w/v, glycine 0.133 mol/L, ratio 0.25
      pulp sheet    → pH 9, inoculum 6.67 %v/v, glycine 0.133 mol/L, ratio 0.25
      glycine sheet → pH 9, inoculum 6.67 %v/v, pulp 1.0 %w/v, ratio 0.25
      substrate sheet→ pH 9, inoculum 6.67 %v/v, pulp 0.33 %w/v, glycine 0.133 mol/L
    """
    sheets = pd.read_excel(path, sheet_name=None)
    times  = [0, 8, 16, 24, 32, 40, 52, 64]
    rows   = []

    # ── Sheet 1: pH variation ─────────────────────────────────────────────────
    ph_map = {"pH=7": 7, "pH=8": 8, "pH=9": 9}
    for _, row in sheets["different pH"].iterrows():
        label = str(row.iloc[0]).strip()
        if label in ph_map:
            for i, t in enumerate(times):
                rows.append({
                    "pH": ph_map[label],
                    "inoculum_pct": 6.67,
                    "pulp_density_pct": 1.0,
                    "glycine_mol_L": 0.133,
                    "substrate_ratio": 0.25,
                    "time_h": t,
                    "gold_pct": float(row.iloc[i + 1]),
                })

    # ── Sheet 2: inoculum variation ───────────────────────────────────────────
    inoc_map = {"0.67%v/v": 0.67, "3.33%v/v": 3.33,
                "6.67%v/v": 6.67, "13.33%v/v": 13.33}
    for _, row in sheets["different additions of bacteria"].iterrows():
        label = str(row.iloc[0]).strip()
        if label in inoc_map:
            for i, t in enumerate(times):
                rows.append({
                    "pH": 9,
                    "inoculum_pct": inoc_map[label],
                    "pulp_density_pct": 1.0,
                    "glycine_mol_L": 0.133,
                    "substrate_ratio": 0.25,
                    "time_h": t,
                    "gold_pct": float(row.iloc[i + 1]),
                })

    # ── Sheet 3: pulp density variation ──────────────────────────────────────
    pulp_map = {"0.33%w/v": 0.33, "0.67%w/v": 0.67, "1%w/v": 1.0,
                "1.33%w/v": 1.33, "1.67%w/v": 1.67}
    for _, row in sheets["different pulp densities"].iterrows():
        label = str(row.iloc[0]).strip()
        if label in pulp_map:
            for i, t in enumerate(times):
                rows.append({
                    "pH": 9,
                    "inoculum_pct": 6.67,
                    "pulp_density_pct": pulp_map[label],
                    "glycine_mol_L": 0.133,
                    "substrate_ratio": 0.25,
                    "time_h": t,
                    "gold_pct": float(row.iloc[i + 1]),
                })

    # ── Sheet 4: glycine concentration variation ──────────────────────────────
    gly_map = {"0.013 mol/L": 0.013, "0.067 mol/L": 0.067,
               "0.133 mol/L": 0.133, "0.267 mol/L": 0.267}
    for _, row in sheets["different glycine concentration"].iterrows():
        label = str(row.iloc[0]).strip()
        if label in gly_map:
            for i, t in enumerate(times):
                rows.append({
                    "pH": 9,
                    "inoculum_pct": 6.67,
                    "pulp_density_pct": 1.0,
                    "glycine_mol_L": gly_map[label],
                    "substrate_ratio": 0.25,
                    "time_h": t,
                    "gold_pct": float(row.iloc[i + 1]),
                })

    # ── Sheet 5: substrate ratio variation ────────────────────────────────────
    sub_map = {"M:G=1:10": 1/10, "M:G=1:8": 1/8,
               "M:G=1:4": 1/4,  "M:G=1:2": 1/2}
    for _, row in sheets["different substrates"].iterrows():
        label = str(row.iloc[0]).strip()
        if label in sub_map:
            for i, t in enumerate(times):
                rows.append({
                    "pH": 9,
                    "inoculum_pct": 6.67,
                    "pulp_density_pct": 0.33,
                    "glycine_mol_L": 0.133,
                    "substrate_ratio": sub_map[label],
                    "time_h": t,
                    "gold_pct": float(row.iloc[i + 1]),
                })

    df = pd.DataFrame(rows)
    print(f"[Stage 0] Tidy dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"          Unique experimental conditions: "
          f"{df.drop_duplicates(subset=['pH','inoculum_pct','pulp_density_pct','glycine_mol_L','substrate_ratio']).shape[0]}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — KINETIC CURVE FITTING
# ══════════════════════════════════════════════════════════════════════════════

def logistic_model(t, E_max, k, t_half):
    """
    Three-parameter logistic (sigmoid) kinetic model for leaching:

        E(t) = E_max / (1 + exp(-k * (t - t_half)))

    Parameters
    ----------
    E_max   : maximum gold extraction asymptote (%)
    k       : rate constant (h⁻¹)  — steepness of the sigmoid
    t_half  : inflection time (h)   — when extraction = E_max / 2
    """
    return E_max / (1.0 + np.exp(-k * (t - t_half)))


def fit_kinetics(df: pd.DataFrame) -> pd.DataFrame:
    """
    For every unique experimental condition, fit the logistic model to the
    time-series data using non-linear least squares (scipy curve_fit).
    Returns a condensed dataframe: one row per condition, columns are the
    three kinetic parameters plus R² and the five process variables.
    """
    condition_cols = ["pH", "inoculum_pct", "pulp_density_pct",
                      "glycine_mol_L", "substrate_ratio"]
    records = []

    for keys, grp in df.groupby(condition_cols):
        t   = grp["time_h"].values.astype(float)
        E   = grp["gold_pct"].values.astype(float)

        # initial guesses: E_max near observed max, k~0.05, t_half~30h
        p0     = [E.max() * 1.1, 0.05, 30.0]
        bounds = ([0, 1e-4, 0], [100, 2.0, 200])

        try:
            popt, _ = curve_fit(logistic_model, t, E, p0=p0,
                                bounds=bounds, maxfev=10000)
            E_pred  = logistic_model(t, *popt)
            r2      = r2_score(E, E_pred)
            record  = dict(zip(condition_cols, keys))
            record.update({"E_max": popt[0], "k": popt[1],
                           "t_half": popt[2], "r2_fit": r2})
            records.append(record)
        except RuntimeError:
            print(f"  [!] Fit failed for condition: {dict(zip(condition_cols, keys))}")

    kinetics = pd.DataFrame(records)
    print(f"\n[Stage 1] Kinetic fits completed: {len(kinetics)} conditions")
    print(f"          Mean R² of fits: {kinetics['r2_fit'].mean():.4f}")
    print(f"          Min  R² of fits: {kinetics['r2_fit'].min():.4f}")
    return kinetics


def plot_kinetic_fits(df: pd.DataFrame, kinetics: pd.DataFrame):
    """
    Grid of subplots — one panel per experimental condition.
    Scatter: observed data. Line: fitted logistic curve.
    Annotated with E_max, k, R².
    """
    condition_cols = ["pH", "inoculum_pct", "pulp_density_pct",
                      "glycine_mol_L", "substrate_ratio"]
    n   = len(kinetics)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3.5))
    axes = axes.flatten()
    t_smooth = np.linspace(0, 70, 300)

    for idx, (_, krow) in enumerate(kinetics.iterrows()):
        ax   = axes[idx]
        cond = {c: krow[c] for c in condition_cols}
        mask = pd.Series([True] * len(df))
        for c, v in cond.items():
            mask &= (df[c] == v)
        grp  = df[mask]

        ax.scatter(grp["time_h"], grp["gold_pct"],
                   color=TEAL, s=55, zorder=5, label="Observed")
        E_fit = logistic_model(t_smooth, krow["E_max"], krow["k"], krow["t_half"])
        ax.plot(t_smooth, E_fit, color=CORAL, lw=2, label="Logistic fit")

        # Determine what varied in this condition for the title
        if   cond["pH"] in [7, 8]:            title = f"pH = {int(cond['pH'])}"
        elif cond["inoculum_pct"] != 6.67:    title = f"Inoc = {cond['inoculum_pct']}%"
        elif cond["pulp_density_pct"] != 1.0: title = f"Pulp = {cond['pulp_density_pct']}%"
        elif cond["glycine_mol_L"] != 0.133:  title = f"Gly = {cond['glycine_mol_L']} M"
        elif cond["substrate_ratio"] != 0.25: title = f"M:G = 1:{int(1/cond['substrate_ratio'])}"
        else:                                  title = "pH9 default"

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Time (h)", fontsize=9)
        ax.set_ylabel("Au extraction (%)", fontsize=9)
        ax.set_ylim(0, 65)
        ax.annotate(
            f"$E_{{max}}$={krow['E_max']:.1f}%\n"
            f"$k$={krow['k']:.3f} h⁻¹\n"
            f"$R²$={krow['r2_fit']:.3f}",
            xy=(0.97, 0.05), xycoords="axes fraction",
            ha="right", va="bottom", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
        )

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle("Logistic Kinetic Model Fits — All Experimental Conditions",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("outputs/01_kinetic_fits.png", bbox_inches="tight")
    plt.close()
    print("[Stage 1] Saved → outputs/01_kinetic_fits.png")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — EXPLORATORY ANALYSIS OF KINETIC PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

def plot_parameter_sensitivity(kinetics: pd.DataFrame):
    """
    Four strip/bar charts showing how E_max and k respond to each process
    variable independently. This replaces a traditional EDA correlation
    heatmap — more meaningful with only 20 data points.
    """
    process_vars = {
        "pH":                 ("pH",                  [7, 8, 9]),
        "Inoculum (%v/v)":    ("inoculum_pct",        [0.67, 3.33, 6.67, 13.33]),
        "Pulp density (%w/v)":("pulp_density_pct",    [0.33, 0.67, 1.0, 1.33, 1.67]),
        "Glycine (mol/L)":    ("glycine_mol_L",       [0.013, 0.067, 0.133, 0.267]),
        "M:G ratio":          ("substrate_ratio",     [0.1, 0.125, 0.25, 0.5]),
    }

    fig, axes = plt.subplots(2, 5, figsize=(20, 7))

    for col_idx, (label, (col, vals)) in enumerate(process_vars.items()):
        subset = kinetics[kinetics[col].isin(vals)].copy()
        subset[col] = subset[col].astype(str)

        for row_idx, (param, color, ylabel) in enumerate([
            ("E_max", TEAL,   "Max extraction, $E_{max}$ (%)"),
            ("k",     CORAL,  "Rate constant, $k$ (h⁻¹)"),
        ]):
            ax = axes[row_idx][col_idx]
            means = subset.groupby(col)[param].mean()
            x_pos = np.arange(len(means))
            ax.bar(x_pos, means.values, color=color, alpha=0.8, width=0.55)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(means.index, fontsize=8, rotation=20, ha="right")
            if col_idx == 0:
                ax.set_ylabel(ylabel, fontsize=9)
            if row_idx == 0:
                ax.set_title(label, fontsize=10, fontweight="bold")

    fig.suptitle("Sensitivity of Kinetic Parameters to Each Process Variable",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("outputs/02_parameter_sensitivity.png", bbox_inches="tight")
    plt.close()
    print("[Stage 2] Saved → outputs/02_parameter_sensitivity.png")


def plot_correlation_matrix(kinetics: pd.DataFrame):
    """
    Pearson correlation heatmap between all process variables and kinetic
    parameters. Annotated with r values. Helps identify which lever controls
    which kinetic parameter most strongly.
    """
    cols = ["pH", "inoculum_pct", "pulp_density_pct",
            "glycine_mol_L", "substrate_ratio", "E_max", "k", "t_half"]
    labels = ["pH", "Inoculum", "Pulp density",
              "Glycine", "M:G ratio", "E_max", "k", "t½"]
    corr = kinetics[cols].corr()
    corr.index   = labels
    corr.columns = labels

    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, ax=ax,
                linewidths=0.5, annot_kws={"size": 10})
    ax.set_title("Pearson Correlation — Process Variables vs Kinetic Parameters",
                 fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig("outputs/03_correlation_matrix.png", bbox_inches="tight")
    plt.close()
    print("[Stage 2] Saved → outputs/03_correlation_matrix.png")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — MACHINE LEARNING ON KINETIC PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

FEATURE_COLS = ["pH", "inoculum_pct", "pulp_density_pct",
                "glycine_mol_L", "substrate_ratio"]
FEATURE_LABELS = ["pH", "Inoculum (%)", "Pulp density (%)", "Glycine (M)", "M:G ratio"]

def build_ml_dataset(kinetics: pd.DataFrame):
    """
    Build X (process conditions) and three y targets (E_max, k, t_half).
    With only 20 samples, Leave-One-Out cross-validation is used throughout —
    it is the statistically correct choice for this sample size.
    """
    X = kinetics[FEATURE_COLS].values
    y_Emax  = kinetics["E_max"].values
    y_k     = kinetics["k"].values
    y_thalf = kinetics["t_half"].values
    return X, y_Emax, y_k, y_thalf


def evaluate_models(X, y, target_name: str, scaler) -> dict:
    """
    Train three models (RF, GBR, SVR) on the full dataset with LOO-CV scoring.
    Returns a results dict with CV R², MAE per model, and the best fitted model.
    """
    X_scaled = scaler.transform(X)
    loo = LeaveOneOut()

    models = {
        "Random Forest":   RandomForestRegressor(n_estimators=300, max_depth=3,
                                                  min_samples_leaf=2, random_state=42),
        "Gradient Boost":  GradientBoostingRegressor(n_estimators=200, max_depth=2,
                                                      learning_rate=0.05, random_state=42),
        "SVR (RBF)":       SVR(kernel="rbf", C=10, epsilon=0.5, gamma="scale"),
    }

    results = {}
    for name, model in models.items():
        # cross_val_predict gives OOF predictions across all LOO folds
        # R² computed over the full OOF vector is meaningful; per-fold R² on 1 sample is not
        y_oof  = cross_val_predict(model, X_scaled, y, cv=loo)
        oof_r2 = r2_score(y, y_oof)
        oof_mae = mean_absolute_error(y, y_oof)
        # std: compute per-fold absolute errors for spread estimate
        fold_maes = np.abs(y - y_oof)
        model.fit(X_scaled, y)
        results[name] = {
            "model":     model,
            "cv_r2":     oof_r2,
            "cv_mae":    oof_mae,
            "cv_r2_std": fold_maes.std(),
        }
        print(f"    {name:20s}  R²={oof_r2:.3f}  MAE={oof_mae:.3f}")

    best_name = max(results, key=lambda n: results[n]["cv_r2"])
    print(f"  → Best model for {target_name}: {best_name}")
    return results, best_name


def plot_model_comparison(all_results: dict):
    """
    Grouped bar chart comparing CV R² of all three models across the three
    kinetic parameter targets (E_max, k, t_half). Error bars show ±1 std.
    """
    targets = list(all_results.keys())
    model_names = list(next(iter(all_results.values())).keys())
    n_models = len(model_names)
    x = np.arange(len(targets))
    width = 0.22
    colors = [TEAL, CORAL, PURPLE]

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (mname, color) in enumerate(zip(model_names, colors)):
        r2s  = [all_results[t][mname]["cv_r2"]  for t in targets]
        ax.bar(x + i * width, r2s, width, label=mname,
               color=color, alpha=0.85)

    ax.set_xticks(x + width)
    ax.set_xticklabels(["$E_{max}$ (max extraction)",
                        "$k$ (rate constant)", "$t_{½}$ (inflection time)"],
                       fontsize=11)
    ax.set_ylabel("LOO Cross-Validation R²", fontsize=11)
    ax.set_ylim(-0.5, 1.05)
    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.4)
    ax.legend(fontsize=10)
    ax.set_title("Model Comparison — LOO-CV R² Across All Kinetic Targets",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("outputs/04_model_comparison.png", bbox_inches="tight")
    plt.close()
    print("[Stage 3] Saved → outputs/04_model_comparison.png")


def plot_predicted_vs_actual(X_scaled, kinetics, all_results, best_models):
    """
    Parity plots (predicted vs actual) for the best model on each target.
    Perfect prediction lies on the diagonal. Colour by condition.
    """
    targets = {
        "E_max":  ("$E_{max}$ (%)",    TEAL),
        "k":      ("$k$ (h⁻¹)",        CORAL),
        "t_half": ("$t_{½}$ (h)",       PURPLE),
    }
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for ax, (target, (label, color)) in zip(axes, targets.items()):
        y_true = kinetics[target].values
        model  = all_results[target][best_models[target]]["model"]
        y_pred = model.predict(X_scaled)
        r2     = r2_score(y_true, y_pred)
        mae    = mean_absolute_error(y_true, y_pred)

        lims = [min(y_true.min(), y_pred.min()) * 0.9,
                max(y_true.max(), y_pred.max()) * 1.05]
        ax.plot(lims, lims, "--", color=GRAY, lw=1.5, zorder=1)
        ax.scatter(y_true, y_pred, color=color, s=70, zorder=5, alpha=0.85)

        for xi, yi, lab in zip(y_true, y_pred,
                                kinetics["pH"].astype(str) + "|" +
                                kinetics["pulp_density_pct"].astype(str)):
            ax.annotate(lab, (xi, yi), fontsize=6, alpha=0.55,
                        xytext=(3, 3), textcoords="offset points")

        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel(f"Observed {label}", fontsize=10)
        ax.set_ylabel(f"Predicted {label}", fontsize=10)
        ax.set_title(f"{label}\n{best_models[target]}  |  R²={r2:.3f}  MAE={mae:.2f}",
                     fontsize=10, fontweight="bold")

    fig.suptitle("Parity Plots — Best Model per Kinetic Parameter",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("outputs/05_parity_plots.png", bbox_inches="tight")
    plt.close()
    print("[Stage 3] Saved → outputs/05_parity_plots.png")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — SHAP INTERPRETATION
# ══════════════════════════════════════════════════════════════════════════════

def plot_shap(X, X_scaled, kinetics, all_results, best_models):
    """
    SHAP (SHapley Additive exPlanations) beeswarm plots for the best model
    predicting E_max. Each dot is one experimental condition; position on
    x-axis shows how much that feature pushed the prediction up or down.
    Color shows the raw feature value (red = high, blue = low).
    This is what you show a metallurgist when they ask 'which lever matters most'.
    """
    target = "E_max"
    model  = all_results[target][best_models[target]]["model"]

    explainer   = shap.TreeExplainer(model) if hasattr(model, "feature_importances_") \
                  else shap.KernelExplainer(model.predict, X_scaled)
    shap_values = explainer.shap_values(X_scaled)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # — Beeswarm / summary plot —
    plt.sca(axes[0])
    shap.summary_plot(shap_values, X_scaled,
                      feature_names=FEATURE_LABELS,
                      show=False, plot_type="dot", color_bar=True)
    axes[0].set_title(f"SHAP values — {best_models[target]} on $E_{{max}}$",
                      fontsize=11, fontweight="bold")

    # — Bar chart of mean |SHAP| (global importance) —
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    sorted_idx    = np.argsort(mean_abs_shap)
    axes[1].barh(np.array(FEATURE_LABELS)[sorted_idx],
                 mean_abs_shap[sorted_idx], color=TEAL, alpha=0.85)
    axes[1].set_xlabel("Mean |SHAP value|", fontsize=10)
    axes[1].set_title("Global feature importance (mean |SHAP|)",
                      fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig("outputs/06_shap_Emax.png", bbox_inches="tight")
    plt.close()
    print("[Stage 4] Saved → outputs/06_shap_Emax.png")

    return shap_values, mean_abs_shap


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 5 — RESPONSE SURFACE AND OPTIMAL CONDITIONS
# ══════════════════════════════════════════════════════════════════════════════

def plot_response_surface(all_results, best_models, scaler):
    """
    2D response surfaces for E_max over a grid of the two most important
    process variables (pulp density vs glycine), with all other variables
    fixed at the experimentally optimal values. Contour lines + filled colour
    give the 'operating map' a process engineer can read directly.
    """
    model = all_results["E_max"][best_models["E_max"]]["model"]

    # Fixed at: pH=9, inoculum=6.67%, pulp and glycine swept
    pulp_range = np.linspace(0.33, 1.67, 80)
    gly_range  = np.linspace(0.013, 0.267, 80)
    PD, GL     = np.meshgrid(pulp_range, gly_range)

    X_grid = np.column_stack([
        np.full(PD.size, 9),
        np.full(PD.size, 6.67),
        PD.ravel(),
        GL.ravel(),
        np.full(PD.size, 0.25),
    ])
    X_grid_scaled = scaler.transform(X_grid)
    Z = model.predict(X_grid_scaled).reshape(PD.shape)

    fig, ax = plt.subplots(figsize=(8, 6))
    cf = ax.contourf(PD, GL, Z, levels=20, cmap="YlOrRd")
    cs = ax.contour(PD, GL, Z, levels=10, colors="black", linewidths=0.5, alpha=0.4)
    ax.clabel(cs, fmt="%.0f%%", fontsize=8, inline=True)
    fig.colorbar(cf, ax=ax, label="Predicted $E_{max}$ (%)")

    # Mark the predicted optimum
    opt_idx   = np.unravel_index(np.argmax(Z), Z.shape)
    ax.scatter(PD[opt_idx], GL[opt_idx], marker="*",
               s=250, color="white", edgecolors="black", zorder=10,
               label=f"Optimum: {Z[opt_idx]:.1f}%")
    ax.set_xlabel("Pulp density (%w/v)", fontsize=11)
    ax.set_ylabel("Glycine concentration (mol/L)", fontsize=11)
    ax.set_title("Predicted $E_{max}$ response surface\n"
                 "(pH=9, inoculum=6.67 %v/v, M:G=1:4)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig("outputs/07_response_surface.png", bbox_inches="tight")
    plt.close()
    print("[Stage 5] Saved → outputs/07_response_surface.png")

    # Print optimal condition
    opt_pulp = PD[opt_idx]; opt_gly = GL[opt_idx]
    print(f"\n[Stage 5] Predicted optimum on response surface:")
    print(f"          Pulp density   = {opt_pulp:.3f} %w/v")
    print(f"          Glycine conc.  = {opt_gly:.4f} mol/L")
    print(f"          Predicted E_max = {Z[opt_idx]:.2f}%")


def print_optimisation_table(kinetics: pd.DataFrame):
    """
    Simple ranked table of all experimental conditions by observed E_max.
    Useful as a sanity check alongside the model predictions.
    """
    ranked = kinetics[FEATURE_COLS + ["E_max", "k", "t_half", "r2_fit"]].sort_values(
        "E_max", ascending=False
    ).reset_index(drop=True)
    ranked.index += 1
    print("\n[Stage 5] Experimental conditions ranked by E_max:")
    print(ranked.to_string(float_format="{:.3f}".format))
    ranked.to_csv("outputs/ranked_conditions.csv", index_label="rank")
    print("[Stage 5] Saved → outputs/ranked_conditions.csv")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  GOLD BIOLEACHING — KINETIC + ML PIPELINE")
    print("=" * 65)

    # 0. Load
    df       = load_and_reshape("RawData.xlsx")

    # 1. Kinetic fits
    kinetics = fit_kinetics(df)
    plot_kinetic_fits(df, kinetics)

    # 2. EDA on kinetic parameters
    plot_parameter_sensitivity(kinetics)
    plot_correlation_matrix(kinetics)

    # 3. ML — build dataset and scale once
    X, y_Emax, y_k, y_thalf = build_ml_dataset(kinetics)
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    all_results  = {}
    best_models  = {}

    for target_name, y in [("E_max", y_Emax), ("k", y_k), ("t_half", y_thalf)]:
        print(f"\n  Evaluating models for target: {target_name}")
        results, best = evaluate_models(X, y, target_name, scaler)
        all_results[target_name] = results
        best_models[target_name] = best

    plot_model_comparison(all_results)
    plot_predicted_vs_actual(X_scaled, kinetics, all_results, best_models)

    # 4. SHAP
    plot_shap(X, X_scaled, kinetics, all_results, best_models)

    # 5. Response surface + optimisation table
    plot_response_surface(all_results, best_models, scaler)
    print_optimisation_table(kinetics)

    print("\n" + "=" * 65)
    print("  Pipeline complete. All outputs in ./outputs/")
    print("=" * 65)


if __name__ == "__main__":
    main()
