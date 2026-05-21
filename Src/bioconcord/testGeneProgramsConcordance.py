from Src.utils.libraries import *
from Src.utils.logger import *
from joblib import Parallel, delayed
from scipy.sparse import issparse
from statsmodels.api import OLS, add_constant
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
from scipy.stats import pearsonr


def _first_index_by_gene(var_names):
    gene_to_idx = {}
    for idx, gene in enumerate(var_names):
        if gene not in gene_to_idx:
            gene_to_idx[gene] = idx
    return gene_to_idx


def _prepare_score_context(adata, n_bins=25):
    X = adata.X
    var_names = np.array(adata.var_names)

    if issparse(X):
        gene_means = np.array(X.mean(axis=0)).ravel()
    else:
        gene_means = X.mean(axis=0)

    bins = np.asarray(pd.qcut(gene_means, n_bins, labels=False, duplicates="drop"))
    bin_members = {}
    for idx, bin_id in enumerate(bins):
        if pd.isna(bin_id):
            continue
        bin_members.setdefault(bin_id, []).append(idx)
    bin_members = {
        bin_id: np.asarray(indices, dtype=np.intp)
        for bin_id, indices in bin_members.items()
    }

    return {
        "X": X,
        "obs_names": adata.obs_names,
        "var_names": var_names,
        "gene_to_idx": _first_index_by_gene(var_names),
        "bins": bins,
        "bin_members": bin_members,
    }


def _score_genes_from_context(
    context,
    prog_name,
    gene_list=None,
    ctrl_size=50,
    random_state=0
):
    X = context["X"]
    gene_to_idx = context["gene_to_idx"]
    bins = context["bins"]
    bin_members = context["bin_members"]

    rng = np.random.default_rng(random_state)

    if gene_list is None or len(gene_list) == 0:
        raise ValueError("You must provide a non-empty gene_list.")

    gene_list = [g for g in gene_list if g in gene_to_idx]
    if len(gene_list) == 0:
        raise ValueError("None of the requested genes are in var_names.")

    target_indices = [gene_to_idx[g] for g in gene_list]

    control_genes = []
    for g_idx in target_indices:
        g_bin = bins[g_idx]
        same_bin = bin_members.get(g_bin)
        if same_bin is None:
            continue
        same_bin = same_bin[same_bin != g_idx]
        if len(same_bin) > 0:
            chosen = rng.choice(same_bin, size=min(ctrl_size, len(same_bin)), replace=False)
            control_genes.extend(chosen)

    control_genes = np.unique(control_genes)

    if issparse(X):
        target_expr = np.array(X[:, target_indices].mean(axis=1)).ravel()
        control_expr = np.array(X[:, control_genes].mean(axis=1)).ravel() if len(control_genes) > 0 else np.zeros(X.shape[0])
    else:
        target_expr = X[:, target_indices].mean(axis=1)
        control_expr = X[:, control_genes].mean(axis=1) if len(control_genes) > 0 else np.zeros(X.shape[0])

    scores = target_expr - control_expr

    return pd.Series(scores, index=context["obs_names"], name=prog_name)


def score_genes_standalone(
    adata,
    prog_name,
    gene_list=None,
    ctrl_size=50,
    n_bins=25,
    random_state=0
):
    """
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with cells in `.obs` and genes in `.var`. 
        The function uses `adata.X` (or `adata.raw.X` if present) for expression values.

    prog_name : str
        Name of the program or pathway being scored. Used to label the output.

    gene_list : list of str, optional (default: None)
        List of genes to compute the program score. If None or empty, 
        the function will return NaN scores for all cells.

    ctrl_size : int, optional (default: 50)
        Number of control genes randomly sampled from the gene pool (genes not in `gene_list`)
        for score normalization. If fewer than `ctrl_size` genes are available, all are used.

    n_bins : int, optional (default: 25)
        Number of expression bins used to match control genes by expression levels. 
        (Currently optional placeholder; can be integrated for Seurat-style binning.)

    random_state : int, optional (default: 0)
        Random seed for reproducible control gene sampling.


    Returns
    -------
    pd.Series
        Gene scores for each cell (index = obs_names or range(n_cells)).
    """

    context = _prepare_score_context(adata, n_bins=n_bins)
    return _score_genes_from_context(
        context,
        prog_name,
        gene_list=gene_list,
        ctrl_size=ctrl_size,
        random_state=random_state
    )


def score_all_programs(adata, programs_dict, n_jobs=15, ctrl_size=50, n_bins=25, random_state=42):
    """
    Compute scores for each gene set (program) in parallel and update adata.obs.

       Parameters
    ----------
    adata : AnnData
        Single-cell AnnData object containing expression values (adata.X) and metadata (adata.obs, adata.var).
        This object will be used as input for computing program scores.

    programs_dict : dict
        Dictionary where keys are program names and values are lists of gene symbols
        corresponding to each program. Each gene list is used for computing a program score in the AnnData.

    n_jobs : int, optional (default=15)
        Number of parallel processes to use when computing program scores.

    ctrl_size : int, optional (default=50)
        Number of control genes randomly sampled per target gene.

    n_bins : int, optional (default=25)
        Number of expression bins used to match control genes.

    random_state : int, optional (default=42)
        Random seed used for each program's control gene sampling.
        
    """

    context = _prepare_score_context(adata, n_bins=n_bins)
    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_score_genes_from_context)(
            context,
            prog_name,
            gene_list=gene_list,
            ctrl_size=ctrl_size,
            random_state=random_state
        )
        for prog_name, gene_list in programs_dict.items()
    )

    # Combine results into DataFrame
    scores_df = pd.concat(results, axis=1)
    scores_df.columns = list(programs_dict.keys())

    # Update adata.obs
    for col in scores_df.columns:
        adata.obs[col] = scores_df[col]

    return adata

def fit_one_column(col, expressionMatrix, designMatrix):
    """Fit OLS for one column and return results as DataFrame."""
    model = sm.OLS(expressionMatrix[col], designMatrix).fit()
    results_df = pd.DataFrame({
        "coef": model.params,
        "pval": model.pvalues
    })
    return results_df.add_prefix(f"{col}_")


def run_program_regression(
    adata: "AnnData",
    perturbationsColumn: str = "gene",
    referenceLevel: str = "non-targeting",
    pathways=None
) -> pd.DataFrame:
    """
    Run OLS regression for each pathway (or gene set) in an AnnData object,
    comparing perturbations against a reference.

    Parameters
    ----------
    adata : AnnData
        Input AnnData object with perturbation metadata in .obs
    perturbationsColumn : str
        The .obs column that contains perturbation labels
    referenceLevel : str
        Reference perturbation level to use as baseline
    pathways : list-like
        List of column names in .obs to use as expression matrix
        (e.g. pathways, gene signatures). If None, all numeric obs columns are used.

    Returns
    -------
    pd.DataFrame
        Concatenated regression results (coefficients & p-values per predictor)
    """

    # Get perturbations
    targetPerturbations = list(adata.obs[perturbationsColumn].unique())
    targetPerturbations.sort()
    targetPerturbations = (
        [referenceLevel] +
        targetPerturbations[:targetPerturbations.index(referenceLevel)] +
        targetPerturbations[targetPerturbations.index(referenceLevel)+1:]
    )

    # Make categorical with ordering
    adata.obs[perturbationsColumn] = pd.Categorical(
        adata.obs[perturbationsColumn],
        categories=targetPerturbations,
        ordered=True
    )

    # Design matrix (dummy coding, first level dropped = reference)
    designMatrix = pd.get_dummies(
        adata.obs[perturbationsColumn],
        drop_first=True
    )

    # Expression matrix: use provided pathways or default to numeric obs
    if pathways is None:
        expressionMatrix = adata.obs.select_dtypes(include="number")
    else:
        expressionMatrix = adata.obs[pathways]

  
    all_results = Parallel(n_jobs=-1)(  # -1 = use all available cores
    delayed(fit_one_column)(col, expressionMatrix, designMatrix)
    for col in expressionMatrix.columns)

    # Concatenate side by side
    final_results = pd.concat(all_results, axis=1)


    return final_results


def plot_coef_correlations(pred_res, real_res, ncols=5, figsize_per_plot=5, save_path=None):
    """
    Plot scatterplots of real vs predicted coefficients for each pathway in one figure.
    Also computes Pearson correlations and returns them as a DataFrame.

    Parameters
    ----------
    pred_res : pd.DataFrame
        DataFrame with predicted coefficients.
    real_res : pd.DataFrame
        DataFrame with real coefficients.
    ncols : int
        Number of subplot columns.
    figsize_per_plot : int
        Size multiplier for each subplot.
    save_path : str or None
        If provided, saves the figure as PDF/PNG.
        If None, saves to './bioconcord_figures/coef_correlations.pdf'.

    Returns
    -------
    results_df : pd.DataFrame
        DataFrame with pathway name, Pearson r, and p-value.
    """

    commonIndex = pred_res.index[[x in real_res.index for x in pred_res.index ]]
    pred_res = pred_res.loc[commonIndex,]
    real_res = real_res.loc[commonIndex,]

    print("alooooo")

    coef_cols = [c for c in real_res.columns if c.endswith("_coef")]
    n_features = len(coef_cols)
    nrows = math.ceil(n_features / ncols)

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(figsize_per_plot*ncols, figsize_per_plot*nrows)
    )
    axes = axes.flatten()

    results = []

    for i, col in enumerate(coef_cols):
        ax = axes[i]
        real_vals = real_res[col].values
        pred_vals = pred_res[col].values

        # Pearson correlation
        r, p = pearsonr(real_vals, pred_vals)
        results.append({"pathway": col.replace("_coef", ""), "pearson_r": r, "pval": p})

        # Scatter plot
        sns.scatterplot(x=real_vals, y=pred_vals, alpha=0.6, s=20, color="steelblue", ax=ax)
        ax.axline((0, 0), slope=1, linestyle="--", color="red")
        ax.set_title(f"{col.replace('_coef','')}\n r={r:.2f}, p={p:.1e}")
        ax.set_xlabel("Real coefficients")
        ax.set_ylabel("Predicted coefficients")

    # Hide unused subplots
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()

    # Default save path
    if save_path is None:
        os.makedirs("./bioconcord_figures", exist_ok=True)
        save_path = "./bioconcord_figures/coef_correlations.pdf"

    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

    return pd.DataFrame(results)

def testGeneProgramsConcordance(pred_adata, real_adata,
                                programs_dict, perturbationsColumn="gene",
                                referenceLevel="non-targeting",ncols=5, figsize_per_plot=5, save_path=None):

    pred_adata = score_all_programs(pred_adata, programs_dict)
    real_adata = score_all_programs(real_adata, programs_dict)

    pred_res = run_program_regression(pred_adata, perturbationsColumn, referenceLevel, programs_dict.keys())
    real_res = run_program_regression(real_adata, perturbationsColumn, referenceLevel, programs_dict.keys())


    results_df = plot_coef_correlations(pred_res, real_res,ncols, figsize_per_plot, save_path)

    return results_df
