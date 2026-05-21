from Src.utils.libraries import *
from Src.utils.logger import *
from joblib import Parallel, delayed
from scipy.sparse import issparse
from statsmodels.api import OLS, add_constant
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
from scipy.stats import pearsonr, t
from pathlib import Path
import h5py
import math
import pickle


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


def _decode_h5_values(values):
    return [
        value.decode() if isinstance(value, bytes) else str(value)
        for value in values
    ]


def _load_gene_names_from_path(path):
    path = Path(path)
    if path.suffix == ".pkl":
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        if isinstance(payload, dict) and "gene_names" in payload:
            return [str(gene) for gene in payload["gene_names"]]
        return [str(gene) for gene in payload]
    if path.suffix == ".npy":
        return [str(gene) for gene in np.load(path, allow_pickle=True)]

    genes = pd.read_csv(path, header=None).iloc[:, 0]
    return [str(gene) for gene in genes]


def _resolve_gene_names(h5, gene_names=None):
    n_vars = h5["X"].shape[1]
    if gene_names is None:
        resolved = _decode_h5_values(h5["var/_index"][:])
    elif isinstance(gene_names, (str, Path)):
        resolved = _load_gene_names_from_path(gene_names)
    else:
        resolved = [str(gene) for gene in gene_names]

    if len(resolved) != n_vars:
        raise ValueError(f"gene_names has length {len(resolved)}, expected {n_vars}.")
    return np.asarray(resolved, dtype=object)


def _read_h5ad_obs_column(h5, column):
    if "obs" not in h5 or column not in h5["obs"]:
        raise KeyError(f"obs column {column!r} not found in {h5.filename}.")

    obj = h5["obs"][column]
    encoding_type = obj.attrs.get("encoding-type") if isinstance(obj, h5py.Group) else None
    if isinstance(encoding_type, bytes):
        encoding_type = encoding_type.decode()
    if isinstance(obj, h5py.Group) and encoding_type == "categorical":
        return np.asarray(obj["codes"][:], dtype=np.int64), _decode_h5_values(obj["categories"][:])

    values = np.asarray(_decode_h5_values(obj[:]), dtype=object)
    categories = list(pd.unique(values))
    category_to_code = {category: idx for idx, category in enumerate(categories)}
    codes = np.asarray([category_to_code[value] for value in values], dtype=np.int64)
    return codes, categories


def _maybe_read_h5ad_obs_column(h5, column):
    if column is None or "obs" not in h5 or column not in h5["obs"]:
        return None, None
    return _read_h5ad_obs_column(h5, column)


def _target_perturbations_from_codes(perturbation_codes, perturbation_categories, reference_level):
    present_codes = sorted(set(int(code) for code in perturbation_codes if code >= 0))
    target_perturbations = [perturbation_categories[code] for code in present_codes]
    target_perturbations.sort()
    if reference_level not in target_perturbations:
        raise ValueError(f"referenceLevel {reference_level!r} is not present in the selected observations.")
    target_perturbations = (
        [reference_level]
        + target_perturbations[:target_perturbations.index(reference_level)]
        + target_perturbations[target_perturbations.index(reference_level) + 1:]
    )
    category_to_code = {category: idx for idx, category in enumerate(perturbation_categories)}
    raw_to_target = np.full(len(perturbation_categories), -1, dtype=np.int64)
    for idx, perturbation in enumerate(target_perturbations):
        raw_to_target[category_to_code[perturbation]] = idx
    return target_perturbations, raw_to_target


def _build_score_plans(var_names, gene_means, programs_dict, ctrl_size=50, n_bins=25, random_state=42):
    gene_to_idx = _first_index_by_gene(var_names)
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

    plans = []
    for prog_name, gene_list in programs_dict.items():
        rng = np.random.default_rng(random_state)
        if gene_list is None or len(gene_list) == 0:
            raise ValueError("You must provide a non-empty gene_list.")

        available_genes = [gene for gene in gene_list if gene in gene_to_idx]
        if len(available_genes) == 0:
            raise ValueError("None of the requested genes are in var_names.")

        target_indices = np.asarray([gene_to_idx[gene] for gene in available_genes], dtype=np.intp)
        control_genes = []
        for gene_idx in target_indices:
            gene_bin = bins[gene_idx]
            same_bin = bin_members.get(gene_bin)
            if same_bin is None:
                continue
            same_bin = same_bin[same_bin != gene_idx]
            if len(same_bin) > 0:
                chosen = rng.choice(same_bin, size=min(ctrl_size, len(same_bin)), replace=False)
                control_genes.extend(chosen)

        plans.append({
            "name": prog_name,
            "target_indices": target_indices,
            "control_indices": np.unique(control_genes),
        })
    return plans


def _score_block_from_plans(block, plans):
    scores = np.empty((block.shape[0], len(plans)), dtype=np.float64)
    for plan_idx, plan in enumerate(plans):
        target_expr = block[:, plan["target_indices"]].mean(axis=1)
        if len(plan["control_indices"]) > 0:
            control_expr = block[:, plan["control_indices"]].mean(axis=1)
        else:
            control_expr = np.zeros(block.shape[0], dtype=target_expr.dtype)
        scores[:, plan_idx] = target_expr - control_expr
    return scores


def _build_score_matrix(plans, n_vars, dtype=np.float32):
    score_matrix = np.zeros((n_vars, len(plans)), dtype=dtype)
    for plan_idx, plan in enumerate(plans):
        target_weight = dtype(1.0 / len(plan["target_indices"]))
        np.add.at(score_matrix[:, plan_idx], plan["target_indices"], target_weight)
        if len(plan["control_indices"]) > 0:
            control_weight = dtype(1.0 / len(plan["control_indices"]))
            score_matrix[plan["control_indices"], plan_idx] -= control_weight
    return score_matrix


def _score_block_from_matrix(block, score_matrix):
    return (block @ score_matrix).astype(np.float64, copy=False)


def _empty_regression_accumulators(n_groups, n_programs):
    return {
        "counts": np.zeros(n_groups, dtype=np.float64),
        "sums": np.zeros((n_groups, n_programs), dtype=np.float64),
        "squared_sums": np.zeros((n_groups, n_programs), dtype=np.float64),
        "reference_sums_of_squares": np.zeros(n_programs, dtype=np.float64),
        "n_obs": 0,
    }


def _update_regression_accumulators(accumulators, scores, target_codes):
    valid_mask = target_codes >= 0
    if not valid_mask.all():
        scores = scores[valid_mask]
        target_codes = target_codes[valid_mask]

    accumulators["n_obs"] += len(target_codes)
    reference_mask = target_codes == 0
    if reference_mask.any():
        accumulators["reference_sums_of_squares"] += (scores[reference_mask] ** 2).sum(axis=0)

    non_reference_mask = target_codes > 0
    if not non_reference_mask.any():
        return

    non_reference_codes = target_codes[non_reference_mask] - 1
    non_reference_scores = scores[non_reference_mask]
    n_groups = len(accumulators["counts"])
    accumulators["counts"] += np.bincount(non_reference_codes, minlength=n_groups).astype(np.float64)
    for col_idx in range(non_reference_scores.shape[1]):
        accumulators["sums"][:, col_idx] += np.bincount(
            non_reference_codes,
            weights=non_reference_scores[:, col_idx],
            minlength=n_groups,
        )
        accumulators["squared_sums"][:, col_idx] += np.bincount(
            non_reference_codes,
            weights=non_reference_scores[:, col_idx] ** 2,
            minlength=n_groups,
        )


def _finalize_regression_accumulators(accumulators, program_names, target_perturbations):
    counts = accumulators["counts"]
    sums = accumulators["sums"]
    squared_sums = accumulators["squared_sums"]

    with np.errstate(divide="ignore", invalid="ignore"):
        params = sums / counts[:, None]
        non_reference_ssr = squared_sums - (sums ** 2 / counts[:, None])

    ssr = accumulators["reference_sums_of_squares"] + non_reference_ssr.sum(axis=0)
    model_rank = int(np.count_nonzero(counts))
    df_resid = accumulators["n_obs"] - model_rank
    if df_resid <= 0:
        pvalues = np.full_like(params, np.nan)
    else:
        mse_resid = ssr / df_resid
        with np.errstate(divide="ignore", invalid="ignore"):
            standard_errors = np.sqrt(mse_resid[None, :] / counts[:, None])
            tvalues = params / standard_errors
        pvalues = 2 * t.sf(np.abs(tvalues), df_resid)

    results = pd.DataFrame(index=pd.Index(target_perturbations[1:]))
    for col_idx, col in enumerate(program_names):
        results[f"{col}_coef"] = params[:, col_idx]
        results[f"{col}_pval"] = pvalues[:, col_idx]
    return results


def _coef_correlations_dataframe(pred_res, real_res):
    commonIndex = pred_res.index[[x in real_res.index for x in pred_res.index]]
    pred_res = pred_res.loc[commonIndex,]
    real_res = real_res.loc[commonIndex,]

    coef_cols = [col for col in real_res.columns if col.endswith("_coef")]
    results = []
    for col in coef_cols:
        r, p = pearsonr(real_res[col].values, pred_res[col].values)
        results.append({"pathway": col.replace("_coef", ""), "pearson_r": r, "pval": p})
    return pd.DataFrame(results)


def _stream_program_regressions_from_h5ad(
    adata_path,
    programs_dict,
    perturbationsColumn="gene",
    referenceLevel="non-targeting",
    contextColumn="context",
    gene_names=None,
    chunk_size=25000,
    ctrl_size=50,
    n_bins=25,
    random_state=42,
    score_backend="matrix",
):
    if score_backend not in {"matrix", "indexed"}:
        raise ValueError("score_backend must be 'matrix' or 'indexed'.")

    adata_path = Path(adata_path)
    with h5py.File(adata_path, "r") as h5:
        X = h5["X"]
        if not isinstance(X, h5py.Dataset):
            raise NotImplementedError("Streaming currently supports dense h5ad X datasets only.")

        n_obs, n_vars = X.shape
        var_names = _resolve_gene_names(h5, gene_names)
        perturbation_codes, perturbation_categories = _read_h5ad_obs_column(h5, perturbationsColumn)
        context_codes, context_categories = _maybe_read_h5ad_obs_column(h5, contextColumn)
        if context_codes is None:
            context_codes = np.zeros(n_obs, dtype=np.int64)
            context_categories = ["all"]

        present_context_codes = [
            code for code in range(len(context_categories))
            if np.any(context_codes == code)
        ]
        context_labels = {
            code: context_categories[code]
            for code in present_context_codes
        }

        gene_sums = {
            code: np.zeros(n_vars, dtype=np.float64)
            for code in present_context_codes
        }
        context_counts = {
            code: 0
            for code in present_context_codes
        }

        for start in range(0, n_obs, chunk_size):
            end = min(start + chunk_size, n_obs)
            block = X[start:end, :]
            block_context_codes = context_codes[start:end]
            for context_code in present_context_codes:
                if len(present_context_codes) == 1:
                    context_block = block
                else:
                    mask = block_context_codes == context_code
                    if not mask.any():
                        continue
                    context_block = block[mask]
                gene_sums[context_code] += context_block.sum(axis=0, dtype=np.float64)
                context_counts[context_code] += context_block.shape[0]

        plans_by_context = {}
        score_matrices_by_context = {}
        target_perturbations_by_context = {}
        raw_to_target_by_context = {}
        accumulators_by_context = {}
        program_names = list(programs_dict.keys())
        for context_code in present_context_codes:
            means = (gene_sums[context_code] / context_counts[context_code]).astype(X.dtype, copy=False)
            plans_by_context[context_code] = _build_score_plans(
                var_names,
                means,
                programs_dict,
                ctrl_size=ctrl_size,
                n_bins=n_bins,
                random_state=random_state,
            )
            if score_backend == "matrix":
                score_matrices_by_context[context_code] = _build_score_matrix(
                    plans_by_context[context_code],
                    n_vars,
                )
            context_mask = context_codes == context_code
            target_perturbations, raw_to_target = _target_perturbations_from_codes(
                perturbation_codes[context_mask],
                perturbation_categories,
                referenceLevel,
            )
            target_perturbations_by_context[context_code] = target_perturbations
            raw_to_target_by_context[context_code] = raw_to_target
            accumulators_by_context[context_code] = _empty_regression_accumulators(
                len(target_perturbations) - 1,
                len(program_names),
            )

        for start in range(0, n_obs, chunk_size):
            end = min(start + chunk_size, n_obs)
            block = X[start:end, :]
            block_context_codes = context_codes[start:end]
            block_perturbation_codes = perturbation_codes[start:end]
            for context_code in present_context_codes:
                if len(present_context_codes) == 1:
                    context_block = block
                    context_perturbation_codes = block_perturbation_codes
                else:
                    mask = block_context_codes == context_code
                    if not mask.any():
                        continue
                    context_block = block[mask]
                    context_perturbation_codes = block_perturbation_codes[mask]

                if score_backend == "matrix":
                    scores = _score_block_from_matrix(
                        context_block,
                        score_matrices_by_context[context_code],
                    )
                else:
                    scores = _score_block_from_plans(context_block, plans_by_context[context_code])
                target_codes = raw_to_target_by_context[context_code][context_perturbation_codes]
                _update_regression_accumulators(
                    accumulators_by_context[context_code],
                    scores,
                    target_codes,
                )

        regression_results = {}
        for context_code in present_context_codes:
            regression_results[context_labels[context_code]] = _finalize_regression_accumulators(
                accumulators_by_context[context_code],
                program_names,
                target_perturbations_by_context[context_code],
            )
    return regression_results


def testGeneProgramsConcordanceStreaming(
    pred_adata_path,
    real_adata_path,
    programs_dict,
    perturbationsColumn="gene",
    referenceLevel="non-targeting",
    contextColumn="context",
    gene_names=None,
    pred_gene_names=None,
    real_gene_names=None,
    chunk_size=25000,
    ctrl_size=50,
    n_bins=25,
    random_state=42,
    score_backend="matrix",
    return_regressions=False,
):
    """Stream h5ad-backed dense X matrices and score each context independently.

    This path uses h5py directly for obs metadata and X chunk reads. If
    contextColumn is present, each context is scored with context-specific
    expression bins and regressions, matching the result of splitting the
    AnnData by context before calling the in-memory implementation.
    score_backend="matrix" computes all program scores for a chunk with one
    dense matrix multiply per context. score_backend="indexed" keeps the
    original per-program column-slicing score kernel for stricter parity checks.
    """

    pred_gene_names = pred_gene_names if pred_gene_names is not None else gene_names
    real_gene_names = real_gene_names if real_gene_names is not None else gene_names

    pred_regressions = _stream_program_regressions_from_h5ad(
        pred_adata_path,
        programs_dict,
        perturbationsColumn=perturbationsColumn,
        referenceLevel=referenceLevel,
        contextColumn=contextColumn,
        gene_names=pred_gene_names,
        chunk_size=chunk_size,
        ctrl_size=ctrl_size,
        n_bins=n_bins,
        random_state=random_state,
        score_backend=score_backend,
    )
    real_regressions = _stream_program_regressions_from_h5ad(
        real_adata_path,
        programs_dict,
        perturbationsColumn=perturbationsColumn,
        referenceLevel=referenceLevel,
        contextColumn=contextColumn,
        gene_names=real_gene_names,
        chunk_size=chunk_size,
        ctrl_size=ctrl_size,
        n_bins=n_bins,
        random_state=random_state,
        score_backend=score_backend,
    )

    pred_contexts = set(pred_regressions)
    real_contexts = set(real_regressions)
    if pred_contexts != real_contexts:
        raise ValueError(
            "pred and real h5ad files contain different contexts: "
            f"pred-only={sorted(pred_contexts - real_contexts)}, "
            f"real-only={sorted(real_contexts - pred_contexts)}."
        )

    results = []
    for context in pred_regressions:
        context_results = _coef_correlations_dataframe(
            pred_regressions[context],
            real_regressions[context],
        )
        context_results.insert(0, "context", context)
        results.append(context_results)

    results_df = pd.concat(results, ignore_index=True)
    if return_regressions:
        return results_df, pred_regressions, real_regressions
    return results_df


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


def _fit_program_regression_vectorized(expression_matrix, perturbation_codes, n_groups):
    y = np.asarray(expression_matrix, dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]

    non_reference_mask = perturbation_codes > 0
    non_reference_codes = perturbation_codes[non_reference_mask] - 1
    y_non_reference = y[non_reference_mask]

    counts = np.bincount(non_reference_codes, minlength=n_groups).astype(np.float64)
    sums = np.vstack([
        np.bincount(non_reference_codes, weights=y_non_reference[:, col], minlength=n_groups)
        for col in range(y.shape[1])
    ]).T
    squared_sums = np.vstack([
        np.bincount(non_reference_codes, weights=y_non_reference[:, col] ** 2, minlength=n_groups)
        for col in range(y.shape[1])
    ]).T

    with np.errstate(divide="ignore", invalid="ignore"):
        params = sums / counts[:, None]
        non_reference_ssr = squared_sums - (sums ** 2 / counts[:, None])

    reference_sums_of_squares = (y[perturbation_codes == 0] ** 2).sum(axis=0)
    ssr = reference_sums_of_squares + non_reference_ssr.sum(axis=0)
    model_rank = int(np.count_nonzero(counts))
    df_resid = y.shape[0] - model_rank

    if df_resid <= 0:
        pvalues = np.full_like(params, np.nan)
        return params, pvalues

    mse_resid = ssr / df_resid
    with np.errstate(divide="ignore", invalid="ignore"):
        standard_errors = np.sqrt(mse_resid[None, :] / counts[:, None])
        tvalues = params / standard_errors
    pvalues = 2 * t.sf(np.abs(tvalues), df_resid)

    return params, pvalues


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

    # Expression matrix: use provided pathways or default to numeric obs
    if pathways is None:
        expressionMatrix = adata.obs.select_dtypes(include="number")
    else:
        expressionMatrix = adata.obs[pathways]

    perturbation_codes = adata.obs[perturbationsColumn].cat.codes.to_numpy()
    n_groups = len(targetPerturbations) - 1
    params, pvalues = _fit_program_regression_vectorized(
        expressionMatrix,
        perturbation_codes,
        n_groups
    )

    index = pd.Index(targetPerturbations[1:])
    final_results = pd.DataFrame(index=index)
    for col_idx, col in enumerate(expressionMatrix.columns):
        final_results[f"{col}_coef"] = params[:, col_idx]
        final_results[f"{col}_pval"] = pvalues[:, col_idx]

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


testGeneProgramsConcordance.__test__ = False
testGeneProgramsConcordanceStreaming.__test__ = False
