import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import statsmodels.api as sm
from scipy.sparse import issparse

from Src.bioconcord.testGeneProgramsConcordance import (
    _coef_correlations_dataframe,
    run_program_regression,
    score_all_programs,
    score_genes_standalone,
    testGeneProgramsConcordanceStreaming as streaming_gene_program_concordance,
)


def _reference_score_genes_standalone(
    adata,
    prog_name,
    gene_list=None,
    ctrl_size=50,
    n_bins=25,
    random_state=0,
):
    X = adata.X
    var_names = np.array(adata.var_names)

    rng = np.random.default_rng(random_state)

    if gene_list is None or len(gene_list) == 0:
        raise ValueError("You must provide a non-empty gene_list.")

    gene_list = [g for g in gene_list if g in var_names]
    if len(gene_list) == 0:
        raise ValueError("None of the requested genes are in var_names.")

    if issparse(X):
        gene_means = np.array(X.mean(axis=0)).ravel()
    else:
        gene_means = X.mean(axis=0)

    bins = pd.qcut(gene_means, n_bins, labels=False, duplicates="drop")

    control_genes = []
    for g in gene_list:
        g_idx = np.where(var_names == g)[0][0]
        g_bin = bins[g_idx]
        same_bin = np.where(bins == g_bin)[0]
        same_bin = same_bin[same_bin != g_idx]
        if len(same_bin) > 0:
            chosen = rng.choice(same_bin, size=min(ctrl_size, len(same_bin)), replace=False)
            control_genes.extend(chosen)

    control_genes = np.unique(control_genes)

    if issparse(X):
        target_expr = np.array(
            X[:, [np.where(var_names == g)[0][0] for g in gene_list]].mean(axis=1)
        ).ravel()
        control_expr = (
            np.array(X[:, control_genes].mean(axis=1)).ravel()
            if len(control_genes) > 0
            else np.zeros(X.shape[0])
        )
    else:
        target_expr = X[:, [np.where(var_names == g)[0][0] for g in gene_list]].mean(axis=1)
        control_expr = (
            X[:, control_genes].mean(axis=1)
            if len(control_genes) > 0
            else np.zeros(X.shape[0])
        )

    scores = target_expr - control_expr

    return pd.Series(scores, index=adata.obs_names, name=prog_name)


def _reference_run_program_regression(
    adata,
    perturbationsColumn="gene",
    referenceLevel="non-targeting",
    pathways=None,
):
    targetPerturbations = list(adata.obs[perturbationsColumn].unique())
    targetPerturbations.sort()
    targetPerturbations = (
        [referenceLevel]
        + targetPerturbations[:targetPerturbations.index(referenceLevel)]
        + targetPerturbations[targetPerturbations.index(referenceLevel) + 1:]
    )

    adata.obs[perturbationsColumn] = pd.Categorical(
        adata.obs[perturbationsColumn],
        categories=targetPerturbations,
        ordered=True,
    )
    designMatrix = pd.get_dummies(
        adata.obs[perturbationsColumn],
        drop_first=True,
    )

    if pathways is None:
        expressionMatrix = adata.obs.select_dtypes(include="number")
    else:
        expressionMatrix = adata.obs[pathways]

    all_results = []
    for col in expressionMatrix.columns:
        model = sm.OLS(expressionMatrix[col], designMatrix).fit()
        results_df = pd.DataFrame({
            "coef": model.params,
            "pval": model.pvalues,
        })
        all_results.append(results_df.add_prefix(f"{col}_"))

    return pd.concat(all_results, axis=1)


def _adata_with_matrix(X):
    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(X.shape[0])])
    var = pd.DataFrame(index=["g0", "g1", "g2", "g1", "g4", "g5", "g6", "g7"])
    return ad.AnnData(X=X, obs=obs, var=var)


def test_score_genes_standalone_matches_reference_dense():
    X = np.array(
        [
            [0.1, 1.0, 0.0, 0.3, 2.2, 1.1, 0.0, 4.0],
            [0.2, 1.3, 0.2, 0.6, 1.9, 1.0, 0.1, 3.8],
            [0.4, 1.2, 0.4, 0.9, 1.7, 0.9, 0.3, 3.6],
            [0.8, 0.9, 0.8, 1.2, 1.5, 0.8, 0.5, 3.4],
            [1.6, 0.8, 1.6, 1.5, 1.3, 0.6, 0.7, 3.2],
            [3.2, 0.7, 3.2, 1.8, 1.1, 0.4, 0.9, 3.0],
        ],
        dtype=np.float32,
    )
    adata = _adata_with_matrix(X)
    genes = ["g1", "missing", "g4", "g7"]

    expected = _reference_score_genes_standalone(
        adata, "program", genes, ctrl_size=2, n_bins=4, random_state=7
    )
    actual = score_genes_standalone(
        adata, "program", genes, ctrl_size=2, n_bins=4, random_state=7
    )

    pd.testing.assert_series_equal(actual, expected, check_exact=True)


def test_score_genes_standalone_matches_reference_sparse():
    X = sp.csr_matrix(
        np.array(
            [
                [0.0, 1.0, 0.0, 0.3, 2.2, 1.1, 0.0, 4.0],
                [0.2, 0.0, 0.2, 0.6, 1.9, 0.0, 0.1, 3.8],
                [0.4, 1.2, 0.0, 0.0, 1.7, 0.9, 0.3, 0.0],
                [0.0, 0.9, 0.8, 1.2, 0.0, 0.8, 0.5, 3.4],
                [1.6, 0.0, 1.6, 1.5, 1.3, 0.0, 0.7, 0.0],
                [3.2, 0.7, 0.0, 1.8, 1.1, 0.4, 0.0, 3.0],
            ],
            dtype=np.float32,
        )
    )
    adata = _adata_with_matrix(X)
    genes = ["g0", "g2", "g6"]

    expected = _reference_score_genes_standalone(
        adata, "program", genes, ctrl_size=2, n_bins=4, random_state=11
    )
    actual = score_genes_standalone(
        adata, "program", genes, ctrl_size=2, n_bins=4, random_state=11
    )

    pd.testing.assert_series_equal(actual, expected, check_exact=True)


def test_score_all_programs_matches_reference_with_threading():
    X = np.arange(80, dtype=np.float32).reshape(10, 8)
    adata = _adata_with_matrix(X.copy())
    expected_adata = _adata_with_matrix(X.copy())
    programs = {
        "p0": ["g0", "g1", "missing"],
        "p1": ["g2", "g4", "g5"],
        "p2": ["g6", "g7"],
    }

    score_all_programs(adata, programs, n_jobs=2, ctrl_size=2, n_bins=4, random_state=42)

    for prog_name, genes in programs.items():
        expected_adata.obs[prog_name] = _reference_score_genes_standalone(
            expected_adata,
            prog_name,
            genes,
            ctrl_size=2,
            n_bins=4,
            random_state=42,
        )
        pd.testing.assert_series_equal(
            adata.obs[prog_name],
            expected_adata.obs[prog_name],
            check_exact=True,
        )


def test_run_program_regression_matches_statsmodels_reference():
    obs = pd.DataFrame({
        "perturbation": [
            "b",
            "control",
            "a",
            "c",
            "a",
            "b",
            "control",
            "c",
            "a",
            "d",
            "d",
            "control",
        ],
        "program_a": np.array([1.0, 0.2, 1.4, -0.3, 2.0, 0.9, 0.4, -0.6, 1.1, 3.0, 2.7, 0.1]),
        "program_b": np.array([0.5, -0.2, 2.4, 1.3, 2.2, 0.7, -0.4, 1.6, 2.1, 0.0, 0.3, -0.1]),
    })
    var = pd.DataFrame(index=["g0", "g1"])
    actual_adata = ad.AnnData(X=np.ones((len(obs), 2)), obs=obs.copy(), var=var)
    expected_adata = ad.AnnData(X=np.ones((len(obs), 2)), obs=obs.copy(), var=var)

    actual = run_program_regression(
        actual_adata,
        perturbationsColumn="perturbation",
        referenceLevel="control",
        pathways=["program_a", "program_b"],
    )
    expected = _reference_run_program_regression(
        expected_adata,
        perturbationsColumn="perturbation",
        referenceLevel="control",
        pathways=["program_a", "program_b"],
    )

    pd.testing.assert_frame_equal(actual, expected, check_exact=False, rtol=1e-12, atol=1e-12)


def test_run_program_regression_matches_statsmodels_reference_default_pathways():
    obs = pd.DataFrame({
        "perturbation": ["control", "x", "y", "x", "control", "y", "z", "z"],
        "program_a": np.array([1.0, 1.2, -0.8, 1.4, 0.7, -0.4, 2.0, 2.2]),
        "program_b": np.array([0.2, 0.4, 1.8, 0.6, -0.1, 1.5, -2.0, -1.8]),
        "label": ["a", "b", "c", "d", "e", "f", "g", "h"],
    })
    var = pd.DataFrame(index=["g0", "g1"])
    actual_adata = ad.AnnData(X=np.ones((len(obs), 2)), obs=obs.copy(), var=var)
    expected_adata = ad.AnnData(X=np.ones((len(obs), 2)), obs=obs.copy(), var=var)

    actual = run_program_regression(
        actual_adata,
        perturbationsColumn="perturbation",
        referenceLevel="control",
    )
    expected = _reference_run_program_regression(
        expected_adata,
        perturbationsColumn="perturbation",
        referenceLevel="control",
    )

    pd.testing.assert_frame_equal(actual, expected, check_exact=False, rtol=1e-12, atol=1e-12)


def _make_streaming_adatas():
    rng = np.random.default_rng(123)
    n_obs = 24
    n_vars = 12
    genes = [f"g{i}" for i in range(n_vars)]
    contexts = np.array(["ctx_a", "ctx_b"] * (n_obs // 2), dtype=object)
    perturbations = np.array(["control", "control", "a", "a", "b", "b"] * 4, dtype=object)

    base = rng.normal(loc=0.0, scale=0.7, size=(n_obs, n_vars)).astype(np.float32)
    real_x = base.copy()
    pred_x = (base * np.float32(0.82)).astype(np.float32)

    real_x[perturbations == "a", :4] += np.float32(0.65)
    real_x[perturbations == "b", 4:8] -= np.float32(0.45)
    real_x[contexts == "ctx_b", 8:] += np.float32(0.3)
    pred_x[perturbations == "a", :4] += np.float32(0.55)
    pred_x[perturbations == "b", 4:8] -= np.float32(0.35)
    pred_x[contexts == "ctx_b", 8:] += np.float32(0.22)

    obs = pd.DataFrame({
        "context": pd.Categorical(contexts, categories=["ctx_a", "ctx_b"]),
        "perturbation": pd.Categorical(perturbations, categories=["control", "a", "b"]),
    }, index=[f"cell_{i}" for i in range(n_obs)])
    var = pd.DataFrame(index=genes)
    programs = {
        "program_0": ["g0"],
        "program_1": ["g4", "g5", "g6"],
        "program_2": ["g8", "g9", "g10"],
    }
    return (
        ad.AnnData(X=pred_x, obs=obs.copy(), var=var.copy()),
        ad.AnnData(X=real_x, obs=obs.copy(), var=var.copy()),
        programs,
        genes,
    )


def _write_h5ad_with_numeric_var_names(adata, path):
    h5ad = adata.copy()
    h5ad.var_names = [str(i) for i in range(h5ad.n_vars)]
    h5ad.write_h5ad(path)


def test_streaming_h5ad_matches_context_split_in_memory(tmp_path):
    pred_adata, real_adata, programs, genes = _make_streaming_adatas()
    pred_path = tmp_path / "pred.h5ad"
    real_path = tmp_path / "real.h5ad"
    _write_h5ad_with_numeric_var_names(pred_adata, pred_path)
    _write_h5ad_with_numeric_var_names(real_adata, real_path)

    results, pred_regressions, real_regressions = streaming_gene_program_concordance(
        pred_path,
        real_path,
        programs,
        perturbationsColumn="perturbation",
        referenceLevel="control",
        contextColumn="context",
        gene_names=genes,
        chunk_size=5,
        ctrl_size=2,
        n_bins=4,
        random_state=42,
        return_regressions=True,
    )
    indexed_results, indexed_pred_regressions, indexed_real_regressions = streaming_gene_program_concordance(
        pred_path,
        real_path,
        programs,
        perturbationsColumn="perturbation",
        referenceLevel="control",
        contextColumn="context",
        gene_names=genes,
        chunk_size=5,
        ctrl_size=2,
        n_bins=4,
        random_state=42,
        score_backend="indexed",
        return_regressions=True,
    )

    expected_results = []
    for context in pred_regressions:
        pred_context = pred_adata[pred_adata.obs["context"] == context].copy()
        real_context = real_adata[real_adata.obs["context"] == context].copy()
        score_all_programs(pred_context, programs, n_jobs=1, ctrl_size=2, n_bins=4, random_state=42)
        score_all_programs(real_context, programs, n_jobs=1, ctrl_size=2, n_bins=4, random_state=42)
        expected_pred = run_program_regression(
            pred_context,
            perturbationsColumn="perturbation",
            referenceLevel="control",
            pathways=programs.keys(),
        )
        expected_real = run_program_regression(
            real_context,
            perturbationsColumn="perturbation",
            referenceLevel="control",
            pathways=programs.keys(),
        )

        pd.testing.assert_frame_equal(
            pred_regressions[context],
            expected_pred,
            check_exact=False,
            rtol=1e-6,
            atol=1e-6,
        )
        pd.testing.assert_frame_equal(
            indexed_pred_regressions[context],
            expected_pred,
            check_exact=False,
            rtol=1e-6,
            atol=1e-6,
        )
        pd.testing.assert_frame_equal(
            real_regressions[context],
            expected_real,
            check_exact=False,
            rtol=1e-6,
            atol=1e-6,
        )
        pd.testing.assert_frame_equal(
            indexed_real_regressions[context],
            expected_real,
            check_exact=False,
            rtol=1e-6,
            atol=1e-6,
        )

        context_results = _coef_correlations_dataframe(expected_pred, expected_real)
        context_results.insert(0, "context", context)
        expected_results.append(context_results)

    expected_results = pd.concat(expected_results, ignore_index=True)
    pd.testing.assert_frame_equal(results, expected_results, check_exact=False, rtol=1e-6, atol=1e-6)
    pd.testing.assert_frame_equal(indexed_results, expected_results, check_exact=False, rtol=1e-6, atol=1e-6)


def test_streaming_h5ad_without_context_matches_whole_in_memory(tmp_path):
    pred_adata, real_adata, programs, _ = _make_streaming_adatas()
    pred_adata.obs = pred_adata.obs.drop(columns=["context"])
    real_adata.obs = real_adata.obs.drop(columns=["context"])
    pred_path = tmp_path / "pred_no_context.h5ad"
    real_path = tmp_path / "real_no_context.h5ad"
    pred_adata.write_h5ad(pred_path)
    real_adata.write_h5ad(real_path)

    results, pred_regressions, real_regressions = streaming_gene_program_concordance(
        pred_path,
        real_path,
        programs,
        perturbationsColumn="perturbation",
        referenceLevel="control",
        contextColumn=None,
        chunk_size=4,
        ctrl_size=2,
        n_bins=4,
        random_state=42,
        return_regressions=True,
    )

    pred_expected_adata = pred_adata.copy()
    real_expected_adata = real_adata.copy()
    score_all_programs(pred_expected_adata, programs, n_jobs=1, ctrl_size=2, n_bins=4, random_state=42)
    score_all_programs(real_expected_adata, programs, n_jobs=1, ctrl_size=2, n_bins=4, random_state=42)
    expected_pred = run_program_regression(
        pred_expected_adata,
        perturbationsColumn="perturbation",
        referenceLevel="control",
        pathways=programs.keys(),
    )
    expected_real = run_program_regression(
        real_expected_adata,
        perturbationsColumn="perturbation",
        referenceLevel="control",
        pathways=programs.keys(),
    )

    pd.testing.assert_frame_equal(
        pred_regressions["all"],
        expected_pred,
        check_exact=False,
        rtol=1e-6,
        atol=1e-6,
    )
    pd.testing.assert_frame_equal(
        real_regressions["all"],
        expected_real,
        check_exact=False,
        rtol=1e-6,
        atol=1e-6,
    )

    expected_results = _coef_correlations_dataframe(expected_pred, expected_real)
    expected_results.insert(0, "context", "all")
    pd.testing.assert_frame_equal(results, expected_results, check_exact=False, rtol=1e-6, atol=1e-6)
