# bioconcord

The usefulness of in-silico experiments depends on whether they can yield the same biological knowledge as a real Perturb-seq experiment. To measure this, we focus on the kinds of analyses researchers usually perform with Perturb-seq data to study biology and test whether in-silico generated datasets can support the same conclusions. This package provides a framework for systematically evaluating the degree of concordance between biological inferences drawn from experimental perturbation screens and those derived from simulated perturbation screens. 

## 📦 Installation

Clone the repository and install locally:

```bash
git clone git@github.com:ArcInstitute/bioconcord.git
cd bioconcord
pip install -e .

```

## Fast Gene Program Concordance on Large h5ad Files

For one ground-truth AnnData, use the streaming regression API to compute and
optionally save per-context gene-program regression tables:

```python
import pandas as pd
from Src.bioconcord import runGeneProgramRegressionsStreaming

programs_df = pd.read_csv("GeneModules.csv", index_col=0)
programs = programs_df.groupby(programs_df.columns[-1])["GeneName"].apply(
    lambda s: list(pd.unique(s.dropna().astype(str)))
).to_dict()

regressions = runGeneProgramRegressionsStreaming(
    adata_path="/path/to/adata_real.h5ad",
    programs_dict=programs,
    perturbationsColumn="perturbation",
    referenceLevel="control",
    contextColumn="context",
    output_path="adata_real_gene_program_regressions.csv",
    chunk_size=25000,
    n_workers=8,
    worker_blas_threads=1,
)
```

For large dense `.h5ad` prediction/real pairs, use the streaming concordance
evaluator instead of loading both AnnData objects into memory. It reuses the
same regression backend, raw h5py reads, context-aware multiprocessing, and the
matrix scoring backend:

```python
import pandas as pd
from Src.bioconcord import testGeneProgramsConcordanceStreaming

programs_df = pd.read_csv("GeneModules.csv", index_col=0)
programs = programs_df.groupby(programs_df.columns[-1])["GeneName"].apply(
    lambda s: list(pd.unique(s.dropna().astype(str)))
).to_dict()

results = testGeneProgramsConcordanceStreaming(
    pred_adata_path="/path/to/adata_pred.h5ad",
    real_adata_path="/path/to/adata_real.h5ad",
    programs_dict=programs,
    perturbationsColumn="perturbation",
    referenceLevel="control",
    contextColumn="context",
    gene_names="/path/to/var_dims.pkl",  # optional; use when h5ad var names are numeric ids
    chunk_size=25000,
    n_workers=8,                         # try 8-16 on large NFS-backed files
    worker_blas_threads=1,               # avoids CPU oversubscription
)
```

Notes:
- `n_workers=1` preserves serial streaming behavior; `n_workers>1` enables
  deterministic multiprocessing over disjoint row chunks.
- `score_backend="matrix"` is the fastest default. Use
  `score_backend="indexed"` for strict parity checks against the original
  per-program scoring kernel.
- The evaluator auto-splits by `contextColumn` when present, so mixed-context
  h5ads are scored as if they had already been split by context.
- The current streaming implementation supports dense `.X`; CSR `.X` support is
  planned separately.

<img width="317" height="316" alt="bioconcord" src="https://github.com/user-attachments/assets/b3883303-44f5-475c-9ec4-dfd9acb35318" />

