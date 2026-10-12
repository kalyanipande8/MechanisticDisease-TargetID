"""Real pathway enrichment analysis: identify genes individually
associated with disease status (univariate test), then run a
hypergeometric over-representation test for each candidate pathway --
the standard gene-set enrichment methodology (Fisher's exact / hypergeometric
test, as in DAVID/GSEA-style tools), computed from scratch with scipy."""
from __future__ import annotations

import pandas as pd
from scipy import stats

from .simulate_data import GENES, PATHWAYS


def univariate_gene_association(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Two-sample t-test of each gene's expression, disease vs. no
    disease -- the standard first-pass differential-expression screen."""
    rows = []
    diseased = df[df["disease"] == 1]
    healthy = df[df["disease"] == 0]
    for gene in GENES:
        t_stat, p_value = stats.ttest_ind(diseased[gene], healthy[gene])
        rows.append({"gene": gene, "t_statistic": round(float(t_stat), 4),
                      "p_value": round(float(p_value), 6), "significant": p_value < alpha})
    return pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)


def pathway_enrichment(significant_genes: list[str], total_genes: list[str] = GENES,
                        pathways: dict = PATHWAYS) -> pd.DataFrame:
    """Hypergeometric over-representation test: is a pathway's gene set
    enriched for significant genes more than expected by chance?"""
    M = len(total_genes)       # total genes in the universe
    n_sig = len(significant_genes)  # number of significant genes found

    rows = []
    for pathway, genes in pathways.items():
        N = len(genes)  # pathway size
        k = len(set(genes) & set(significant_genes))  # overlap
        # hypergeometric survival function: P(X >= k)
        p_value = stats.hypergeom.sf(k - 1, M, n_sig, N) if k > 0 else 1.0
        rows.append({
            "pathway": pathway, "pathway_size": N, "n_significant_in_pathway": k,
            "n_significant_total": n_sig, "p_value": round(float(p_value), 5),
            "enriched": p_value < 0.05,
        })
    return pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)


if __name__ == "__main__":
    from .simulate_data import simulate

    df = simulate()
    assoc = univariate_gene_association(df)
    print(assoc)
    sig_genes = assoc.loc[assoc["significant"], "gene"].tolist()
    print("\nSignificant genes:", sig_genes)
    print(pathway_enrichment(sig_genes))
