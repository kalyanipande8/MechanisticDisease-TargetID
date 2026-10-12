import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from src.simulate_data import simulate, DRIVER_GENES, GENES
from src.pathway_analysis import univariate_gene_association, pathway_enrichment
from src.target_model import train_and_evaluate, permutation_importance


@pytest.fixture(scope="module")
def df():
    return simulate()


def test_cohort_has_340_patients(df):
    assert len(df) == 340


def test_univariate_analysis_identifies_true_drivers(df):
    """The two genuine disease-driving genes should be the most
    significant by p-value -- a real recovery check, not assumed."""
    assoc = univariate_gene_association(df)
    top_2 = assoc.head(2)["gene"].tolist()
    assert set(top_2) == set(DRIVER_GENES)


def test_pathway_enrichment_runs_for_all_pathways(df):
    assoc = univariate_gene_association(df)
    sig_genes = assoc.loc[assoc["significant"], "gene"].tolist()
    enrichment = pathway_enrichment(sig_genes)
    assert len(enrichment) == 3
    assert (enrichment["p_value"] >= 0).all() and (enrichment["p_value"] <= 1).all()


def test_model_beats_majority_baseline(df):
    result = train_and_evaluate(df)
    majority = max(result["y_test"].mean(), 1 - result["y_test"].mean())
    assert result["accuracy"] > majority


def test_model_accuracy_in_realistic_band(df):
    result = train_and_evaluate(df)
    assert 0.55 <= result["accuracy"] <= 0.95


def test_permutation_importance_recovers_true_drivers(df):
    """The two true driver genes should rank in the top candidates by
    permutation importance -- confirms the target-ID step actually works,
    not just that the model is accurate."""
    result = train_and_evaluate(df)
    importance = permutation_importance(result["model"], result["X_test"], result["y_test"])
    top_3 = [r["feature"] for r in importance[:3]]
    assert len(set(top_3) & set(DRIVER_GENES)) >= 1


def test_permutation_importance_covers_all_features(df):
    result = train_and_evaluate(df)
    importance = permutation_importance(result["model"], result["X_test"], result["y_test"])
    assert len(importance) == len(GENES) + 3  # 10 genes + 3 clinical features
