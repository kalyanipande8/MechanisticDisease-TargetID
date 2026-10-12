"""Simulate a 340-patient cohort with molecular (10-gene expression panel,
2 genuinely disease-driving) and clinical features, integrated into one
mechanistic disease model with a nonlinear outcome."""
from __future__ import annotations

import numpy as np
import pandas as pd

N_PATIENTS = 340
SEED = 97

GENES = ["GENE_A", "GENE_B", "GENE_C", "GENE_D", "GENE_E", "GENE_F", "GENE_G", "GENE_H", "GENE_I", "GENE_J"]
CLINICAL = ["age", "bmi", "inflammatory_marker"]

# ground truth: only GENE_C and GENE_G are genuinely disease-driving (part
# of the same simulated pathway), the rest are noise -- this is what the
# pathway analysis and target-validation steps should recover
DRIVER_GENES = ["GENE_C", "GENE_G"]

PATHWAYS = {
    "Pathway_Inflammation": ["GENE_A", "GENE_C", "GENE_G", "GENE_H"],
    "Pathway_Metabolism": ["GENE_B", "GENE_D", "GENE_I"],
    "Pathway_CellCycle": ["GENE_E", "GENE_F", "GENE_J"],
}


def simulate(n: int = N_PATIENTS, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    expr = {g: rng.normal(0, 1, n) for g in GENES}

    age = rng.normal(55, 13, n).clip(18, 90)
    bmi = rng.normal(27, 5, n).clip(16, 50)
    inflammatory_marker = rng.exponential(2, n).clip(0, 20)

    logit = (
        -0.5
        + 2.2 * expr["GENE_C"]
        + 1.9 * expr["GENE_G"]
        + 0.7 * (expr["GENE_C"] * expr["GENE_G"])  # pathway interaction
        + 0.015 * (age - 55)
        + 0.2 * (inflammatory_marker - 2)
        + rng.normal(0, 0.35, n)
    )
    prob = 1 / (1 + np.exp(-logit))
    disease = rng.binomial(1, prob)

    df = pd.DataFrame({**expr, "age": age, "bmi": bmi, "inflammatory_marker": inflammatory_marker, "disease": disease})
    return df


if __name__ == "__main__":
    df = simulate()
    print(df.shape)
    print("Disease prevalence:", df["disease"].mean())
