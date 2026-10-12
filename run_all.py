"""End-to-end: simulate the mechanistic disease cohort, run pathway
enrichment, train the PyTorch classifier, and rank target candidates by
permutation importance."""
import json
import os

from src.simulate_data import simulate, DRIVER_GENES
from src.pathway_analysis import univariate_gene_association, pathway_enrichment
from src.target_model import train_and_evaluate, permutation_importance

os.makedirs("data", exist_ok=True)
os.makedirs("results", exist_ok=True)

print("=== Simulating 340-patient mechanistic disease cohort ===")
df = simulate()
df.to_csv("data/cohort.csv", index=False)
print(df.shape, "disease prevalence:", round(df["disease"].mean(), 3))

print("\n=== Univariate gene-disease association (differential expression) ===")
assoc = univariate_gene_association(df)
print(assoc)
sig_genes = assoc.loc[assoc["significant"], "gene"].tolist()

print("\n=== Pathway enrichment (hypergeometric test) ===")
enrichment = pathway_enrichment(sig_genes)
print(enrichment)

print("\n=== PyTorch mechanistic disease classifier ===")
result = train_and_evaluate(df)
print(f"Accuracy: {result['accuracy']}, n_train={result['n_train']}, n_test={result['n_test']}")

print("\n=== Target identification (permutation importance) ===")
importance = permutation_importance(result["model"], result["X_test"], result["y_test"])
for r in importance[:5]:
    print(r)

top_candidates = [r["feature"] for r in importance[:3]]
driver_genes_recovered = [g for g in DRIVER_GENES if g in top_candidates]

summary = {
    "n_patients": len(df),
    "n_significant_genes_univariate": len(sig_genes),
    "significant_genes": sig_genes,
    "model_accuracy": result["accuracy"],
    "top_3_target_candidates": top_candidates,
    "true_driver_genes": DRIVER_GENES,
    "n_true_drivers_in_top_3": len(driver_genes_recovered),
}
with open("results/summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n=== Summary ===")
print(json.dumps(summary, indent=2))
