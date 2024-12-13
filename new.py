import matplotlib.pyplot as plt
import numpy as np



# Define the methods and their corresponding accuracies
methods = [
    "No Preprocessing",
    "CVSS-NLMS",
    "SLRF",
    "CVSS-NLMS + SLRF"
]
accuracies = [90.95, 95.22, 94.33, 99.40]

# Plot the ablation study results
plt.figure(figsize=(10, 6))
plt.barh(methods, accuracies, color=['gray', 'blue', 'orange', 'green'], edgecolor='black')

# Annotate each bar with accuracy values
for i, acc in enumerate(accuracies):
    plt.text(acc - 2, i, f"{acc:.2f}%", ha='right', va='center', color="white", fontweight='bold')

# Set plot details
plt.xlabel("Accuracy (%)")
plt.ylabel("Preprocessing Methods")
# plt.title("Ablation Study: Effect of Preprocessing Methods on Model Accuracy")
plt.xlim(80, 100)
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Show plot
plt.tight_layout()
plt.savefig("ablation_study_results.svg", format="svg", dpi=300)
plt.show()
