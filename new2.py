import matplotlib.pyplot as plt
import numpy as np
import time
import psutil
import os

# Dummy computational efficiency data for illustration
methods = [
    "No Preprocessing",
    "CVSS-NLMS",
    "SLRF",
    "CVSS-NLMS + SLRF"
]
accuracies = [90.95, 95.22, 94.33, 99.40]
execution_time = [5.2, 6.5, 6.8, 7.3]  # Time in milliseconds per signal
memory_usage = [120, 150, 145, 180]  # Peak memory usage in MB
inference_time = [0.8, 1.0, 1.1, 1.2]  # Time in milliseconds per signal

# Plotting Accuracy vs Computational Metrics
plt.figure(figsize=(14, 8))

# Subplot 1: Accuracy Comparison
plt.subplot(2, 2, 1)
plt.barh(methods, accuracies, color=['gray', 'blue', 'orange', 'green'], edgecolor='black')
plt.xlabel("Accuracy (%)")
plt.title("Model Accuracy with Preprocessing")
for i, acc in enumerate(accuracies):
    plt.text(acc - 5, i, f"{acc:.2f}%", ha='right', va='center', color="white", fontweight='bold')

# Subplot 2: Execution Time
plt.subplot(2, 2, 2)
plt.barh(methods, execution_time, color=['gray', 'blue', 'orange', 'green'], edgecolor='black')
plt.xlabel("Execution Time (ms per signal)")
plt.title("Preprocessing Execution Time")
for i, et in enumerate(execution_time):
    plt.text(et - 0.5, i, f"{et:.1f} ms", ha='right', va='center', color="white", fontweight='bold')

# Subplot 3: Memory Usage
plt.subplot(2, 2, 3)
plt.barh(methods, memory_usage, color=['gray', 'blue', 'orange', 'green'], edgecolor='black')
plt.xlabel("Memory Usage (MB)")
plt.title("Peak Memory Usage During Preprocessing")
for i, mu in enumerate(memory_usage):
    plt.text(mu - 10, i, f"{mu} MB", ha='right', va='center', color="white", fontweight='bold')

# Subplot 4: Inference Time
plt.subplot(2, 2, 4)
plt.barh(methods, inference_time, color=['gray', 'blue', 'orange', 'green'], edgecolor='black')
plt.xlabel("Inference Time (ms per signal)")
plt.title("Model Inference Time")
for i, it in enumerate(inference_time):
    plt.text(it - 0.2, i, f"{it:.1f} ms", ha='right', va='center', color="white", fontweight='bold')

plt.tight_layout()
plt.savefig("computational_efficiency_metrics.svg", format="svg", dpi=300)
plt.show()
