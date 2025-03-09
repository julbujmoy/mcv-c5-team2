import matplotlib.pyplot as plt
import numpy as np

# Data
models = [
    "Pretrained YOLO", "No Aug", "Color Aug", "Geometric Aug", "Advanced Aug", "Default Aug"
]

classes = ["All", "Car", "Pedestrian"]

precision = [
    [0.738, 0.746, 0.731],  # Pretrained YOLO
    [0.727, 0.679, 0.775],  # No Aug
    [0.793, 0.765, 0.822],  # Color Aug
    [0.768, 0.711, 0.824],  # Geometric Aug
    [0.768, 0.711, 0.824],  # Advanced Aug
    [0.803, 0.754, 0.851]   # Default Aug
]

recall = [
    [0.664, 0.687, 0.641],  # Pretrained YOLO
    [0.622, 0.727, 0.516],  # No Aug
    [0.650, 0.816, 0.485],  # Color Aug
    [0.649, 0.810, 0.488],  # Geometric Aug
    [0.649, 0.810, 0.488],  # Advanced Aug
    [0.627, 0.800, 0.454]   # Default Aug
]

map50 = [
    [0.712, 0.748, 0.677],  # Pretrained YOLO
    [0.670, 0.727, 0.613],  # No Aug
    [0.709, 0.817, 0.601],  # Color Aug
    [0.703, 0.793, 0.612],  # Geometric Aug
    [0.703, 0.793, 0.612],  # Advanced Aug
    [0.706, 0.808, 0.604]   # Default Aug
]

map50_95 = [
    [0.494, 0.568, 0.421],  # Pretrained YOLO
    [0.425, 0.520, 0.331],  # No Aug
    [0.507, 0.653, 0.360],  # Color Aug
    [0.472, 0.604, 0.340],  # Geometric Aug
    [0.472, 0.604, 0.340],  # Advanced Aug
    [0.503, 0.649, 0.357]   # Default Aug
]

metrics = [precision, recall, map50, map50_95]
metric_names = ["Precision (P)", "Recall (R)", "mAP@50", "mAP@50-95"]

# Graph
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

x = np.arange(len(models))
width = 0.25

# Subgraphs
for idx, ax in enumerate(axes):
    for i, cls in enumerate(classes):
        ax.bar(x + i * width, [metrics[idx][m][i] for m in range(len(models))], width, label=cls)

    ax.set_title(metric_names[idx])
    ax.set_xticks(x + width)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylabel(metric_names[idx])
    ax.legend()

plt.tight_layout()
plt.show()
