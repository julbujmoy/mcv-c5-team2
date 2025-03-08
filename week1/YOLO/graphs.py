import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Datos de los modelos
models = ["No Fine-Tune", "Non-Aug", "Color-Aug", "Geometric-Aug", "Advance-Aug", "Default-Aug"]
classes = ["All", "Car", "Pedestrian"]

# Métricas (Precision, Recall, mAP@50, mAP@50-95)
metrics = {
    "Precision": [
        [0.0152, 0.0249, 0.00547],  # No Fine-Tune
        [0.727, 0.679, 0.775],      # Non-Aug
        [0.793, 0.765, 0.822],      # Color-Aug
        [0.768, 0.711, 0.824],      # Geometric-Aug
        [0.792, 0.73, 0.855],       # Advance-Aug
        [0.803, 0.754, 0.851]       # Default-Aug
    ],
    "Recall": [
        [0.163, 0.14, 0.186],
        [0.622, 0.727, 0.516],
        [0.65, 0.816, 0.485],
        [0.649, 0.81, 0.488],
        [0.645, 0.799, 0.492],
        [0.627, 0.8, 0.454]
    ],
    "mAP@50": [
        [0.00904, 0.0144, 0.00372],
        [0.67, 0.727, 0.613],
        [0.709, 0.817, 0.601],
        [0.703, 0.793, 0.612],
        [0.719, 0.794, 0.644],
        [0.706, 0.808, 0.604]
    ],
    "mAP@50-95": [
        [0.00375, 0.00602, 0.00148],
        [0.425, 0.52, 0.331],
        [0.507, 0.653, 0.36],
        [0.472, 0.604, 0.34],
        [0.512, 0.646, 0.378],
        [0.503, 0.649, 0.357]
    ]
}

# Crear gráficos para cada métrica
sns.set(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle("Model Performance Comparison", fontsize=16)

for ax, (metric_name, values) in zip(axes.flatten(), metrics.items()):
    values = np.array(values).T  # Transponer para obtener valores por clase
    x = np.arange(len(models))
    width = 0.2
    
    for i, (cls, cls_values) in enumerate(zip(classes, values)):
        ax.bar(x + i * width, cls_values, width, label=cls)
    
    ax.set_xticks(x + width)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylabel(metric_name)
    ax.set_title(metric_name + " by Model")
    ax.legend()

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
