"""Diagramas de fiabilidad de los cuatro modelos calibrados (conjunto de test), en una figura 2x2.

Solo lee resultados ya guardados (datos_procesados.pt y mlp_calibrado.pth de cada dataset); no reentrena.
Uso, desde la raíz del proyecto:  python comparativa/figura_fiabilidad.py
Genera comparativa/figuras/fiabilidad_datasets.{png,pdf}.
"""
import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
from comun.modelo import cargar_modelo, logits_de, sigmoide, metricas

DATASETS = [('breast_cancer', 'Breast Cancer'), ('magic_telescope', 'MAGIC'),
            ('adult_income', 'Adult'), ('higgs', 'HIGGS')]
N_BINS = 10
FIG = os.path.join(RAIZ, 'comparativa', 'figuras')

fig, axes = plt.subplots(2, 2, figsize=(9, 9), sharex=True, sharey=True)
for ax, (carpeta, nombre) in zip(axes.flat, DATASETS):
    res = os.path.join(RAIZ, carpeta, 'resultados')
    datos = torch.load(os.path.join(res, 'datos_procesados.pt'), weights_only=False)
    X, y = datos['X_test'], datos['y_test'].numpy().reshape(-1)
    modelo = cargar_modelo(os.path.join(res, 'mlp_calibrado.pth'), X.shape[1])
    z = logits_de(modelo, X)
    p = sigmoide(z)

    # mismos intervalos que el ECE de comun/modelo.metricas
    bins = np.linspace(0, 1, N_BINS + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, N_BINS - 1)
    ocupados = [b for b in range(N_BINS) if (idx == b).any()]
    xs = np.array([p[idx == b].mean() for b in ocupados])
    ys = np.array([y[idx == b].mean() for b in ocupados])
    ns = np.array([(idx == b).sum() for b in ocupados])

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='calibración perfecta')
    ax.plot(xs, ys, color='#2a78d6', alpha=0.6, linewidth=1.2)
    ax.scatter(xs, ys, s=15 + 400 * ns / ns.max(), color='#2a78d6', alpha=0.7,
               edgecolor='white', linewidth=0.5, zorder=3)

    ece = metricas(z, y, n_bins=N_BINS)['ECE']
    ax.set_title(f'{nombre}', fontsize=12, style='italic')
    ax.text(0.97, 0.04, f'$m = {len(y)}$\n$\\mathrm{{ECE}} = {ece:.3f}$', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='0.8'))
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02); ax.set_aspect('equal')
    ax.grid(alpha=0.3)
    print(f'{nombre:14s} m={len(y):6d}  ECE={ece:.3f}  instancias por intervalo: {ns.tolist()}')

for ax in axes[1]:
    ax.set_xlabel('probabilidad predicha media', fontsize=11)
for ax in axes[:, 0]:
    ax.set_ylabel('fracción real de la clase positiva', fontsize=11)
axes[0, 0].legend(loc='upper left', fontsize=9)

plt.tight_layout()
os.makedirs(FIG, exist_ok=True)
for ext in ('png', 'pdf'):
    plt.savefig(os.path.join(FIG, f'fiabilidad_datasets.{ext}'), dpi=200, bbox_inches='tight')
print('Guardado: comparativa/figuras/fiabilidad_datasets.{png,pdf}')
