"""Figura 2x2 de las curvas ICE de una variable: filas = escala (probabilidad, logit),
columnas = sin centrar / centradas en t_1. Para la memoria (Sección 6.1.2).

Solo lee resultados ya guardados (resultados/curvas_ice.pkl y datos_procesados.pt); no reentrena.
Uso, desde la raíz del proyecto:  python breast_cancer/figura_ice_2x2.py ["worst texture"]
Genera breast_cancer/figuras/ice_2x2_<variable>.{png,pdf}.
"""
import os
import sys

import joblib
import numpy as np
import torch
import matplotlib.pyplot as plt

CARPETA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(CARPETA))
from comun.graficos import COLOR_CLASE, GRIS

VAR = sys.argv[1] if len(sys.argv) > 1 else 'worst texture'
RES, FIG = os.path.join(CARPETA, 'resultados'), os.path.join(CARPETA, 'figuras')

c = joblib.load(os.path.join(RES, 'curvas_ice.pkl'))
datos = torch.load(os.path.join(RES, 'datos_procesados.pt'), weights_only=False)
y, clases = c['y'].astype(int), c['clases']
x = c['eje_x'][VAR]

# valores observados de la variable en la muestra ICE, en unidades originales (marcas en el eje)
media, desv = datos['desescalado'][VAR]
X = datos['X_test'][c['idx_sub']]
x_obs = X[:, datos['variables'].index(VAR)].numpy() * desv + media

ETIQUETA = {'probabilidad': r'$\hat p$', 'logit': r'$\hat\eta$'}
fig, axes = plt.subplots(2, 2, figsize=(9, 7.2), sharex=True)
for fila, escala in enumerate(('probabilidad', 'logit')):
    crudas = c['ice_crudas'][VAR][escala]
    centradas = crudas - crudas[:, [0]]          # centrado en el primer punto de la malla, t_1
    for col, (M, centrado) in enumerate(((crudas, False), (centradas, True))):
        ax = axes[fila, col]
        for k in (0, 1):
            ax.plot(x, M[y == k].T, color=COLOR_CLASE[k], alpha=0.25, linewidth=0.7)
            ax.plot([], [], color=COLOR_CLASE[k], linewidth=1.5, label=clases[k])
        ax.plot(x, M.mean(axis=0), color='black', linewidth=2, label='PDP')
        if not centrado:
            lo = ax.get_ylim()[0]
            ax.plot(x_obs, np.full(len(x_obs), lo), '|', color=GRIS, alpha=0.6, markersize=8)
        if centrado:
            ax.axhline(0, color=GRIS, linewidth=0.6)
        ax.set_ylabel(ETIQUETA[escala] + (' centrado' if centrado else ''), fontsize=11)
        ax.grid(alpha=0.3)

axes[0, 0].set_title('Sin centrar', fontsize=12)
axes[0, 1].set_title(r'Centradas en $t_1$', fontsize=12)
for ax in axes[1]:
    ax.set_xlabel(f'{VAR}', fontsize=11)
manejadores, etiquetas = axes[0, 0].get_legend_handles_labels()
fig.legend(manejadores, etiquetas, loc='upper center', ncol=3, fontsize=10, frameon=False,
           bbox_to_anchor=(0.5, 1.03))

plt.tight_layout()
os.makedirs(FIG, exist_ok=True)
nombre = 'ice_2x2_' + VAR.replace(' ', '_')
for ext in ('png', 'pdf'):
    plt.savefig(os.path.join(FIG, f'{nombre}.{ext}'), dpi=200, bbox_inches='tight')
print(f'Guardado: breast_cancer/figuras/{nombre}.{{png,pdf}}  ({len(y)} curvas, {len(x)} puntos de malla)')
