"""Figura 1x2 del clustering funcional (G=2) de una variable: columnas = escala (probabilidad, logit).
Curvas ICE centradas coloreadas por grupo, con la curva media de cada grupo; la leyenda indica el
tamaño del grupo y su proporción de la clase positiva. Para la memoria (Sección 6.1.3).

Solo lee resultados ya guardados (resultados/curvas_ice.pkl y clustering.pkl); no recalcula nada.
Uso, desde la raíz del proyecto:  python breast_cancer/figura_clusters_1x2.py ["worst texture"]
Genera breast_cancer/figuras/clusters_1x2_<variable>.{png,pdf}.
"""
import os
import sys

import joblib
import numpy as np
import matplotlib.pyplot as plt

CARPETA = os.path.dirname(os.path.abspath(__file__))
VAR = sys.argv[1] if len(sys.argv) > 1 else 'worst texture'
G = 2
RES, FIG = os.path.join(CARPETA, 'resultados'), os.path.join(CARPETA, 'figuras')
COLOR_GRUPO = ('#2ca02c', '#8e44ad')   # distintos de los colores de clase, para no confundirlos

c = joblib.load(os.path.join(RES, 'curvas_ice.pkl'))
cl = joblib.load(os.path.join(RES, 'clustering.pkl'))['res_clustering']
y, clases, x = c['y'].astype(int), c['clases'], c['eje_x'][VAR]

ETIQUETA = {'probabilidad': r'$\tilde p$', 'logit': r'$\tilde\eta$'}
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
for ax, escala in zip(axes, ('probabilidad', 'logit')):
    crudas = c['ice_crudas'][VAR][escala]
    centradas = crudas - crudas[:, [0]]
    res = cl[escala][VAR][G]
    labels = res['labels']
    # grupos ordenados por su cambio final medio, para que el "grupo 1" sea siempre el de menor cambio
    orden = np.argsort([centradas[labels == g, -1].mean() for g in range(G)])
    for i, g in enumerate(orden):
        m = labels == g
        ax.plot(x, centradas[m].T, color=COLOR_GRUPO[i], alpha=0.2, linewidth=0.7)
        ax.plot(x, centradas[m].mean(axis=0), color=COLOR_GRUPO[i], linewidth=2.4,
                label=f'Grupo {i + 1} ($n={m.sum()}$; {100 * y[m].mean():.0f} % {clases[1]}s)')
    ax.axhline(0, color='#52514e', linewidth=0.6)
    ax.set_title(f'{escala.capitalize()} (ARI con la clase: {res["ARI_vs_clase"]:.2f})', fontsize=11)
    ax.set_xlabel(VAR, fontsize=11)
    ax.set_ylabel(ETIQUETA[escala] + ' centrado', fontsize=11)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(alpha=0.3)

plt.tight_layout()
os.makedirs(FIG, exist_ok=True)
nombre = 'clusters_1x2_' + VAR.replace(' ', '_')
for ext in ('png', 'pdf'):
    plt.savefig(os.path.join(FIG, f'{nombre}.{ext}'), dpi=200, bbox_inches='tight')
print(f'Guardado: breast_cancer/figuras/{nombre}.{{png,pdf}}')
