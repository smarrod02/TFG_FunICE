"""Figura 1x2 del AUC bruto frente al AUC residual de las dos primeras componentes, para los cuatro
conjuntos de datos: columnas = escala (probabilidad, logit). Cada punto es un par (variable, componente);
relleno si el intervalo bootstrap del residuo queda por encima de 0.5. Para la memoria (Sección 6.2.1).

Solo lee resultados ya guardados (comparativa.pkl); no recalcula nada.
Uso, desde la raíz del proyecto:  python comparativa/figura_auc_residual.py
Genera comparativa/figuras/auc_bruto_vs_residual.{png,pdf}.
"""
import os

import joblib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

CARPETA = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(CARPETA, 'figuras')

d = joblib.load(os.path.join(CARPETA, 'comparativa.pkl'))['df_comp']
d = d[d['componente'] <= 2]

# Distintos de los colores de clase; la forma del marcador distingue además el conjunto
ESTILO = {'Breast Cancer': ('#e87ba4', 'o'), 'MAGIC': ('#1baf7a', 's'),
          'Adult': ('#eda100', '^'), 'HIGGS': ('#2a78d6', 'D')}

fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), sharex=True, sharey=True)
for ax, escala in zip(axes, ('probabilidad', 'logit')):
    ax.plot([0.45, 1.0], [0.45, 1.0], color='#52514e', linewidth=0.8, linestyle='--', zorder=1)
    ax.axhline(0.5, color='#52514e', linewidth=0.8, linestyle=':', zorder=1)
    for nombre, (color, marcador) in ESTILO.items():
        g = d[(d['escala'] == escala) & (d['dataset'] == nombre)]
        for signif, relleno in ((True, color), (False, 'white')):
            s = g[g['residual_significativo'] == signif]
            ax.scatter(s['AUC_bruto'], s['AUC_residual'], s=46, marker=marcador, facecolor=relleno,
                       edgecolor=color, linewidth=1.4, zorder=3)
    ax.set_xlim(0.48, 1.0); ax.set_ylim(0.48, 1.0)
    ax.set_aspect('equal')
    ax.set_xlabel(r'$\mathrm{AUC}_{\mathrm{bruto}}$', fontsize=11)
    ax.set_title(escala.capitalize(), fontsize=12)
    ax.grid(alpha=0.3)
axes[0].set_ylabel(r'$\mathrm{AUC}_{\mathrm{residual}}$', fontsize=11)
axes[0].text(0.90, 0.955, r'$y = x$', fontsize=9, color='#52514e', ha='right', va='top')

manejadores = [Line2D([], [], marker=m, linestyle='', markersize=7, markerfacecolor=c,
                      markeredgecolor=c, label=n) for n, (c, m) in ESTILO.items()]
manejadores += [Line2D([], [], marker='o', linestyle='', markersize=7, markerfacecolor='#52514e',
                       markeredgecolor='#52514e', label='IC por encima de 0.5'),
                Line2D([], [], marker='o', linestyle='', markersize=7, markerfacecolor='white',
                       markeredgecolor='#52514e', markeredgewidth=1.4, label='IC incluye 0.5')]
fig.legend(handles=manejadores, loc='upper center', ncol=6, fontsize=9.5, frameon=False,
           bbox_to_anchor=(0.5, 1.08), handletextpad=0.3, columnspacing=1.2)

plt.tight_layout()
os.makedirs(FIG, exist_ok=True)
for ext in ('png', 'pdf'):
    plt.savefig(os.path.join(FIG, f'auc_bruto_vs_residual.{ext}'), dpi=200, bbox_inches='tight')
print('Guardado: comparativa/figuras/auc_bruto_vs_residual.{png,pdf}')
