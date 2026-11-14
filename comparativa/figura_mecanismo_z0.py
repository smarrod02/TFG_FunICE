"""Figura 1x3 del mecanismo del punto de partida, para los cuatro conjuntos de datos. Eje X común:
corr(z0, logit predicho). Paneles: fracción recuperada; ΔAUC del test incremental en probabilidad; ΔAUC en
logit. Cada punto es una variable; en los paneles de ΔAUC, relleno si p < 0.05. Misma codificación de
conjuntos que figura_auc_residual.py. Para la memoria (Sección 6.2.2).

Solo lee resultados ya guardados (comparativa.pkl); no recalcula nada.
Uso, desde la raíz del proyecto:  python comparativa/figura_mecanismo_z0.py
Genera comparativa/figuras/mecanismo_z0_1x3.{png,pdf}.
"""
import os

import joblib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

CARPETA = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(CARPETA, 'figuras')

d = joblib.load(os.path.join(CARPETA, 'comparativa.pkl'))['df_var']
X = 'corr(z0,logit_real)'

# La misma codificación que figura_auc_residual.py
ESTILO = {'Breast Cancer': ('#e87ba4', 'o'), 'MAGIC': ('#1baf7a', 's'),
          'Adult': ('#eda100', '^'), 'HIGGS': ('#2a78d6', 'D')}
GRIS = '#52514e'

fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharex=True)
paneles = [('logit', 'fraccion_capturada_z0', 'Fracción recuperada', 'Fracción recuperada'),
           ('probabilidad', 'delta_AUC', r'$\Delta\mathrm{AUC}$', r'Test incremental: probabilidad'),
           ('logit', 'delta_AUC', r'$\Delta\mathrm{AUC}$', r'Test incremental: logit')]
for ax, (escala, col, etiqueta, titulo) in zip(axes, paneles):
    de = d[d['escala'] == escala]
    for nombre, (color, marcador) in ESTILO.items():
        g = de[de['dataset'] == nombre]
        if col == 'delta_AUC':
            grupos = ((g[g['p_perm'] < 0.05], color), (g[g['p_perm'] >= 0.05], 'white'))
        else:
            grupos = ((g, color),)
        for s, relleno in grupos:
            ax.scatter(s[X], s[col], s=46, marker=marcador, facecolor=relleno, edgecolor=color,
                       linewidth=1.4, zorder=3)
    rho = stats.spearmanr(de[X], de[col])[0]
    ax.axhline(1 if col == 'fraccion_capturada_z0' else 0, color=GRIS, linewidth=0.8, linestyle='--', zorder=1)
    ax.set_xlabel(r'$\mathrm{corr}(z_0, \hat\eta)$', fontsize=11)
    ax.set_ylabel(etiqueta, fontsize=11)
    ax.set_title(titulo, fontsize=12)
    ax.grid(alpha=0.3)
axes[2].sharey(axes[1])

manejadores = [Line2D([], [], marker=m, linestyle='', markersize=7, markerfacecolor=c,
                      markeredgecolor=c, label=n) for n, (c, m) in ESTILO.items()]
manejadores += [Line2D([], [], marker='o', linestyle='', markersize=7, markerfacecolor=GRIS,
                       markeredgecolor=GRIS, label=r'$p < 0.05$'),
                Line2D([], [], marker='o', linestyle='', markersize=7, markerfacecolor='white',
                       markeredgecolor=GRIS, markeredgewidth=1.4, label=r'$p \geq 0.05$')]
fig.legend(handles=manejadores, loc='upper center', ncol=6, fontsize=9.5, frameon=False,
           bbox_to_anchor=(0.5, 1.08), handletextpad=0.3, columnspacing=1.2)

plt.tight_layout()
os.makedirs(FIG, exist_ok=True)
for ext in ('png', 'pdf'):
    plt.savefig(os.path.join(FIG, f'mecanismo_z0_1x3.{ext}'), dpi=200, bbox_inches='tight')
print('Guardado: comparativa/figuras/mecanismo_z0_1x3.{png,pdf}')
