"""Figura 1x2 de las puntuaciones del ACPF de una variable (xi_1 frente a xi_2), coloreadas por clase:
columnas = escala (probabilidad, logit). En cada eje se indica el AUC simétrico de esa componente
frente a la clase. Para la memoria (Sección 6.1.4).

Solo lee resultados ya guardados (resultados/curvas_ice.pkl y fpca.pkl); no recalcula el ACPF.
Uso, desde la raíz del proyecto:  python breast_cancer/figura_scores_1x2.py ["worst texture"]
Genera breast_cancer/figuras/scores_1x2_<variable>.{png,pdf}.
"""
import os
import sys

import joblib
import matplotlib.pyplot as plt

CARPETA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(CARPETA))
from comun.graficos import COLOR_CLASE
from comun.punto_partida import auc_simetrico

VAR = sys.argv[1] if len(sys.argv) > 1 else 'worst texture'
RES, FIG = os.path.join(CARPETA, 'resultados'), os.path.join(CARPETA, 'figuras')

c = joblib.load(os.path.join(RES, 'curvas_ice.pkl'))
fpca = joblib.load(os.path.join(RES, 'fpca.pkl'))['res_fpca']
y, clases = c['y'].astype(int), c['clases']

fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
for ax, escala in zip(axes, ('probabilidad', 'logit')):
    S = fpca[escala][VAR]['scores']
    for k in (0, 1):
        ax.scatter(S[y == k, 0], S[y == k, 1], s=22, alpha=0.75, color=COLOR_CLASE[k],
                   edgecolor='white', linewidth=0.4, label=clases[k])
    ax.axhline(0, color='#52514e', linewidth=0.6); ax.axvline(0, color='#52514e', linewidth=0.6)
    a1, a2 = auc_simetrico(y, S[:, 0]), auc_simetrico(y, S[:, 1])
    ax.set_xlabel(rf'$\xi_1$  ($\mathrm{{AUC}} = {a1:.2f}$)', fontsize=11)
    ax.set_ylabel(rf'$\xi_2$  ($\mathrm{{AUC}} = {a2:.2f}$)', fontsize=11)
    ax.set_title(escala.capitalize(), fontsize=12)
    ax.grid(alpha=0.3)

manejadores, etiquetas = axes[0].get_legend_handles_labels()
fig.legend(manejadores, etiquetas, loc='upper center', ncol=2, fontsize=10, frameon=False,
           bbox_to_anchor=(0.5, 1.05), markerscale=1.5)

plt.tight_layout()
os.makedirs(FIG, exist_ok=True)
nombre = 'scores_1x2_' + VAR.replace(' ', '_')
for ext in ('png', 'pdf'):
    plt.savefig(os.path.join(FIG, f'{nombre}.{ext}'), dpi=200, bbox_inches='tight')
print(f'Guardado: breast_cancer/figuras/{nombre}.{{png,pdf}}')
