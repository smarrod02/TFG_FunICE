"""Figura 2x3 de las componentes del ACPF de una variable: filas = escala (probabilidad, logit),
columnas = componentes 1-3. Cada panel muestra la media mu(t) y mu(t) ± sqrt(lambda_k)·phi_k(t)
(representación de Ramsay y Silverman). Para la memoria (Sección 6.1.3).

Solo lee resultados ya guardados (resultados/curvas_ice.pkl y fpca.pkl); no recalcula el ACPF.
Uso, desde la raíz del proyecto:  python breast_cancer/figura_componentes_2x3.py ["worst texture"]
Genera breast_cancer/figuras/componentes_2x3_<variable>.{png,pdf}.
"""
import os
import sys

import joblib
import numpy as np
import matplotlib.pyplot as plt

CARPETA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(CARPETA))

VAR = sys.argv[1] if len(sys.argv) > 1 else 'worst texture'
RES, FIG = os.path.join(CARPETA, 'resultados'), os.path.join(CARPETA, 'figuras')
M, N_PUNTOS = 3, 300

curvas = joblib.load(os.path.join(RES, 'curvas_ice.pkl'))['suavizadas']
fpca = joblib.load(os.path.join(RES, 'fpca.pkl'))['res_fpca']

ETIQUETA = {'probabilidad': r'$\tilde p$', 'logit': r'$\tilde\eta$'}
fig, axes = plt.subplots(2, M, figsize=(11, 6.6), sharex=True)
for fila, escala in enumerate(('probabilidad', 'logit')):
    fd, res = curvas[escala][VAR], fpca[escala][VAR]
    t = np.linspace(*fd.domain_range[0], N_PUNTOS)
    mu = fd.mean()(t)[0, :, 0]
    for k in range(M):
        ax = axes[fila, k]
        phi = res['fpca'].components_[k](t)[0, :, 0]
        a = np.sqrt(res['autovalores'][k])
        ax.plot(t, mu + a * phi, color='#e34948', linewidth=1.6, label=r'$\mu + \sqrt{\lambda_k}\,\phi_k$')
        ax.plot(t, mu - a * phi, color='#2a78d6', linewidth=1.6, label=r'$\mu - \sqrt{\lambda_k}\,\phi_k$')
        ax.plot(t, mu, color='black', linewidth=2, label=r'$\mu$')
        ax.axhline(0, color='#52514e', linewidth=0.6)
        ax.set_title(rf'Componente {k + 1} ($\rho_{k + 1} = {100 * res["varianza_explicada"][k]:.1f}\,\%$)',
                     fontsize=11)
        ax.grid(alpha=0.3)
    axes[fila, 0].set_ylabel(f'{escala}\n' + ETIQUETA[escala], fontsize=11)
    # misma escala vertical en los tres paneles de cada fila, para comparar amplitudes
    lo = min(ax.get_ylim()[0] for ax in axes[fila]); hi = max(ax.get_ylim()[1] for ax in axes[fila])
    for ax in axes[fila]:
        ax.set_ylim(lo, hi)

for ax in axes[1]:
    ax.set_xlabel(VAR, fontsize=11)
manejadores, etiquetas = axes[0, 0].get_legend_handles_labels()
fig.legend(manejadores, etiquetas, loc='upper center', ncol=3, fontsize=10, frameon=False,
           bbox_to_anchor=(0.5, 1.04))

plt.tight_layout()
os.makedirs(FIG, exist_ok=True)
nombre = 'componentes_2x3_' + VAR.replace(' ', '_')
for ext in ('png', 'pdf'):
    plt.savefig(os.path.join(FIG, f'{nombre}.{ext}'), dpi=200, bbox_inches='tight')
print(f'Guardado: breast_cancer/figuras/{nombre}.{{png,pdf}}')
