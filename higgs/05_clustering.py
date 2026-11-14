"""Etapa 4 de FunICE (II): clustering funcional en las dos escalas.

k-centros funcionales proyectados (Dabo-Niang et al., 2008; Chiou y Li, 2007), implementados en
`comun/clustering.py`. Cada clúster tiene su propia media y su propio ACPF local, y cada curva se asigna al clúster
que mejor la reconstruye (error L2). Se prueba K en `VALORES_K` (criterio del codo) y se analiza `K_ELEGIDO`.

Lee resultados/curvas_ice.pkl; escribe resultados/clustering.pkl.
"""
import sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # rutas relativas a la carpeta del script
import matplotlib
matplotlib.use('Agg')  # las figuras se guardan en disco, no se muestran
sys.path.insert(0, '..')
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import joblib

import config as cfg

RES, FIG = 'resultados', 'figuras'
os.makedirs(RES, exist_ok=True); os.makedirs(FIG, exist_ok=True)
np.random.seed(cfg.SEMILLA); torch.manual_seed(cfg.SEMILLA)
pd.set_option('display.width', 180); pd.set_option('display.max_columns', 30)
print('Dataset:', cfg.NOMBRE)

from comun.clustering import clustering_todas
from comun import graficos

ice = joblib.load(f'{RES}/curvas_ice.pkl')
variables_top, y, clases = ice['variables_top'], ice['y'], ice['clases']
res_cl = clustering_todas(ice['suavizadas'], variables_top, y, valores_k=cfg.VALORES_K, semilla=cfg.SEMILLA)

filas = [{'escala': e, 'variable': v, 'K': K, 'convergió': r['convergio'], 'iteraciones': r['n_iteraciones'],
          'error_reconstruccion': r['error_reconstruccion'], 'ARI_vs_clase': r['ARI_vs_clase'],
          'tamaños': [int(np.sum(r['labels'] == c)) for c in range(K)]}
         for e in res_cl for v in variables_top for K, r in res_cl[e][v].items()]
df_cl = pd.DataFrame(filas)
print(df_cl.round(4).to_string(index=False))


# ----------------------------------------------------------------------------------------------------
# 1. Criterio del codo
# ----------------------------------------------------------------------------------------------------

fig, axes = plt.subplots(2, len(variables_top), figsize=(4.2 * len(variables_top), 6.5))
for i, escala in enumerate(('probabilidad', 'logit')):
    for ax, v in zip(axes[i], variables_top):
        d = df_cl[(df_cl['escala'] == escala) & (df_cl['variable'] == v)]
        ax.plot(d['K'], d['error_reconstruccion'], marker='o', color=graficos.COLOR_ESCALA[escala])
        ax.set_title(f'{v} ({escala})', fontsize=10); ax.set_xticks(list(cfg.VALORES_K)); ax.grid(alpha=0.3)
    axes[i][0].set_ylabel('error de reconstrucción')
for ax in axes[1]: ax.set_xlabel('K')
plt.tight_layout(); plt.savefig(f'{FIG}/05_codo.png', dpi=110, bbox_inches='tight'); plt.close()


# ----------------------------------------------------------------------------------------------------
# 2. Clústeres para `K_ELEGIDO`
# ----------------------------------------------------------------------------------------------------

for escala in ('probabilidad', 'logit'):
    graficos.clusters(ice['suavizadas'][escala], res_cl[escala], variables_top, cfg.K_ELEGIDO, escala)
    plt.savefig(f'{FIG}/05_clusters_{escala}.png', dpi=110, bbox_inches='tight'); plt.close()

for escala in ('probabilidad', 'logit'):
    print(f'================ {escala} ================')
    for v in variables_top:
        r = res_cl[escala][v][cfg.K_ELEGIDO]
        tabla = pd.crosstab(pd.Series(r['labels'], name='clúster'), pd.Series([clases[c] for c in y], name='clase'))
        print(f'--- {v} (ARI = {r["ARI_vs_clase"]:.3f})'); print(tabla); print()

# Se guardan etiquetas, medias locales y métricas (no las matrices de distancias completas)
ligero = {e: {v: {K: {k: r[k] for k in ('labels', 'K', 'convergio', 'n_iteraciones', 'error_reconstruccion',
                                          'ARI_vs_clase', 'info_clusters')}
                      for K, r in res_cl[e][v].items()} for v in variables_top} for e in res_cl}
joblib.dump({'res_clustering': ligero, 'df_clustering': df_cl}, f'{RES}/clustering.pkl')
print('Guardado: clustering.pkl')
