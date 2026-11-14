"""Comparación de las escalas de probabilidad y logit en este conjunto de datos (`comun/escalas.py`).

1. Saturación: qué parte de las curvas oculta la sigmoide (curvas planas en probabilidad que sí varían en logit).
2. Suavizado: hiperparámetros elegidos por GCV y error de la representación B-spline.
3. ACPF: varianza explicada, parecido de las autofunciones (`sim_autofuncion`) y de las puntuaciones
   (`sim_scores`) entre escalas, y separación de clases de cada componente.
4. Punto de partida: resultados del test incremental en cada escala.
5. Clustering: coincidencia de las particiones de ambas escalas (ARI) y relación de cada una con la clase.

Lee resultados/curvas_ice.pkl, fpca.pkl, clustering.pkl y punto_partida.pkl;
escribe resultados/logit_vs_probabilidad.pkl.
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

from comun.escalas import saturacion, comparar_fpca, comparar_clustering
from comun import graficos

ice = joblib.load(f'{RES}/curvas_ice.pkl')
res_fpca = joblib.load(f'{RES}/fpca.pkl')['res_fpca']
res_cl = joblib.load(f'{RES}/clustering.pkl')['res_clustering']
pp = joblib.load(f'{RES}/punto_partida.pkl')
variables_top, y = ice['variables_top'], ice['y']
crudas = {v: {**ice['ice_crudas'][v], 'eje_x': ice['eje_x'][v]} for v in variables_top}


# ----------------------------------------------------------------------------------------------------
# 1. Saturación de la sigmoide
# ----------------------------------------------------------------------------------------------------

df_sat = saturacion(crudas, variables_top)
print(df_sat.round(3).to_string(index=False))


# ----------------------------------------------------------------------------------------------------
# `frac_planas_prob_pero_no_logit` cuenta las curvas que en probabilidad apenas se mueven (rango < 0.02) mientras que su logit cambia en más de una unidad. En probabilidad, la FPCA no puede distinguir esas curvas entre sí; en logit, sí.
# ----------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------
# 2. Suavizado
# ----------------------------------------------------------------------------------------------------

df_suav = pd.concat([ice['hiperparametros'][e].assign(escala=e) for e in ('probabilidad', 'logit')])
print(df_suav[['escala', 'variable', 'n_basis', 'lambda', 'error_max', 'techo_rejilla']].to_string(index=False))


# ----------------------------------------------------------------------------------------------------
# 3. FPCA en las dos escalas
# ----------------------------------------------------------------------------------------------------

df_fpca_cmp = comparar_fpca(res_fpca, variables_top, y, ice['eje_x'])
print(df_fpca_cmp.round(3).to_string(index=False))

fig, axes = plt.subplots(1, len(variables_top), figsize=(4.2 * len(variables_top), 3.8), sharey=True)
for ax, v in zip(axes, variables_top):
    d = df_fpca_cmp[df_fpca_cmp['variable'] == v]
    w = 0.38
    ax.bar(d['componente'] - w / 2, d['var_expl_prob'], w, color=graficos.COLOR_ESCALA['probabilidad'], label='probabilidad')
    ax.bar(d['componente'] + w / 2, d['var_expl_logit'], w, color=graficos.COLOR_ESCALA['logit'], label='logit')
    ax.set_xticks([1, 2, 3]); ax.set_xlabel('componente'); ax.set_title(v); ax.grid(axis='y', alpha=0.3)
axes[0].set_ylabel('varianza explicada'); axes[0].legend(fontsize=8)
plt.tight_layout(); plt.savefig(f'{FIG}/07_varianza_escalas.png', dpi=110, bbox_inches='tight'); plt.close()


# ----------------------------------------------------------------------------------------------------
# 4. Punto de partida en cada escala
# ----------------------------------------------------------------------------------------------------

print(pp['resumen'].to_string()); print()
cmp_pp = (pp['df_var']['probabilidad'][['variable', 'delta_AUC', 'p_perm']]
          .merge(pp['df_var']['logit'][['variable', 'delta_AUC', 'p_perm']], on='variable', suffixes=('_prob', '_logit')))
print(cmp_pp.round(3).to_string(index=False))


# ----------------------------------------------------------------------------------------------------
# 5. Clustering en cada escala
# ----------------------------------------------------------------------------------------------------

df_cl_cmp = comparar_clustering(res_cl, variables_top, cfg.K_ELEGIDO)
print(df_cl_cmp.round(3).to_string(index=False))


# ----------------------------------------------------------------------------------------------------
# 6. Tabla resumen: probabilidad frente a logit en este dataset
# ----------------------------------------------------------------------------------------------------

def por_escala(f):
    return {'probabilidad': f('probabilidad'), 'logit': f('logit')}

dv, dc = pp['df_var'], pp['df_comp']
c1 = df_fpca_cmp[df_fpca_cmp['componente'] == 1]
tabla = pd.DataFrame({
    'curvas ocultas por saturación (mediana entre variables)': {
        'probabilidad': round(df_sat['frac_planas_prob_pero_no_logit'].median(), 3), 'logit': 0.0},
    'varianza comp1 (mediana)': {'probabilidad': round(c1['var_expl_prob'].median(), 3),
                                 'logit': round(c1['var_expl_logit'].median(), 3)},
    'mejor AUC de clase de un score (mediana entre variables)': {
        'probabilidad': round(df_fpca_cmp.groupby('variable')['AUC_prob'].max().median(), 3),
        'logit': round(df_fpca_cmp.groupby('variable')['AUC_logit'].max().median(), 3)},
    'ΔAUC z0+forma (mediana)': por_escala(lambda e: round(dv[e]['delta_AUC'].median(), 3)),
    'vars con ΔAUC signif.': por_escala(lambda e: int((dv[e]['p_perm'] < 0.05).sum())),
    'pares residual signif.': por_escala(lambda e: int(dc[e]['residual_significativo'].sum())),
    'ARI clúster-clase (mediana)': {'probabilidad': round(df_cl_cmp['ARI_clase_prob'].median(), 3),
                                    'logit': round(df_cl_cmp['ARI_clase_logit'].median(), 3)},
    'rango mediano de una curva (mediana entre variables)': {
        'probabilidad': round(df_sat['rango_mediano_prob'].median(), 3),
        'logit': round(df_sat['rango_mediano_logit'].median(), 3)},
    'rango máximo de una curva (mediana entre variables)': {
        'probabilidad': round(df_sat['rango_max_prob'].median(), 3),
        'logit': round(df_sat['rango_max_logit'].median(), 3)},
}).T
print(f'Concordancia entre escalas: sim. autofunción comp1 (mediana) = {c1["sim_autofuncion"].median():.3f}, '
      f'sim. scores comp1 = {c1["sim_scores"].median():.3f}, '
      f'ARI entre particiones (mediana) = {df_cl_cmp["ARI_prob_vs_logit"].median():.3f}')
print(tabla.to_string())


joblib.dump({'saturacion': df_sat, 'suavizado': df_suav, 'fpca': df_fpca_cmp, 'clustering': df_cl_cmp,
             'punto_partida': cmp_pp, 'tabla_resumen': tabla}, f'{RES}/logit_vs_probabilidad.pkl')
print('Guardado: logit_vs_probabilidad.pkl')
