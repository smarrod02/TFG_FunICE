"""Análisis de la dependencia del punto de partida de las curvas ICE.

z0 es el logit del modelo con la variable barrida fijada en su mínimo (primer punto de la malla) y las demás
entradas en su valor observado. Es el valor que el centrado resta a cada curva. Pregunta: ¿cuánta información sobre
la clase aporta la forma de la curva ICE, más allá de su punto de partida? Métodos en `comun/punto_partida.py`; se
aplican igual a las dos escalas.

| Medida                          | Qué indica                                                                     |
|---------------------------------|--------------------------------------------------------------------------------|
| AUC_z0                          | capacidad de z0 para separar las clases                                        |
| fraccion_capturada_z0           | (AUC_z0 - 0.5) / (AUC_modelo - 0.5): fracción recuperada                        |
| corr(z0, logit_real)            | parecido entre el punto de partida y la predicción del modelo                  |
| AUC_residual (IC 95 % bootstrap)| AUC de cada puntuación tras descontar la dependencia lineal de z0             |
| delta_AUC, p_perm               | test incremental: regresión logística CV 5x5 y p-valor por permutación        |
| AUC_z0_predice_cluster          | si es alto, el clustering funcional equivale a un corte por z0                 |

Lee resultados/curvas_ice.pkl, fpca.pkl y clustering.pkl; escribe resultados/punto_partida.pkl.
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

from comun.punto_partida import diagnostico, clusters_vs_punto_partida
from comun import graficos

ice = joblib.load(f'{RES}/curvas_ice.pkl')
res_fpca = joblib.load(f'{RES}/fpca.pkl')['res_fpca']
res_cl = joblib.load(f'{RES}/clustering.pkl')['res_clustering']
variables_top, y, clases, z0, logit_real = ice['variables_top'], ice['y'], ice['clases'], ice['z0'], ice['logit_real']


# ----------------------------------------------------------------------------------------------------
# 1. Punto de partida frente a la clase y frente a la predicción del modelo
# ----------------------------------------------------------------------------------------------------

fig, axes = plt.subplots(2, len(variables_top), figsize=(4.2 * len(variables_top), 7))
for i, v in enumerate(variables_top):
    for c in (0, 1):
        axes[0][i].hist(z0[v][y == c], bins=20, alpha=0.6, color=graficos.COLOR_CLASE[c], label=clases[c])
        axes[1][i].scatter(z0[v][y == c], logit_real[y == c], s=10, alpha=0.6, color=graficos.COLOR_CLASE[c])
    axes[0][i].set_title(v); axes[0][i].set_xlabel('z0')
    axes[1][i].set_xlabel('z0'); axes[1][i].grid(alpha=0.3)
    axes[1][i].set_title(f'corr = {np.corrcoef(z0[v], logit_real)[0, 1]:.2f}', fontsize=10)
axes[0][0].legend(fontsize=8); axes[0][0].set_ylabel('instancias'); axes[1][0].set_ylabel('logit real del modelo')
plt.tight_layout(); plt.savefig(f'{FIG}/06_z0.png', dpi=110, bbox_inches='tight'); plt.close()


# ----------------------------------------------------------------------------------------------------
# 2. Diagnóstico por escala
# ----------------------------------------------------------------------------------------------------

df_var, df_comp = {}, {}
for escala in ('probabilidad', 'logit'):
    df_var[escala], df_comp[escala] = diagnostico(y, z0, logit_real, res_fpca[escala],
                                                  B=cfg.N_BOOTSTRAP, n_perm=cfg.N_PERMUTACIONES, semilla=cfg.SEMILLA)
    print(f'================ {escala} ================')
    print(df_var[escala].round(3).to_string(index=False)); print()
    print(df_comp[escala].round(3).to_string(index=False)); print()

for escala in ('probabilidad', 'logit'):
    graficos.auc_residual(df_comp[escala], variables_top, escala)
    plt.savefig(f'{FIG}/06_auc_residual_{escala}.png', dpi=110, bbox_inches='tight'); plt.close()


# ----------------------------------------------------------------------------------------------------
# 3. Clústeres funcionales frente a `z0`
# ----------------------------------------------------------------------------------------------------

df_cl_z0 = {e: clusters_vs_punto_partida(y, z0, res_cl[e], cfg.K_ELEGIDO) for e in ('probabilidad', 'logit')}
for e, d in df_cl_z0.items():
    print(f'================ {e} ================'); print(d.round(3).to_string(index=False)); print()


# ----------------------------------------------------------------------------------------------------
# 4. Resumen
# ----------------------------------------------------------------------------------------------------

filas = []
for e in ('probabilidad', 'logit'):
    dv, dc, dz = df_var[e], df_comp[e], df_cl_z0[e]
    filas.append({'escala': e, 'AUC_modelo': dv['AUC_modelo'].iloc[0],
                  'AUC_z0 (mín-máx)': f"{dv['AUC_z0'].min():.3f}-{dv['AUC_z0'].max():.3f}",
                  'z0 captura (mediana)': f"{100 * dv['fraccion_capturada_z0'].median():.0f}%",
                  'pares (var,comp) residual signif.': f"{int(dc['residual_significativo'].sum())}/{len(dc)}",
                  'ΔAUC z0+forma (mediana)': round(dv['delta_AUC'].median(), 3),
                  'vars ΔAUC signif. (p<0.05)': f"{int((dv['p_perm'] < 0.05).sum())}/{len(dv)}",
                  'AUC z0→clúster (mediana)': round(dz['AUC_z0_predice_cluster'].median(), 3)
                                              if 'AUC_z0_predice_cluster' in dz else np.nan,
                  'ARI clúster-clase (mediana)': round(dz['ARI_vs_clase'].median(), 3)})
df_resumen = pd.DataFrame(filas).set_index('escala')
print(df_resumen.to_string())

joblib.dump({'df_var': df_var, 'df_comp': df_comp, 'df_clusters_z0': df_cl_z0, 'resumen': df_resumen},
            f'{RES}/punto_partida.pkl')
print('Guardado: punto_partida.pkl')
