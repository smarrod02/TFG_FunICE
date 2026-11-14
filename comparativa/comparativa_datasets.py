"""Comparativa entre los cuatro conjuntos de datos.

Se ejecuta después de los scripts 01-07 de cada conjunto. No recalcula nada: solo lee resultados/*.pkl de cada uno.

1. Modelos calibrados (Etapa 1).
2. Dependencia del punto de partida por escala, con el mecanismo que la explica.
3. Probabilidad frente a logit.

Escribe comparativa.pkl, que leen figura_auc_residual.py y figura_mecanismo_z0.py, y las figuras
figuras/punto_partida_datasets.png y figuras/mecanismo_z0.png.
"""
import sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # rutas relativas a la carpeta del script
import matplotlib
matplotlib.use('Agg')  # las figuras se guardan en disco, no se muestran
sys.path.insert(0, '..')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from scipy import stats

pd.set_option('display.width', 200); pd.set_option('display.max_columns', 30)
os.makedirs('figuras', exist_ok=True)

DATASETS = {'Breast Cancer': 'breast_cancer', 'MAGIC': 'magic_telescope', 'Adult': 'adult_income', 'HIGGS': 'higgs'}
COLORES = {'Breast Cancer': '#2a78d6', 'MAGIC': '#eb6834', 'Adult': '#1baf7a', 'HIGGS': '#eda100'}
MARCADOR = {'probabilidad': dict(marker='o'), 'logit': dict(marker='s', facecolor='white')}
GRIS = '#52514e'
plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False})

R = {}
for nombre, carpeta in DATASETS.items():
    base = f'../{carpeta}/resultados'
    R[nombre] = {'mlp': joblib.load(f'{base}/informe_mlp.pkl'), 'pp': joblib.load(f'{base}/punto_partida.pkl'),
                 'esc': joblib.load(f'{base}/logit_vs_probabilidad.pkl'), 'ice': joblib.load(f'{base}/curvas_ice.pkl')}
nombres = list(DATASETS)
print('Cargados:', nombres)


# ----------------------------------------------------------------------------------------------------
# 1. Modelos calibrados
# ----------------------------------------------------------------------------------------------------

filas = []
for n in nombres:
    m = R[n]['mlp']['metricas'].loc[('modelo', 'test')]
    filas.append({'dataset': n, 'n_ICE': len(R[n]['ice']['y']), 'variables_top': ', '.join(R[n]['ice']['variables_top']),
                  'lr': R[n]['mlp']['config']['lr'], 'l2': R[n]['mlp']['config']['l2'],
                  'exactitud': m['exactitud'], 'AUC': m['AUC'], 'BCE': m['BCE'], 'ECE': m['ECE'],
                  '|logit| mediana': m['|logit| mediana']})
df_modelos = pd.DataFrame(filas).set_index('dataset')
print(df_modelos.round(3).to_string())


# ----------------------------------------------------------------------------------------------------
# 2. Dependencia del punto de partida
# ----------------------------------------------------------------------------------------------------

df_var = pd.concat([R[n]['pp']['df_var'][e].assign(dataset=n, escala=e) for n in nombres for e in ('probabilidad', 'logit')])
df_comp = pd.concat([R[n]['pp']['df_comp'][e].assign(dataset=n, escala=e) for n in nombres for e in ('probabilidad', 'logit')])
df_clz0 = pd.concat([R[n]['pp']['df_clusters_z0'][e].assign(dataset=n, escala=e) for n in nombres for e in ('probabilidad', 'logit')])

def resumen(g):
    n, e = g.name
    c = df_comp[(df_comp['dataset'] == n) & (df_comp['escala'] == e)]
    z = df_clz0[(df_clz0['dataset'] == n) & (df_clz0['escala'] == e)]
    return pd.Series({'AUC modelo': round(g['AUC_modelo'].iloc[0], 3),
                      'AUC z0': f"{g['AUC_z0'].min():.3f}-{g['AUC_z0'].max():.3f}",
                      'z0 captura (mediana)': f"{100 * g['fraccion_capturada_z0'].median():.0f}%",
                      'corr(z0, logit real) (mediana)': round(g['corr(z0,logit_real)'].median(), 3),
                      'pares residual signif.': f"{int(c['residual_significativo'].sum())}/{len(c)}",
                      'ΔAUC z0+forma (mediana)': round(g['delta_AUC'].median(), 3),
                      'vars ΔAUC signif.': f"{int((g['p_perm'] < 0.05).sum())}/{len(g)}",
                      'AUC z0→clúster (mediana)': round(z['AUC_z0_predice_cluster'].median(), 3),
                      'ARI clúster-clase (mediana)': round(z['ARI_vs_clase'].median(), 3)})

tabla_pp = df_var.groupby(['dataset', 'escala'], sort=False).apply(resumen, include_groups=False)
print(tabla_pp.to_string())

fig, axes = plt.subplots(1, 3, figsize=(19, 5))
desp = {'probabilidad': -0.14, 'logit': 0.14}

ax = axes[0]
for i, n in enumerate(nombres):
    g = df_var[(df_var['dataset'] == n) & (df_var['escala'] == 'logit')]
    ax.scatter(np.full(len(g), i) + np.linspace(-0.12, 0.12, len(g)), g['AUC_z0'], s=55, color=COLORES[n],
               edgecolor='white', linewidth=1.2, zorder=3)
    ax.hlines(g['AUC_modelo'].iloc[0], i - 0.3, i + 0.3, color=GRIS, linewidth=2)
ax.set_xticks(range(len(nombres))); ax.set_xticklabels(nombres); ax.set_ylim(0.5, 1.0); ax.set_ylabel('AUC')
ax.set_title('(a) AUC de z0 solo (puntos: variables)\nlínea: AUC del propio modelo'); ax.grid(axis='y', alpha=0.3)

for ax, col, titulo, ref in ((axes[1], 'AUC_residual', '(b) AUC del mejor score FPCA tras quitar z0', 0.5),
                             (axes[2], 'delta_AUC', '(c) ΔAUC al añadir la forma (ξ1-ξ3) a z0', 0.0)):
    for i, n in enumerate(nombres):
        for e in ('probabilidad', 'logit'):
            if col == 'AUC_residual':
                d = df_comp[(df_comp['dataset'] == n) & (df_comp['escala'] == e)].groupby('variable')[col].max()
            else:
                d = df_var[(df_var['dataset'] == n) & (df_var['escala'] == e)].set_index('variable')[col]
            kw = dict(MARCADOR[e]); fc = kw.pop('facecolor', COLORES[n])
            ax.scatter(np.full(len(d), i + desp[e]) + np.linspace(-0.06, 0.06, len(d)), d, s=45, facecolor=fc,
                       edgecolor=COLORES[n], linewidth=1.5, zorder=3, **kw)
    ax.axhline(ref, color=GRIS, linewidth=1, linestyle='--')
    ax.set_xticks(range(len(nombres))); ax.set_xticklabels(nombres); ax.set_title(titulo); ax.grid(axis='y', alpha=0.3)
for e in ('probabilidad', 'logit'):
    kw = dict(MARCADOR[e]); fc = kw.pop('facecolor', GRIS)
    axes[2].scatter([], [], s=45, facecolor=fc, edgecolor=GRIS, linewidth=1.5, label=e, **kw)
axes[2].legend(title='escala', fontsize=8)
axes[1].set_ylabel('AUC residual'); axes[2].set_ylabel('ΔAUC (CV)')
plt.tight_layout(); plt.savefig('figuras/punto_partida_datasets.png', dpi=150, bbox_inches='tight'); plt.close()


# ----------------------------------------------------------------------------------------------------
# Mecanismo
#
# `z0` es la predicción del modelo con una sola variable movida a su mínimo. Si la predicción depende poco de esa variable concreta, `z0` casi coincide con la predicción real, y la curva centrada apenas puede añadir información de clase. Esto se contrasta sobre las 20 variables de los cuatro datasets.
# ----------------------------------------------------------------------------------------------------

dl = df_var[df_var['escala'] == 'logit']
r1 = stats.spearmanr(dl['corr(z0,logit_real)'], dl['fraccion_capturada_z0'])
print(f"corr(z0, logit real) vs fracción del AUC del modelo que captura z0: Spearman rho={r1[0]:.2f} (p={r1[1]:.1g})")
for e in ('probabilidad', 'logit'):
    d = df_var[df_var['escala'] == e]
    r2 = stats.spearmanr(d['corr(z0,logit_real)'], d['delta_AUC'])
    print(f"corr(z0, logit real) vs ΔAUC [{e:12s}]: Spearman rho={r2[0]:.2f} (p={r2[1]:.1g})")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
for n in nombres:
    g = dl[dl['dataset'] == n]
    axes[0].scatter(g['corr(z0,logit_real)'], g['fraccion_capturada_z0'], s=55, color=COLORES[n],
                    edgecolor='white', linewidth=1.2, label=n, zorder=3)
    for e in ('probabilidad', 'logit'):
        g = df_var[(df_var['dataset'] == n) & (df_var['escala'] == e)]
        kw = dict(MARCADOR[e]); fc = kw.pop('facecolor', COLORES[n])
        axes[1].scatter(g['corr(z0,logit_real)'], g['delta_AUC'], s=45, facecolor=fc, edgecolor=COLORES[n],
                        linewidth=1.5, zorder=3, **kw)
axes[0].axhline(1, color=GRIS, linestyle='--', linewidth=1)
axes[0].set_xlabel('corr(z0, logit real)'); axes[0].set_ylabel('(AUC z0 − 0.5) / (AUC modelo − 0.5)')
axes[0].set_title('z0 recupera más AUC del modelo cuanto más se parece a la predicción'); axes[0].legend(fontsize=8)
axes[1].axhline(0, color=GRIS, linewidth=1)
axes[1].set_xlabel('corr(z0, logit real)'); axes[1].set_ylabel('ΔAUC (z0 + forma frente a z0)')
axes[1].set_title('...y menos aporta la forma de la curva\n(círculo: probabilidad, cuadrado: logit)')
for ax in axes: ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig('figuras/mecanismo_z0.png', dpi=150, bbox_inches='tight'); plt.close()


# ----------------------------------------------------------------------------------------------------
# 3. Probabilidad frente a logit
# ----------------------------------------------------------------------------------------------------

tabla_escalas = pd.concat({n: R[n]['esc']['tabla_resumen'] for n in nombres}, axis=1)
print(tabla_escalas.to_string())

sat = pd.concat([R[n]['esc']['saturacion'].assign(dataset=n) for n in nombres])
fp = pd.concat([R[n]['esc']['fpca'].assign(dataset=n) for n in nombres])
cl = pd.concat([R[n]['esc']['clustering'].assign(dataset=n) for n in nombres])
print('Saturación (mediana entre variables):')
print(sat.groupby('dataset', sort=False)[['frac_puntos_saturados', 'frac_planas_prob_pero_no_logit',
                                         'rango_max_logit']].median().round(3).to_string())
print('\nConcordancia entre escalas (mediana entre variables), componente 1 y 2:')
print(fp[fp['componente'] <= 2].groupby(['dataset', 'componente'], sort=False)[['sim_autofuncion', 'sim_scores']]
      .median().round(3).to_string())
print('\nClustering: ARI entre la partición en probabilidad y en logit (mediana):')
print(cl.groupby('dataset', sort=False)['ARI_prob_vs_logit'].median().round(3).to_string())


joblib.dump({'modelos': df_modelos, 'punto_partida': tabla_pp, 'df_var': df_var, 'df_comp': df_comp,
             'escalas': tabla_escalas}, 'comparativa.pkl')
print('Guardado: comparativa.pkl, figuras/punto_partida_datasets.png, figuras/mecanismo_z0.png')
