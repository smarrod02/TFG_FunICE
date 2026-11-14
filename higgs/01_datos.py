"""Datos — HIGGS

[HIGGS](https://archive.ics.uci.edu/dataset/280/higgs) (Baldi et al., 2014) en la versión solo numérica del benchmark de Grinsztajn et al. (2022), OpenML `data_id=44129`: 24 variables continuas (momentos, ángulos, energía faltante y 7 masas invariantes; las `b-tag` discretas ya están excluidas). Señal de bosón de Higgs (1) frente a fondo (0), clases balanceadas. Es el más difícil de los cuatro: la señal está en interacciones no lineales y ninguna variable separa sola (|r| ≤ 0.15).

Se usa una submuestra estratificada de 60000 de las 940160 instancias, para que el entrenamiento tarde lo mismo que en Adult.
"""
import sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # rutas relativas a la carpeta del script
import matplotlib
matplotlib.use('Agg')  # las figuras se guardan en disco, no se muestran
sys.path.insert(0, '..')
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import config as cfg
RES = 'resultados'
os.makedirs(RES, exist_ok=True)
FIG = 'figuras'
os.makedirs(FIG, exist_ok=True)
np.random.seed(cfg.SEMILLA); torch.manual_seed(cfg.SEMILLA)


# ----------------------------------------------------------------------------------------------------
# 1. Carga
# ----------------------------------------------------------------------------------------------------

from sklearn.datasets import fetch_openml
d = fetch_openml(data_id=44129, as_frame=True, parser='auto')
completo = d.frame.copy(); completo['target'] = completo['target'].astype(int)
df, _ = train_test_split(completo, train_size=60000, stratify=completo['target'], random_state=cfg.SEMILLA)
df = df.reset_index(drop=True)
y = df.pop('target').to_numpy()
variables = list(df.columns)
clases = ['fondo', 'señal']
variables_ice = variables_escaladas = variables      # todas continuas
X = df[variables].to_numpy(dtype=np.float64)
print(df.shape); print(df.head())


# ----------------------------------------------------------------------------------------------------
# 2. Análisis exploratorio
# ----------------------------------------------------------------------------------------------------

print(df.dtypes.value_counts()); print('Nulos:', int(df.isna().sum().sum()))
print(pd.Series(y).value_counts().rename(index=dict(enumerate(clases))))

corr = df[variables_ice].assign(target=y).corr()
plt.figure(figsize=(12, 10)); sns.heatmap(corr, cmap='coolwarm', center=0); plt.title('Correlación')
plt.savefig(f'{FIG}/01_correlacion.png', dpi=110, bbox_inches='tight'); plt.close()
print(corr['target'].drop('target').sort_values(key=np.abs, ascending=False).round(3).head(12).to_string())

n = len(variables_ice); ncol = 6; nfil = int(np.ceil(n / ncol))
fig, axes = plt.subplots(nfil, ncol, figsize=(4 * ncol, 3 * nfil))
for ax, v in zip(axes.flatten(), variables_ice):
    for c in (0, 1):
        sns.kdeplot(df.loc[y == c, v], ax=ax, fill=True, alpha=0.3, label=clases[c], warn_singular=False)
    ax.set_title(v, fontsize=10); ax.set_xlabel(''); ax.set_ylabel('')
for ax in axes.flatten()[n:]: ax.axis('off')
axes.flatten()[0].legend(fontsize=8); plt.tight_layout()
plt.savefig(f'{FIG}/01_distribuciones.png', dpi=110, bbox_inches='tight'); plt.close()


# ----------------------------------------------------------------------------------------------------
# 3. División 80/20 estratificada y normalización
#
# `StandardScaler` ajustado solo con el entrenamiento, sobre las variables continuas (que van primero en `X`).
# ----------------------------------------------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=cfg.SEMILLA, stratify=y)
scaler = StandardScaler()
X_train_s, X_test_s = X_train.copy(), X_test.copy()
X_train_s[:, :len(variables_escaladas)] = scaler.fit_transform(X_train[:, :len(variables_escaladas)])
X_test_s[:, :len(variables_escaladas)] = scaler.transform(X_test[:, :len(variables_escaladas)])
print('Train:', X_train.shape, ' Test:', X_test.shape, f' % clase 1 test: {100 * y_test.mean():.1f}')


# ----------------------------------------------------------------------------------------------------
# 4. Guardar
# ----------------------------------------------------------------------------------------------------

joblib.dump(scaler, f'{RES}/scaler.pkl')
desescalado = {str(v): (float(scaler.mean_[i]), float(scaler.scale_[i])) for i, v in enumerate(variables_escaladas)}
torch.save({
    'X_train': torch.FloatTensor(X_train_s), 'y_train': torch.FloatTensor(y_train).unsqueeze(1),
    'X_test': torch.FloatTensor(X_test_s), 'y_test': torch.FloatTensor(y_test).unsqueeze(1),
    'variables': [str(v) for v in variables],            # columnas de entrada del MLP, en orden
    'variables_ice': [str(v) for v in variables_ice],    # variables continuas candidatas a curva ICE
    'desescalado': desescalado,              # var -> (media, desviación) para volver a unidades originales
    'clases': list(clases),                  # clases[0] = clase 0, clases[1] = clase 1 (positiva)
}, f'{RES}/datos_procesados.pt')
print('Guardado: scaler.pkl, datos_procesados.pt | entradas:', len(variables), '| candidatas ICE:', len(variables_ice))
