"""Datos — Adult / Census Income

[Adult / Census Income](https://archive.ics.uci.edu/dataset/2/adult) (OpenML `adult`, v2): 48842 personas del censo de EEUU de 1994; predecir si la renta anual supera 50.000 $ (clase 1, ≈25 %). Mezcla de variables numéricas y categóricas con interacciones entre ellas.

Preprocesado:
- Se eliminan las filas con nulos en `workclass`, `occupation` o `native-country` (7.4 %).
- Se descarta `education`, que es exactamente redundante con `education-num`.
- `native-country` se agrupa en los 10 países más frecuentes + `Other`.
- One-hot (`drop_first`) de las 7 categóricas. Entrada del MLP: 6 numéricas + 45 one-hot = 51 columnas.
- Solo las 6 numéricas pueden tener curva ICE (`variables_ice`). Durante el barrido, las one-hot se mantienen en el valor real de cada persona.
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
d = fetch_openml(name='adult', version=2, as_frame=True, parser='auto')
df = d.data.copy(); objetivo = d.target.astype(str)
nulos = df[['workclass', 'occupation', 'native-country']].isna().any(axis=1)
print(f'Filas con nulos eliminadas: {nulos.sum()} ({100 * nulos.mean():.2f} %)')
df, objetivo = df.loc[~nulos].reset_index(drop=True), objetivo.loc[~nulos].reset_index(drop=True)

numericas = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
categoricas = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
df['native-country'] = df['native-country'].astype(str)
top_paises = df['native-country'].value_counts().nlargest(10).index
df['native-country'] = df['native-country'].where(df['native-country'].isin(top_paises), 'Other')
onehot = pd.get_dummies(df[categoricas].astype(str), drop_first=True).astype(np.float64)

variables = numericas + list(onehot.columns)          # numéricas primero (el scaler solo las toca a ellas)
variables_ice = variables_escaladas = numericas
X = pd.concat([df[numericas].astype(np.float64), onehot], axis=1)[variables].to_numpy()
y = (objetivo == '>50K').astype(int).to_numpy()
clases = ['<=50K', '>50K']
print('Instancias:', len(y), '| entradas:', len(variables), f'({len(numericas)} numéricas + {onehot.shape[1]} one-hot)')


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
