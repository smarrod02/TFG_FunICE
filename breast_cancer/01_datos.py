"""Datos — Breast Cancer Wisconsin

[Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) (`sklearn.datasets.load_breast_cancer`): 569 tumores, 30 variables continuas extraídas de imágenes de biopsia, diagnóstico benigno/maligno. Es casi linealmente separable.

sklearn codifica 0 = maligno y 1 = benigno; se invierte para que la clase positiva (1) sea maligno, como en el resto del proyecto.
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

import config as cfg # config.py
RES = 'resultados' 
os.makedirs(RES, exist_ok=True) # Crea la carpeta si no existe
FIG = 'figuras'
os.makedirs(FIG, exist_ok=True)
np.random.seed(cfg.SEMILLA); torch.manual_seed(cfg.SEMILLA) # Fija la semilla para la generación de números aleatorios para permitir la reproducibilidad
# En el archivo de configuración 'config.py' está el valor de la variable SEMILLA


# ----------------------------------------------------------------------------------------------------
# 1. Carga
# ----------------------------------------------------------------------------------------------------

from sklearn.datasets import load_breast_cancer
datos = load_breast_cancer()
variables = list(datos.feature_names)
df = pd.DataFrame(datos.data, columns=variables)
y = 1 - datos.target
clases = ['benigno', 'maligno']
variables_ice = variables_escaladas = variables      # todas continuas
X = df[variables].to_numpy(dtype=np.float64)
print(df.shape); print(df.head())


# ----------------------------------------------------------------------------------------------------
# 2. Análisis exploratorio
# ----------------------------------------------------------------------------------------------------

# Tipos de variables
print(df.dtypes.value_counts())

# Valores nulos
print('Nulos:', int(df.isna().sum().sum()))

# Desblanceo de clases
print(pd.Series(y).value_counts().rename(index=dict(enumerate(clases))))

# Crea un dataframe con el array de numpy 'y'y cuenta cuántos ejemplos hay de cada número
# enumerate(clases) -> toma una lista llamada clases y la empareja con números (0: 'Beningno', 1: 'Maligno'
# Conviertes las parejas en un diccionario
# rename -> Conviertes los números en las clases reales en el conteo


# ----------------------------------------------------------------------------------------------------
# Resultado:
# - Todas las variables son continuas
# - No hay valores nulos
# - Hay desbalanceo de clases (63/37)
# ----------------------------------------------------------------------------------------------------

corr = df[variables_ice].assign(target=y).corr()
# Estudiar la correlación también con la variable objetivo
plt.figure(figsize=(12, 10)); sns.heatmap(corr, cmap='coolwarm', center=0); plt.title('Correlación')
plt.savefig(f'{FIG}/01_correlacion.png', dpi=110, bbox_inches='tight'); plt.close()
print(corr['target'].drop('target').sort_values(key=np.abs, ascending=False).round(3).head(12).to_string())


# ----------------------------------------------------------------------------------------------------
# Comparar cómo se distribuye cada variable en función de las dos clases objetivo
#
# Es una herramienta visual excelente para ver si una variable sirve para separar las clases
# ----------------------------------------------------------------------------------------------------

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
# RESULTADO:
# - Hay variables como: texture_error, symmetry_error, mean_fractal_dimension que visualmente no parecen separar las clases
# - Otras variables omo: worst_concave_points o mean_concavity que visualmente separan las clases
# ----------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------
# 3. División 80/20 estratificada y normalización
#
# `StandardScaler` ajustado solo con el entrenamiento, sobre las variables continuas (que van primero en `X`).
# ----------------------------------------------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=cfg.SEMILLA, stratify=y) 
# 20 % test y estratificada para que la proporcion entre clases se mantenga tanto en train como en test
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
# diccionario que guarda para cada variable su media y desviación típica en el conjunto de entrenamiento

torch.save({
    'X_train': torch.FloatTensor(X_train_s), 'y_train': torch.FloatTensor(y_train).unsqueeze(1),
    'X_test': torch.FloatTensor(X_test_s), 'y_test': torch.FloatTensor(y_test).unsqueeze(1),
    'variables': [str(v) for v in variables],            # columnas de entrada del MLP, en orden
    'variables_ice': [str(v) for v in variables_ice],    # variables continuas candidatas a curva ICE
    'desescalado': desescalado,              # var -> (media, desviación) para volver a unidades originales
    'clases': list(clases),                  # clases[0] = clase 0, clases[1] = clase 1 (positiva)
}, f'{RES}/datos_procesados.pt')
print('Guardado: scaler.pkl, datos_procesados.pt | entradas:', len(variables), '| candidatas ICE:', len(variables_ice))
