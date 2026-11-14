"""Etapas 2 y 3 de FunICE: curvas ICE en escala de probabilidad y de logit, y su representación funcional.

Solo se usa la salida del modelo. Para cada variable de `variables_top` y cada instancia de la muestra de test, se
recorre la variable en una malla de `N_PUNTOS_MALLA` puntos entre su mínimo y su máximo, manteniendo las demás
entradas en su valor observado. El barrido está vectorizado: una sola llamada al modelo por variable.

- Logit z(x): salida de la red.
- Probabilidad p(x) = sigma(z(x)): se obtiene del mismo barrido.

Las dos escalas se centran en el primer punto de la malla. El valor que se resta es el punto de partida (z0 en
logit, p0 = sigma(z0) en probabilidad), que se guarda para el análisis de su dependencia. Después se suavizan con
B-splines de orden 4 y penalización L2 de la segunda derivada, eligiendo `n_basis` y lambda por GCV
(`comun/ice.py`). Se guarda también la predicción del modelo en cada instancia (`logit_real`).

Lee resultados/datos_procesados.pt, variables_top.pt y mlp_calibrado.pth; escribe resultados/curvas_ice.pkl.
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

from sklearn.metrics import roc_auc_score
from comun.modelo import cargar_modelo, logits_de
from comun.ice import submuestra_estratificada, extraer_ice, procesar_escalas
from comun import graficos

datos = torch.load(f'{RES}/datos_procesados.pt', weights_only=False)
variables, clases, desescalado = datos['variables'], datos['clases'], datos['desescalado']
variables_top = torch.load(f'{RES}/variables_top.pt')
modelo = cargar_modelo(f'{RES}/mlp_calibrado.pth', datos['X_train'].shape[1])

y_test_full = datos['y_test'].numpy().reshape(-1).astype(int)
idx_sub = submuestra_estratificada(y_test_full, cfg.N_SUB, cfg.SEMILLA)
X, y = datos['X_test'][idx_sub], y_test_full[idx_sub]
logit_real = logits_de(modelo, X)
print(f'Submuestra ICE: {len(idx_sub)} de {len(y_test_full)} instancias de test, % {clases[1]} = {100 * y.mean():.1f}')
print(f'AUC del modelo: submuestra {roc_auc_score(y, logit_real):.3f} | '
      f'test completo {roc_auc_score(y_test_full, logits_de(modelo, datos["X_test"])):.3f}')
print('variables_top:', variables_top)


# ----------------------------------------------------------------------------------------------------
# 1. Barrido ICE
# ----------------------------------------------------------------------------------------------------

ice = extraer_ice(modelo, X, variables, variables_top, desescalado, n_puntos=cfg.N_PUNTOS_MALLA)
X_obs = {v: X[:, variables.index(v)].numpy() * desescalado[v][1] + desescalado[v][0] for v in variables_top}
z0 = {v: ice[v]['logit'][:, 0] for v in variables_top}
for v in variables_top:
    print(f'{v:28s} malla [{ice[v]["eje_x"][0]:.4g}, {ice[v]["eje_x"][-1]:.4g}]   '
          f'z0 medio={z0[v].mean():+.2f}   corr(z0, logit real)={np.corrcoef(z0[v], logit_real)[0, 1]:.3f}')

for escala in ('probabilidad', 'logit'):
    graficos.curvas_ice(ice, variables_top, y, clases, escala=escala, X_eje=X_obs)
    plt.savefig(f'{FIG}/03_ice_{escala}.png', dpi=110, bbox_inches='tight'); plt.show()


# ----------------------------------------------------------------------------------------------------
# Las marcas verticales del eje inferior son los valores observados de la variable: fuera de ellas, la curva es extrapolación del modelo. En escala de probabilidad, las curvas que empiezan cerca de 0 o de 1 quedan aplanadas por la saturación de la sigmoide; en logit no hay saturación, pero los valores pueden crecer mucho en las zonas de extrapolación.
# ----------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------
# 2. Centrado y suavizado B-spline (GCV)
# ----------------------------------------------------------------------------------------------------

procesadas = procesar_escalas(ice, variables_top)

for escala in ('probabilidad', 'logit'):
    suaves = {v: procesadas[escala]['suavizadas'][v].data_matrix[:, :, 0] for v in variables_top}
    graficos.curvas_ice(ice, variables_top, y, clases, escala=escala, centradas=suaves)
    plt.savefig(f'{FIG}/03_ice_{escala}_centradas.png', dpi=110, bbox_inches='tight'); plt.show()
    print(procesadas[escala]['hiperparametros'].round(6).to_string(index=False))


# ----------------------------------------------------------------------------------------------------
# Nota sobre el GCV. Si las variables tocan el techo de la rejilla (`n_basis=40`, lambda mínima) es porque las curvas ICE son la salida determinista de la red, no observaciones con ruido, así que no hay un equilibrio sesgo-varianza que el GCV pueda optimizar. El suavizado actúa entonces como una representación en base B-spline casi interpolante (ver `error_max`).
# ----------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------
# 3. Guardar
# ----------------------------------------------------------------------------------------------------

paquete = {
    'variables_top': variables_top, 'clases': clases, 'idx_sub': idx_sub, 'y': y, 'logit_real': logit_real,
    'z0': z0,                                             # punto de partida en logit; p0 = sigmoide(z0)
    'eje_x': {v: ice[v]['eje_x'] for v in variables_top},
    'ice_crudas': {v: {e: ice[v][e].astype(np.float32) for e in ('probabilidad', 'logit')} for v in variables_top},
    'suavizadas': {e: procesadas[e]['suavizadas'] for e in ('probabilidad', 'logit')},
    'hiperparametros': {e: procesadas[e]['hiperparametros'] for e in ('probabilidad', 'logit')},
    'config_malla': {'n_puntos': cfg.N_PUNTOS_MALLA},
}
joblib.dump(paquete, f'{RES}/curvas_ice.pkl')
print('Guardado: curvas_ice.pkl')
