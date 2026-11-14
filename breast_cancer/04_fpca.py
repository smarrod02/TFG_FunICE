"""Etapa 4 de FunICE (I): ACPF en las dos escalas.

ACPF (`N_COMPONENTES` componentes) de las curvas ICE centradas y suavizadas, por variable y escala
(`comun/fpca.py`). Cada curva queda resumida en sus puntuaciones xi_1, xi_2, xi_3. Se calcula también la varianza
explicada por cada componente y el AUC simétrico de cada puntuación frente a la clase real.

Lee resultados/curvas_ice.pkl; escribe resultados/fpca.pkl.
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

from comun.fpca import fpca_todas
from comun.punto_partida import auc_simetrico
from comun import graficos

ice = joblib.load(f'{RES}/curvas_ice.pkl')
variables_top, y, clases = ice['variables_top'], ice['y'], ice['clases']
res_fpca = fpca_todas(ice['suavizadas'], variables_top, cfg.N_COMPONENTES)

filas = []
for escala in ('probabilidad', 'logit'):
    for v in variables_top:
        r = res_fpca[escala][v]
        for k in range(cfg.N_COMPONENTES):
            filas.append({'escala': escala, 'variable': v, 'componente': k + 1,
                          'var_explicada': r['varianza_explicada'][k],
                          'AUC_clase': auc_simetrico(y, r['scores'][:, k])})
df_fpca = pd.DataFrame(filas)
print(df_fpca.pivot_table(index='variable', columns=['escala', 'componente'], values='var_explicada',
                          sort=False).round(3).to_string())


# ----------------------------------------------------------------------------------------------------
# 1. Autofunciones (media ± √λₖ·φₖ)
# ----------------------------------------------------------------------------------------------------

for escala in ('probabilidad', 'logit'):
    for v in variables_top:
        graficos.componentes_libro(ice['suavizadas'][escala][v], res_fpca[escala][v], v, escala, cfg.N_COMPONENTES)
        plt.savefig(f"{FIG}/04_componentes_{escala}_{v.replace(' ', '_')}.png", dpi=110, bbox_inches='tight')
        plt.close()


# ----------------------------------------------------------------------------------------------------
# 2. Scores y clase real
# ----------------------------------------------------------------------------------------------------

for escala in ('probabilidad', 'logit'):
    graficos.scores_por_clase(res_fpca[escala], variables_top, y, clases, escala)
    plt.savefig(f'{FIG}/04_scores_{escala}.png', dpi=110, bbox_inches='tight'); plt.close()

print('AUC de cada score frente a la clase real (simétrico, max(AUC, 1-AUC)):')
print(df_fpca.pivot_table(index='variable', columns=['escala', 'componente'], values='AUC_clase',
                          sort=False).round(3).to_string())


# ----------------------------------------------------------------------------------------------------
# El AUC de los scores frente a la clase no indica todavía si la forma de la curva aporta información: puede venir enteramente del punto de partida `z0`, con el que los scores están correlacionados. Eso se estudia en `06_punto_partida.py`.
# ----------------------------------------------------------------------------------------------------

joblib.dump({'res_fpca': res_fpca, 'df_fpca': df_fpca}, f'{RES}/fpca.pkl')
print('Guardado: fpca.pkl')
