"""Etapa 1 de FunICE: MLP calibrado y selección de variables mediante SHAP.

Mismo protocolo en los cuatro conjuntos de datos:

1. Validación separada: 20 % del entrenamiento, estratificada. El test no se usa ni para entrenar ni para elegir
   hiperparámetros.
2. Regularización L2 (`weight_decay`) y parada temprana sobre la BCE de validación, restaurando los mejores pesos.
3. Búsqueda de `lr` en {1e-2, 1e-3} y `l2` en {0, 1e-3, 1e-2, 1e-1} con 3 semillas; se elige la menor BCE de
   validación media.
4. Evaluación en test: métricas de clasificación (exactitud, AUC, precisión, sensibilidad, F1) y de calibración
   (BCE, Brier, ECE), diagrama de fiabilidad y matriz de confusión.
5. SHAP para elegir las variables sobre las que se construyen las curvas ICE (`variables_top`), entre las variables
   continuas (`variables_ice`).

Arquitectura `ExplicableMLP` (16 -> 8 -> 1, ReLU, dropout 0.2, salida en logit): `comun/modelo.py`.
Lee resultados/datos_procesados.pt; escribe resultados/mlp_calibrado.pth, variables_top.pt e informe_mlp.pkl.
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

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from comun.modelo import entrenar, buscar_hiperparametros, logits_de, metricas, sigmoide, ranking_shap


# ----------------------------------------------------------------------------------------------------
# 1. Datos y validación
# ----------------------------------------------------------------------------------------------------

datos = torch.load(f'{RES}/datos_procesados.pt', weights_only=False)
X_train, y_train, X_test, y_test = datos['X_train'], datos['y_train'], datos['X_test'], datos['y_test']
variables, variables_ice, clases = datos['variables'], datos['variables_ice'], datos['clases']

idx_tr, idx_val = train_test_split(np.arange(len(X_train)), test_size=0.2,
                                   stratify=y_train.numpy().reshape(-1), random_state=cfg.SEMILLA)
X_tr, y_tr, X_val, y_val = X_train[idx_tr], y_train[idx_tr], X_train[idx_val], y_train[idx_val]
for nombre, yy in (('entrenamiento', y_tr), ('validación', y_val), ('test', y_test)):
    print(f'{nombre:14s} n={len(yy):6d}  % {clases[1]} = {100 * yy.mean().item():.1f}')
print('Entradas del MLP:', X_train.shape[1], '| variables candidatas a ICE:', len(variables_ice))


# ----------------------------------------------------------------------------------------------------
# 2. Búsqueda de hiperparámetros y entrenamiento final
# ----------------------------------------------------------------------------------------------------

torch.set_num_threads(cfg.HILOS_TORCH)
df_busqueda, resumen = buscar_hiperparametros(X_tr, y_tr, X_val, y_val, batch_size=cfg.BATCH_SIZE)
print(resumen.round(4).to_string(index=False))
LR, L2 = float(resumen.iloc[0]['lr']), float(resumen.iloc[0]['l2'])
print(f'\nConfiguración elegida: lr={LR}, l2={L2}')

modelo, historial, bce_val, mejor_epoca = entrenar(LR, L2, cfg.SEMILLA, X_tr, y_tr, X_val, y_val,
                                                    batch_size=cfg.BATCH_SIZE)
print(f'Mejor época: {mejor_epoca} | BCE validación: {bce_val:.4f} | épocas entrenadas: {len(historial)}')

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(historial['epoca'], historial['loss_train'], label='entrenamiento')
ax.plot(historial['epoca'], historial['loss_val'], label='validación')
ax.axvline(mejor_epoca, color='gray', linestyle='--', label=f'mejor época ({mejor_epoca})')
ax.set_xlabel('época'); ax.set_ylabel('BCE'); ax.set_title('Curvas de pérdida'); ax.legend(); ax.grid(alpha=0.3)
plt.savefig(f'{FIG}/02_perdida.png', dpi=130, bbox_inches='tight'); plt.close()


# ----------------------------------------------------------------------------------------------------
# 3. Evaluación y calibración
# ----------------------------------------------------------------------------------------------------

z_val, z_test = logits_de(modelo, X_val), logits_de(modelo, X_test)
df_metricas = pd.DataFrame({('modelo', 'validación'): metricas(z_val, y_val.numpy()),
                            ('modelo', 'test'): metricas(z_test, y_test.numpy())}).T
print(df_metricas.round(4).to_string())

y_t = y_test.numpy().reshape(-1)
fig, ax = plt.subplots(1, 3, figsize=(17, 4.3))
ax[0].hist(np.abs(z_test), bins=40, color='#2a78d6', alpha=0.8)
ax[0].set_xlabel('|logit|'); ax[0].set_ylabel('instancias (test)'); ax[0].set_title('Confianza del modelo')

p = sigmoide(z_test); bins = np.linspace(0, 1, 11); idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
ocupados = [b for b in range(10) if (idx == b).any()]
xs = [p[idx == b].mean() for b in ocupados]; ys = [y_t[idx == b].mean() for b in ocupados]
ns = np.array([(idx == b).sum() for b in ocupados])
ax[1].plot(xs, ys, color='#2a78d6', alpha=0.6)
ax[1].scatter(xs, ys, s=20 + 300 * ns / ns.max(), color='#2a78d6', alpha=0.7)
ax[1].plot([0, 1], [0, 1], 'k--', label='calibración perfecta')
ax[1].set_xlabel('probabilidad predicha'); ax[1].set_ylabel(f'fracción real de {clases[1]}')
ax[1].set_title('Diagrama de fiabilidad (test; tamaño ∝ nº instancias)'); ax[1].legend()

for c, color in zip((0, 1), ('#2a78d6', '#eb6834')):
    ax[2].hist(z_test[y_t == c], bins=40, alpha=0.6, color=color, label=clases[c])
ax[2].axvline(0, color='k', linewidth=0.8)
ax[2].set_xlabel('logit'); ax[2].set_title('Logits por clase (test)'); ax[2].legend()
plt.tight_layout(); plt.savefig(f'{FIG}/02_calibracion.png', dpi=130, bbox_inches='tight'); plt.close()

print('Matriz de confusión (test), filas = real, columnas = predicción:')
print(pd.DataFrame(confusion_matrix(y_t, (z_test > 0).astype(int)), index=clases, columns=clases))


# ----------------------------------------------------------------------------------------------------
# 4. SHAP: variables para las curvas ICE
#
# `variables_top` = las `N_VARIABLES_TOP` variables continuas con mayor |SHAP| medio.
# ----------------------------------------------------------------------------------------------------

n_shap = min(1000, len(X_test))
importancia, _ = ranking_shap(modelo, X_train[:100], X_test[:n_shap], variables)
print('Top 15 por |SHAP| medio (todas las entradas):')
print(importancia.head(15).round(4).to_string())

importancia_ice = importancia.loc[[v for v in importancia.index if v in variables_ice]]
variables_top = [str(v) for v in importancia_ice.index[:cfg.N_VARIABLES_TOP]]
print('\nvariables_top (SHAP):', variables_top)


# ----------------------------------------------------------------------------------------------------
# 5. Guardar
# ----------------------------------------------------------------------------------------------------

torch.save(modelo.state_dict(), f'{RES}/mlp_calibrado.pth')
torch.save(variables_top, f'{RES}/variables_top.pt')
joblib.dump({'metricas': df_metricas, 'busqueda': df_busqueda, 'resumen_busqueda': resumen,
             'config': {'lr': LR, 'l2': L2, 'mejor_epoca': mejor_epoca, 'semilla': cfg.SEMILLA,
                        'batch_size': cfg.BATCH_SIZE},
             'historial': historial, 'ranking_shap': importancia,
             'variables_top': variables_top}, f'{RES}/informe_mlp.pkl')
print('Guardado: mlp_calibrado.pth, variables_top.pt, informe_mlp.pkl')
