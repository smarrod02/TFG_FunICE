"""Etapa 1 de FunICE: MLP calibrado y selección de variables mediante SHAP.

Protocolo, igual en todos los datasets: validación separada (20 % del entrenamiento, estratificada), Adam con
regularización L2 (`weight_decay`), parada temprana sobre la BCE de validación restaurando los mejores pesos, y
búsqueda de `lr` y `l2` con 3 semillas eligiendo la menor BCE de validación media (regla de puntuación propia:
premia la calibración).
"""
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score, recall_score, f1_score)


class ExplicableMLP(nn.Module):
    """Misma arquitectura en los cuatro datasets: 16 -> 8 -> 1, ReLU, dropout 0.2. Devuelve el LOGIT."""

    def __init__(self, input_dim):
        super(ExplicableMLP, self).__init__()
        self.layer0 = nn.Linear(input_dim, 16)
        self.relu1 = nn.ReLU()
        self.layer1 = nn.Linear(16, 8)
        self.relu2 = nn.ReLU()
        self.layer2 = nn.Linear(8, 1)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.dropout(self.relu1(self.layer0(x)))
        x = self.dropout(self.relu2(self.layer1(x)))
        return self.layer2(x)  # logit (sin sigmoide): la pérdida es BCEWithLogitsLoss


def cargar_modelo(ruta_pesos, input_dim):
    modelo = ExplicableMLP(input_dim)
    modelo.load_state_dict(torch.load(ruta_pesos))
    modelo.eval()
    return modelo


def logits_de(modelo, X):
    modelo.eval()
    with torch.no_grad():
        return modelo(X).numpy().reshape(-1)


def sigmoide(z):
    return 0.5 * (1 + np.tanh(0.5 * z))  # estable numéricamente


def entrenar(lr, l2, semilla, X_tr, y_tr, X_val, y_val, batch_size=64, max_epocas=300, paciencia=25):
    """Adam + L2 + parada temprana sobre la BCE de validación. Devuelve (modelo, historial, mejor_bce, mejor_epoca)."""
    torch.manual_seed(semilla)
    modelo = ExplicableMLP(X_tr.shape[1])
    criterio = nn.BCEWithLogitsLoss()
    optimizador = optim.Adam(modelo.parameters(), lr=lr, weight_decay=l2)
    cargador = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True,
                          generator=torch.Generator().manual_seed(semilla))

    mejor, mejores_pesos, mejor_epoca, sin_mejora = np.inf, None, 0, 0
    historial = []
    for epoca in range(1, max_epocas + 1):
        modelo.train()
        perdida_train = 0.0
        for xb, yb in cargador:
            optimizador.zero_grad()
            perdida = criterio(modelo(xb), yb)
            perdida.backward()
            optimizador.step()
            perdida_train += perdida.item() * len(xb)
        perdida_train /= len(X_tr)

        modelo.eval()
        with torch.no_grad():
            perdida_val = criterio(modelo(X_val), y_val).item()
        historial.append((epoca, perdida_train, perdida_val))

        if perdida_val < mejor - 1e-4:
            mejor, mejores_pesos, mejor_epoca, sin_mejora = perdida_val, copy.deepcopy(modelo.state_dict()), epoca, 0
        else:
            sin_mejora += 1
            if sin_mejora >= paciencia:
                break

    modelo.load_state_dict(mejores_pesos)
    modelo.eval()
    return modelo, pd.DataFrame(historial, columns=['epoca', 'loss_train', 'loss_val']), mejor, mejor_epoca


def buscar_hiperparametros(X_tr, y_tr, X_val, y_val, batch_size=64,
                           lrs=(1e-2, 1e-3), l2s=(0.0, 1e-3, 1e-2, 1e-1), semillas=(0, 1, 2)):
    """Rejilla lr x l2 con varias semillas. Devuelve (df_busqueda, resumen ordenado por BCE de validación media)."""
    filas = []
    for lr in lrs:
        for l2 in l2s:
            for s in semillas:
                _, _, mejor, mejor_epoca = entrenar(lr, l2, s, X_tr, y_tr, X_val, y_val, batch_size=batch_size)
                filas.append({'lr': lr, 'l2': l2, 'semilla': s, 'BCE_val': mejor, 'mejor_epoca': mejor_epoca})
    df = pd.DataFrame(filas)
    resumen = (df.groupby(['lr', 'l2'])
               .agg(BCE_val=('BCE_val', 'mean'), sd=('BCE_val', 'std'), epoca_media=('mejor_epoca', 'mean'))
               .reset_index().sort_values('BCE_val'))
    return df, resumen


def metricas(z, y, n_bins=10):
    """Métricas de clasificación y de calibración, todas calculadas a partir del logit z."""
    y = np.asarray(y).reshape(-1).astype(int)
    p = sigmoide(z)
    pred = (z > 0).astype(int)
    bce = np.mean(np.logaddexp(0, -z) * y + np.logaddexp(0, z) * (1 - y))
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    ece = sum((idx == b).mean() * abs(y[idx == b].mean() - p[idx == b].mean())
              for b in range(n_bins) if (idx == b).any())
    return {'exactitud': accuracy_score(y, pred), 'AUC': roc_auc_score(y, z),
            'precisión': precision_score(y, pred, zero_division=0), 'recall': recall_score(y, pred),
            'F1': f1_score(y, pred), 'BCE': bce, 'Brier': np.mean((p - y) ** 2), 'ECE': ece,
            '|logit| mediana': np.median(np.abs(z)), 'frac no saturada': np.mean((p > 0.01) & (p < 0.99))}


def ranking_shap(modelo, X_fondo, X_explicar, variables):
    """|SHAP| medio por variable (DeepExplainer). Devuelve (Series ordenada, matriz de valores SHAP)."""
    import shap
    modelo.eval()
    explicador = shap.DeepExplainer(modelo, X_fondo)
    valores = np.asarray(explicador.shap_values(X_explicar))
    if valores.ndim == 3:
        valores = valores[:, :, 0]
    importancia = pd.Series(np.abs(valores).mean(axis=0), index=list(variables)).sort_values(ascending=False)
    return importancia, valores
