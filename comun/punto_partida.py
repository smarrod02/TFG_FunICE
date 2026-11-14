"""Análisis de la dependencia del punto de partida (z0) de las curvas ICE.

z0 = logit del modelo con la variable barrida fijada en el primer punto de la malla (su mínimo) y el resto de
variables en su valor observado; p0 = sigma(z0). Como las curvas se centran en ese punto, z0 es justo la
información que el centrado "quita" de la curva. La pregunta es cuánta información de clase queda en la FORMA.

Análisis (los mismos en las dos escalas; el control es siempre z0):
    1. AUC de z0 solo y fracción del AUC del propio modelo que ya recupera: (AUC_z0 - 0.5) / (AUC_modelo - 0.5).
    2. Correlación de z0 con la predicción real del modelo (mecanismo: si la variable barrida pesa poco, z0 ≈ predicción).
    3. Para cada score FPCA: rho de Spearman con z0, AUC bruto y AUC del residuo de `score ~ z0`, con IC95% bootstrap.
    4. Test incremental: regresión logística CV (5x5) de clase ~ z0 frente a clase ~ z0 + scores; p-valor por
       permutación de los scores con los mismos pliegues.
    5. Relación de los clústeres funcionales con z0 y con la clase.
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold


def auc_simetrico(y, x):
    a = roc_auc_score(y, x)
    return max(a, 1 - a)


def residuo_sobre(z0, x):
    reg = LinearRegression().fit(z0.reshape(-1, 1), x)
    return x - reg.predict(z0.reshape(-1, 1))


def remuestreos_estratificados(y, B=1000, semilla=42):
    rng = np.random.default_rng(semilla)
    i0, i1 = np.where(y == 0)[0], np.where(y == 1)[0]
    return [np.concatenate([rng.choice(i0, len(i0)), rng.choice(i1, len(i1))]) for _ in range(B)]


def ic_auc_residual(y, z0, x, remuestreos):
    """IC95% bootstrap del AUC del residuo, con la orientación fijada en la muestra completa (si no, max(auc, 1-auc)
    sesgaría el intervalo hacia arriba cerca de 0.5)."""
    signo = 1 if roc_auc_score(y, residuo_sobre(z0, x)) >= 0.5 else -1
    valores = [roc_auc_score(y[i], signo * residuo_sobre(z0[i], x[i])) for i in remuestreos]
    return np.percentile(valores, [2.5, 97.5])


def _auc_cv(X, y, pliegues):
    """AUC fuera de muestra medio de una regresión logística (con estandarización) sobre pliegues fijos."""
    aucs = []
    for tr, te in pliegues:
        modelo = make_pipeline(StandardScaler(), LogisticRegression()).fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], modelo.decision_function(X[te])))
    return float(np.mean(aucs))


def test_incremental(y, z0, S, n_perm=200, semilla=42, n_jobs=-1):
    """AUC CV de clase ~ z0 (M0), ~ scores (M1), ~ z0 + scores (M2); p-valor de ΔAUC = M2 - M0 por permutación.

    Validación cruzada estratificada 5x5 con los MISMOS pliegues para el ΔAUC observado y para la distribución nula,
    que se obtiene permutando los scores entre instancias (se rompe su relación con la clase y con z0, z0 intacto).
    """
    from joblib import Parallel, delayed
    pliegues = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=semilla).split(S, y))
    Z = z0.reshape(-1, 1)
    m0, m1, m2 = _auc_cv(Z, y, pliegues), _auc_cv(S, y, pliegues), _auc_cv(np.hstack([Z, S]), y, pliegues)
    rng = np.random.default_rng(semilla)
    permutaciones = [rng.permutation(len(S)) for _ in range(n_perm)]
    # 1 hilo BLAS por proceso: los ajustes son diminutos y más hilos solo compiten entre sí
    nulos = np.array(Parallel(n_jobs=n_jobs, inner_max_num_threads=1)(
        delayed(_auc_cv)(np.hstack([Z, S[p]]), y, pliegues) for p in permutaciones)) - m0
    return {'AUC_cv_z0': m0, 'AUC_cv_forma': m1, 'AUC_cv_z0+forma': m2, 'delta_AUC': m2 - m0,
            'delta_nulo_medio': nulos.mean(), 'p_perm': (1 + np.sum(nulos >= m2 - m0)) / (1 + n_perm)}


def diagnostico(y, z0_por_var, logit_real, res_fpca_escala, B=1000, n_perm=200, semilla=42):
    """Diagnóstico completo para UNA escala. Devuelve (df por variable, df por variable y componente)."""
    y = np.asarray(y).reshape(-1).astype(int)
    remuestreos = remuestreos_estratificados(y, B, semilla)
    auc_modelo = roc_auc_score(y, logit_real)
    filas_var, filas_comp = [], []
    for var, z0 in z0_por_var.items():
        S = res_fpca_escala[var]['scores']
        auc_z0 = auc_simetrico(y, z0)
        for k in range(S.shape[1]):
            lo, hi = ic_auc_residual(y, z0, S[:, k], remuestreos)
            filas_comp.append({'variable': var, 'componente': k + 1,
                               'var_explicada': res_fpca_escala[var]['varianza_explicada'][k],
                               'rho(z0,score)': stats.spearmanr(z0, S[:, k])[0],
                               'AUC_bruto': auc_simetrico(y, S[:, k]),
                               'AUC_residual': auc_simetrico(y, residuo_sobre(z0, S[:, k])),
                               'IC_residual_lo': lo, 'IC_residual_hi': hi, 'residual_significativo': lo > 0.5})
        filas_var.append({'variable': var, 'AUC_modelo': auc_modelo, 'AUC_z0': auc_z0,
                          'corr(z0,logit_real)': stats.pearsonr(z0, logit_real)[0],
                          'fraccion_capturada_z0': (auc_z0 - 0.5) / (auc_modelo - 0.5),
                          **test_incremental(y, z0, S, n_perm, semilla)})
    return pd.DataFrame(filas_var), pd.DataFrame(filas_comp)


def clusters_vs_punto_partida(y, z0_por_var, res_clustering_escala, K):
    """Para cada variable: ARI frente a la clase, AUC de z0 como predictor del clúster (K=2) y z0 medio por clúster."""
    filas = []
    for var, z0 in z0_por_var.items():
        r = res_clustering_escala[var][K]
        labels = r['labels']
        fila = {'variable': var, 'K': K, 'ARI_vs_clase': r['ARI_vs_clase'], 'convergio': r['convergio']}
        if K == 2 and len(np.unique(labels)) == 2:
            fila['AUC_z0_predice_cluster'] = auc_simetrico(labels, z0)
        for c in range(K):
            fila[f'n_c{c}'] = int(np.sum(labels == c))
            fila[f'z0_medio_c{c}'] = float(z0[labels == c].mean()) if np.any(labels == c) else np.nan
        filas.append(fila)
    return pd.DataFrame(filas)
