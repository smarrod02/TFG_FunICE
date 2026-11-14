"""Comparación de las escalas de probabilidad y logit (Sección de comparación de escalas de la memoria).

Saturación de la sigmoide, parecido de las componentes del ACPF entre escalas y coincidencia de las particiones
del clustering funcional.
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import adjusted_rand_score

from .fpca import autofunciones
from .punto_partida import auc_simetrico


def saturacion(ice, variables_top, umbral=0.01, rango_plano_prob=0.02, rango_vivo_logit=1.0):
    """Cuánto 'esconde' la sigmoide: fracción de puntos saturados y de curvas planas en probabilidad pero no en logit."""
    filas = []
    for var in variables_top:
        p, z = ice[var]['probabilidad'], ice[var]['logit']
        rango_p = p.max(axis=1) - p.min(axis=1)
        rango_z = z.max(axis=1) - z.min(axis=1)
        filas.append({'variable': var,
                      'frac_puntos_saturados': np.mean((p < umbral) | (p > 1 - umbral)),
                      'frac_curvas_planas_en_prob': np.mean(rango_p < rango_plano_prob),
                      'frac_planas_prob_pero_no_logit': np.mean((rango_p < rango_plano_prob) & (rango_z > rango_vivo_logit)),
                      'rango_mediano_prob': np.median(rango_p), 'rango_mediano_logit': np.median(rango_z),
                      'rango_max_prob': rango_p.max(), 'rango_max_logit': rango_z.max()})
    return pd.DataFrame(filas)


def comparar_fpca(res_fpca, variables_top, y, t_por_var):
    """Por variable y componente: varianza explicada, AUC de clase y similitud entre escalas.

    - sim_autofuncion: |correlación| entre la autofunción k en probabilidad y en logit (forma, sin importar el signo).
    - sim_scores: |Spearman| entre los scores k de ambas escalas (¿ordenan igual a las instancias?).
    """
    filas = []
    for var in variables_top:
        rp, rl = res_fpca['probabilidad'][var], res_fpca['logit'][var]
        fp, fl = autofunciones(rp, t_por_var[var]), autofunciones(rl, t_por_var[var])
        for k in range(rp['scores'].shape[1]):
            filas.append({'variable': var, 'componente': k + 1,
                          'var_expl_prob': rp['varianza_explicada'][k], 'var_expl_logit': rl['varianza_explicada'][k],
                          'AUC_prob': auc_simetrico(y, rp['scores'][:, k]),
                          'AUC_logit': auc_simetrico(y, rl['scores'][:, k]),
                          'sim_autofuncion': abs(np.corrcoef(fp[k], fl[k])[0, 1]),
                          'sim_scores': abs(stats.spearmanr(rp['scores'][:, k], rl['scores'][:, k])[0])})
    return pd.DataFrame(filas)


def comparar_clustering(res_cl, variables_top, K):
    """ARI entre la partición en probabilidad y en logit (¿agrupan igual?) y ARI de cada una frente a la clase."""
    return pd.DataFrame([{'variable': var, 'K': K,
                          'ARI_prob_vs_logit': adjusted_rand_score(res_cl['probabilidad'][var][K]['labels'],
                                                                   res_cl['logit'][var][K]['labels']),
                          'ARI_clase_prob': res_cl['probabilidad'][var][K]['ARI_vs_clase'],
                          'ARI_clase_logit': res_cl['logit'][var][K]['ARI_vs_clase']}
                         for var in variables_top])
