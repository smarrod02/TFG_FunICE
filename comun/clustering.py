"""Etapa 4 de FunICE (II): clustering funcional por k-centros funcionales proyectados.

Dabo-Niang et al. (2008); Chiou y Li (2007). Resumen:
    - Distancia L2: d(f, g)^2 = integral de (f - g)^2.
    - Cada clúster c tiene su media mu_c y su FPCA local {phi_j^(c)}.
    - Reconstrucción de g_i en la base de c: mu_c + sum_{j<=M_c} <g_i - mu_c, phi_j^(c)> phi_j^(c).
    - Asignación: c*(g_i) = argmin_c || g_i - reconstrucción_c(g_i) ||^2.
    - Inicialización: FPCA global + k-means sobre los scores; se itera hasta que no cambian las etiquetas.

Truco de implementación: `FPCA.transform` centra con la media aprendida en `fit`, así que ajustar la FPCA solo con
las curvas del clúster y transformar TODAS las curvas da directamente los scores xi_j^(c)(g_i).
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from skfda.preprocessing.dim_reduction import FPCA


def elegir_m_c(varianza_explicada, umbral=0.90, minimo=1, maximo=None):
    """Mínimo número de componentes cuya varianza explicada acumulada alcanza `umbral`."""
    m_c = int(np.searchsorted(np.cumsum(varianza_explicada), umbral) + 1)
    m_c = max(min(m_c, len(varianza_explicada)), minimo)
    return min(m_c, maximo) if maximo is not None else m_c


def _inicializacion(fd, K, n_componentes_max, umbral, semilla):
    np.random.seed(semilla)
    n_max = min(n_componentes_max, fd.n_samples - 1)
    fpca_global = FPCA(n_components=n_max)
    scores = fpca_global.fit_transform(fd)
    m = elegir_m_c(fpca_global.explained_variance_ratio_, umbral, maximo=n_max)
    return KMeans(n_clusters=K, n_init=10, random_state=semilla).fit_predict(scores[:, :m])


def _iteracion(fd, labels, K, n_componentes_max, umbral):
    grid = fd.grid_points[0]
    datos = fd.data_matrix[:, :, 0]
    distancias2 = np.full((fd.n_samples, K), np.inf)
    info = {}
    for c in range(K):
        fd_c = fd[labels == c]
        if fd_c.n_samples < 2:
            info[c] = {'m_c': 0, 'mu': None, 'n_curvas': fd_c.n_samples}
            continue
        n_max = min(n_componentes_max, fd_c.n_samples - 1)
        fpca_c = FPCA(n_components=n_max).fit(fd_c)
        m_c = elegir_m_c(fpca_c.explained_variance_ratio_, umbral, maximo=n_max)
        scores = fpca_c.transform(fd)[:, :m_c]
        mu_c = fpca_c.mean_.data_matrix[0, :, 0]
        reconstruccion = mu_c[None, :] + scores @ fpca_c.components_.data_matrix[:m_c, :, 0]
        distancias2[:, c] = np.trapezoid((datos - reconstruccion) ** 2, grid, axis=1)
        info[c] = {'m_c': m_c, 'mu': mu_c, 'n_curvas': fd_c.n_samples}
    return np.argmin(distancias2, axis=1), distancias2, info


def k_centros_funcionales(fd, K, umbral_varianza=0.90, n_componentes_max=10, max_iter=20, semilla=42):
    labels = _inicializacion(fd, K, n_componentes_max, umbral_varianza, semilla)
    np.random.seed(semilla)
    convergio = False
    for iteracion in range(1, max_iter + 1):
        nuevas, distancias2, info = _iteracion(fd, labels, K, n_componentes_max, umbral_varianza)
        if np.array_equal(nuevas, labels):
            convergio = True
            break
        labels = nuevas
    labels = nuevas
    return {'labels': labels, 'K': K, 'convergio': convergio, 'n_iteraciones': iteracion,
            'distancias2': distancias2, 'info_clusters': info,
            'error_reconstruccion': float(distancias2[np.arange(len(labels)), labels].sum())}


def clustering_todas(curvas_por_escala, variables_top, y, valores_k=(2, 3, 4), **kwargs):
    """Devuelve escala -> var -> K -> resultado (con ARI frente a la clase real)."""
    resultado = {}
    for escala, curvas in curvas_por_escala.items():
        resultado[escala] = {}
        for var in variables_top:
            resultado[escala][var] = {}
            for K in valores_k:
                r = k_centros_funcionales(curvas[var], K, **kwargs)
                r['ARI_vs_clase'] = adjusted_rand_score(y, r['labels'])
                resultado[escala][var][K] = r
    return resultado
