"""Etapas 2 y 3 de FunICE: curvas ICE de la salida del modelo y su representación funcional.

Solo se usa la salida final de la red. Un único barrido vectorizado por variable da el logit en cada punto de la
malla; la probabilidad es sigma(logit). Ambas escalas se centran en el primer punto de la malla (mínimo de la
variable) y se suavizan con B-splines y penalización L2 de la segunda derivada, eligiendo (n_basis, lambda) por GCV.
"""
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

import skfda
from skfda.representation.basis import BSplineBasis
from skfda.preprocessing.smoothing import BasisSmoother
from skfda.misc.regularization import L2Regularization
from skfda.misc.operators import LinearDifferentialOperator
from skfda.preprocessing.smoothing.validation import SmoothingParameterSearch, LinearSmootherGeneralizedCVScorer

from .modelo import sigmoide

CANDIDATOS_N_BASIS = (10, 15, 20, 25, 30, 35, 40)
CANDIDATOS_LAMBDA = np.logspace(-5, 1, 40)
ORDEN_BSPLINE = 4


def submuestra_estratificada(y, n, semilla=42):
    """Índices (ordenados) de una submuestra estratificada de tamaño n. Si n es None o >= len(y), todos."""
    y = np.asarray(y).reshape(-1)
    if n is None or n >= len(y):
        return np.arange(len(y))
    idx, _ = train_test_split(np.arange(len(y)), train_size=n, stratify=y, random_state=semilla)
    return np.sort(idx)


def malla(x_columna, n_puntos=1000):
    """Malla equiespaciada entre el mínimo y el máximo de la variable."""
    x = np.asarray(x_columna)
    return np.linspace(x.min(), x.max(), n_puntos)


def barrido_ice(modelo, X, j, valores):
    """Logit del modelo para cada instancia de X con la columna j sustituida por cada valor de la malla.

    Vectorizado: se construye un tensor (n * len(valores), p) y se evalúa en una sola llamada. Devuelve (n, len(valores)).
    """
    X_np = X.numpy() if torch.is_tensor(X) else np.asarray(X, dtype=np.float32)
    n, G = X_np.shape[0], len(valores)
    bloque = np.repeat(X_np[:, None, :], G, axis=1)
    bloque[:, :, j] = np.asarray(valores)[None, :]
    modelo.eval()
    with torch.no_grad():
        z = modelo(torch.as_tensor(bloque.reshape(-1, X_np.shape[1]), dtype=torch.float32))
    return z.numpy().reshape(n, G)


def extraer_ice(modelo, X, variables, variables_top, desescalado, n_puntos=1000):
    """Curvas ICE sin centrar en ambas escalas para cada variable de `variables_top`.

    El modelo se evalúa en la malla en unidades estandarizadas; `desescalado[var] = (media, desviacion)` la convierte
    a unidades originales (`eje_x`), que es el dominio sobre el que se definen las curvas funcionales.
    Devuelve dict var -> {'malla_escalada', 'eje_x', 'logit', 'probabilidad'}; las matrices son (n, n_puntos).
    """
    variables = list(variables)
    X_np = X.numpy() if torch.is_tensor(X) else np.asarray(X)
    salida = {}
    for var in variables_top:
        j = variables.index(var)
        m = malla(X_np[:, j], n_puntos)
        z = barrido_ice(modelo, X, j, m)
        media, desviacion = desescalado[var]
        salida[var] = {'malla_escalada': m, 'eje_x': m * desviacion + media, 'logit': z, 'probabilidad': sigmoide(z)}
    return salida


def centrar(curvas):
    """Centra cada curva en su primer punto. Devuelve (curvas_centradas, punto_de_partida)."""
    base = curvas[:, 0].copy()
    return curvas - base[:, None], base


def suavizar_gcv(curvas, eje_x, candidatos_n_basis=CANDIDATOS_N_BASIS, candidatos_lambda=CANDIDATOS_LAMBDA,
                 orden=ORDEN_BSPLINE):
    """Suavizado B-spline con penalización L2 de la 2ª derivada; (n_basis, lambda) elegidos por GCV.

    Devuelve (FDataGrid suavizado evaluado en `eje_x`, dict con hiperparámetros y error máximo frente a la curva cruda).
    Nota: las curvas ICE son la salida DETERMINISTA de la red (no hay ruido), así que el GCV suele preferir el máximo
    número de bases y el mínimo lambda (techo de la rejilla).
    """
    fd = skfda.FDataGrid(data_matrix=curvas, grid_points=eje_x)
    rango = (eje_x.min(), eje_x.max())
    mejor = {'gcv': -np.inf}
    for n_basis in candidatos_n_basis:
        base = BSplineBasis(domain_range=rango, n_basis=n_basis, order=orden)
        suavizador = BasisSmoother(basis=base, regularization=L2Regularization(LinearDifferentialOperator(2)),
                                   return_basis=False)
        busqueda = SmoothingParameterSearch(suavizador, candidatos_lambda,
                                            scoring=LinearSmootherGeneralizedCVScorer())
        busqueda.fit(fd)
        if busqueda.best_score_ > mejor['gcv']:
            mejor = {'gcv': busqueda.best_score_, 'n_basis': n_basis,
                     'lambda': busqueda.best_params_['smoothing_parameter']}

    base = BSplineBasis(domain_range=rango, n_basis=mejor['n_basis'], order=orden)
    suavizador = BasisSmoother(basis=base, smoothing_parameter=mejor['lambda'],
                               regularization=L2Regularization(LinearDifferentialOperator(2)))
    fd_suave = suavizador.fit_transform(fd)
    mejor['error_max'] = float(np.abs(fd_suave.data_matrix[:, :, 0] - curvas).max())
    mejor['techo_rejilla'] = mejor['n_basis'] == max(candidatos_n_basis)
    return fd_suave, mejor


def procesar_escalas(ice, variables_top, verbose=True):
    """Centra y suaviza las curvas de cada variable en las dos escalas.

    Devuelve dict escala -> {'centradas': {var: matriz}, 'suavizadas': {var: FDataGrid},
                              'punto_partida': {var: vector}, 'hiperparametros': DataFrame}.
    """
    resultado = {}
    for escala in ('probabilidad', 'logit'):
        centradas, suavizadas, base, filas = {}, {}, {}, []
        for var in variables_top:
            centradas[var], base[var] = centrar(ice[var][escala])
            suavizadas[var], hp = suavizar_gcv(centradas[var], ice[var]['eje_x'])
            filas.append({'variable': var, **hp})
            if verbose:
                aviso = '  (techo de la rejilla)' if hp['techo_rejilla'] else ''
                print(f"[{escala:12s}] {var}: n_basis={hp['n_basis']}, lambda={hp['lambda']:.1e}, "
                      f"error máx={hp['error_max']:.2e}{aviso}")
        resultado[escala] = {'centradas': centradas, 'suavizadas': suavizadas, 'punto_partida': base,
                             'hiperparametros': pd.DataFrame(filas)}
    return resultado
