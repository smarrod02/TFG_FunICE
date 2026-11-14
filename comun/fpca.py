"""Etapa 4 de FunICE (I): ACPF de las curvas ICE suavizadas, por variable y escala."""
import numpy as np
from skfda.preprocessing.dim_reduction import FPCA


def fpca_variable(fd, n_componentes=3, semilla=0):
    """FPCA sobre un FDataGrid. Se fija la semilla global porque el solver de PCA de skfda es aleatorizado."""
    np.random.seed(semilla)
    fpca = FPCA(n_components=n_componentes)
    scores = fpca.fit_transform(fd)
    return {'fpca': fpca, 'scores': scores, 'varianza_explicada': fpca.explained_variance_ratio_,
            'autovalores': fpca.explained_variance_}


def fpca_todas(curvas_por_escala, variables_top, n_componentes=3):
    """curvas_por_escala: escala -> {var: FDataGrid}. Devuelve escala -> var -> resultado de `fpca_variable`."""
    return {escala: {var: fpca_variable(curvas[var], n_componentes) for var in variables_top}
            for escala, curvas in curvas_por_escala.items()}


def autofunciones(res_fpca, t):
    """Matriz (n_componentes, len(t)) con las autofunciones evaluadas en t."""
    return res_fpca['fpca'].components_(t)[:, :, 0]
