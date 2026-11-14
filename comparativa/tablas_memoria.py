"""Tablas LaTeX de los resultados de la memoria (Capítulo 6), generadas a partir de los resultados guardados.

Solo lee resultados/*.pkl de cada conjunto de datos; no recalcula nada. Se ejecuta después de los scripts 01-07
de los cuatro conjuntos. Uso, desde la raíz del repositorio:

    python comparativa/tablas_memoria.py              # lee los resultados de este repositorio
    python comparativa/tablas_memoria.py <carpeta>    # lee los resultados de otra copia del proyecto

Genera en comparativa/tablas/ (cada fichero se incluye en la memoria con \\input):
    tabla_metricas.tex              tab:metricasModelos     rendimiento y calibración de los modelos
    tabla_variables_shap.tex        tab:variablesSHAP       variables seleccionadas mediante SHAP
    tabla_varianza_acpf.tex         tab:varianzaACPF        varianza explicada por el ACPF
    tabla_auc_componentes.tex       tab:aucComponentes      AUC simétrico de las puntuaciones frente a la clase
    tabla_auc_punto_partida.tex     tab:aucPuntoPartida     capacidad discriminante del punto de partida
    tabla_auc_residual.tex          tab:aucResidualProb/Logit   AUC bruto, residual e IC bootstrap
    tabla_test_incremental.tex      tab:testIncremental     test incremental

Paquetes LaTeX necesarios: booktabs y, para la tabla de variables SHAP, tabularx.
"""
import math
import os
import sys

import joblib
import torch

RAIZ_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAIZ = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else RAIZ_REPO
SALIDA = os.path.join(RAIZ_REPO, 'comparativa', 'tablas')

DATASETS = {'Breast Cancer': 'breast_cancer', 'MAGIC': 'magic_telescope', 'Adult': 'adult_income', 'HIGGS': 'higgs'}
ESCALAS = ('probabilidad', 'logit')

# Descripción de las variables seleccionadas (tabla de SHAP)
DESCRIPCION = {
    'worst texture': r'Textura (desviación típica de los niveles de gris), peor valor\textsuperscript{$\dagger$}',
    'mean concave points': 'Número de concavidades del contorno, media',
    'worst radius': 'Radio del núcleo, peor valor',
    'worst area': 'Área del núcleo, peor valor',
    'worst symmetry': 'Simetría del núcleo, peor valor',
    'fAlpha': 'Ángulo del eje mayor de la elipse respecto al origen',
    'fWidth': 'Eje menor de la elipse',
    'fLength': 'Eje mayor de la elipse',
    'fSize': 'Logaritmo de la intensidad total de la imagen',
    'fConc': 'Fracción de la intensidad en los dos píxeles más brillantes',
    'age': 'Edad',
    'capital-gain': 'Ganancias de capital',
    'education-num': 'Nivel educativo (numérico)',
    'hours-per-week': 'Horas trabajadas por semana',
    'capital-loss': 'Pérdidas de capital',
    'm_wwbb': 'Masa invariante del sistema $WWbb$',
    'm_bb': r'Masa invariante de los dos \textit{jets} $b$',
    'm_wbb': 'Masa invariante del sistema $Wbb$',
    'm_jjj': r'Masa invariante de tres \textit{jets}',
    'jet_1_pt': r'Momento transverso del primer \textit{jet}',
}


# ----------------------------------------------------------------------------------------------------------------
# Carga de resultados
# ----------------------------------------------------------------------------------------------------------------

def cargar(carpeta):
    res = os.path.join(RAIZ, carpeta, 'resultados')
    pp = joblib.load(os.path.join(res, 'punto_partida.pkl'))
    return {'mlp': joblib.load(os.path.join(res, 'informe_mlp.pkl')),
            'variables_top': [str(v) for v in torch.load(os.path.join(res, 'variables_top.pt'))],
            'fpca': joblib.load(os.path.join(res, 'fpca.pkl')),
            'df_var': pp['df_var'], 'df_comp': pp['df_comp']}


R = {nombre: cargar(carpeta) for nombre, carpeta in DATASETS.items()}


# ----------------------------------------------------------------------------------------------------------------
# Utilidades de formato
# ----------------------------------------------------------------------------------------------------------------

def tt(var):
    """Nombre de variable en \\texttt, escapando los guiones bajos."""
    return r'\texttt{%s}' % var.replace('_', r'\_')


def potencia10(x):
    """Hiperparámetro como potencia de 10 ($10^{-2}$) o $0$."""
    return '$0$' if x == 0 else '$10^{%d}$' % round(math.log10(x))


def negrita(texto, condicion):
    return r'\textbf{%s}' % texto if condicion else texto


def cabecera_conjunto(nombre, n_columnas, primero, nota=''):
    """Fila con el nombre del conjunto de datos, precedida de un pequeño espacio salvo en el primero."""
    filas = [] if primero else [r'\addlinespace[3pt]']
    filas.append(r'\multicolumn{%d}{@{}l}{\textit{%s}%s} \\' % (n_columnas, nombre, nota))
    return filas


def auc_modelo_test(nombre):
    return R[nombre]['mlp']['metricas'].loc[('modelo', 'test')]['AUC']


def guardar(nombre_fichero, comentario, tex):
    cabecera = ('%% %s\n%% Generado por comparativa/tablas_memoria.py a partir de los resultados guardados; '
                'no editar a mano.\n' % comentario)
    with open(os.path.join(SALIDA, nombre_fichero), 'w') as f:
        f.write(cabecera + tex.lstrip('\n'))
    print('Generado:', os.path.join('comparativa', 'tablas', nombre_fichero))


# ----------------------------------------------------------------------------------------------------------------
# 1. Rendimiento y calibración de los modelos
# ----------------------------------------------------------------------------------------------------------------

def tabla_metricas():
    filas = []
    for nombre in DATASETS:
        cfg = R[nombre]['mlp']['config']
        m = R[nombre]['mlp']['metricas'].loc[('modelo', 'test')]
        valores = ' & '.join('%.3f' % m[c] for c in ('exactitud', 'AUC', 'F1', 'BCE', 'Brier', 'ECE'))
        filas.append(r'\textit{%s} & %s & %s & %s \\' % (nombre, potencia10(cfg['lr']), potencia10(cfg['l2']),
                                                         valores))
    return r'''
\begin{table}[htbp]
\centering
\footnotesize
\setlength{\tabcolsep}{3.6pt}
\caption[Rendimiento y calibración de los modelos]{Hiperparámetros seleccionados y
métricas de clasificación y calibración de los cuatro modelos sobre el conjunto de test.
Las flechas indican si un valor mayor ($\uparrow$) o menor ($\downarrow$) es mejor.}
\label{tab:metricasModelos}
\begin{tabular}{@{}l cc ccc ccc@{}}
\toprule
 & \multicolumn{2}{c}{Hiperparámetros}
 & \multicolumn{3}{c}{Clasificación}
 & \multicolumn{3}{c}{Calibración} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-6} \cmidrule(l){7-9}
Conjunto & $\alpha$ & $\lambda_{L^2}$
 & Exact.\,$\uparrow$ & \acs{AUC}\,$\uparrow$ & F1\,$\uparrow$
 & \acs{BCE}\,$\downarrow$ & Brier\,$\downarrow$ & \acs{ECE}\,$\downarrow$ \\
\midrule
''' + '\n'.join(filas) + r'''
\bottomrule
\end{tabular}
\end{table}
'''


# ----------------------------------------------------------------------------------------------------------------
# 2. Variables seleccionadas mediante SHAP
# ----------------------------------------------------------------------------------------------------------------

def tabla_variables_shap():
    filas = []
    for i, nombre in enumerate(DATASETS):
        nota = r'\textsuperscript{$*$}' if nombre == 'Adult' else ''
        filas += cabecera_conjunto(nombre, 4, i == 0, nota)
        ranking = R[nombre]['mlp']['ranking_shap']
        for k, var in enumerate(R[nombre]['variables_top'], start=1):
            filas.append(r'%d & %s & %s & %.3f \\' % (k, tt(var), DESCRIPCION.get(var, ''), ranking[var]))
    return r'''
\begin{table}[htbp]
\centering
\footnotesize
\setlength{\tabcolsep}{4pt}
\caption[Variables seleccionadas mediante SHAP]{Variables seleccionadas mediante \acs{SHAP}
para la construcción de las curvas \acs{ICE}, ordenadas por su importancia global $I_j$
(media del valor absoluto de \acs{SHAP}, en unidades de logit).}
\label{tab:variablesSHAP}
\begin{tabularx}{\linewidth}{@{}c l >{\raggedright\arraybackslash}X r@{}}
\toprule
 & Variable & Descripción & $I_j$ \\
\midrule
''' + '\n'.join(filas) + r'''
\bottomrule
\multicolumn{4}{@{}>{\raggedright\arraybackslash}p{\linewidth}@{}}{\scriptsize
\textsuperscript{$\dagger$}En \textit{Breast Cancer}, cada característica se mide sobre
todos los núcleos de la imagen; la \emph{media} es su promedio y el \emph{peor valor}, la
media de los tres valores más altos.} \\
\multicolumn{4}{@{}>{\raggedright\arraybackslash}p{\linewidth}@{}}{\scriptsize
\textsuperscript{$*$}Solo se consideran las variables continuas. Las variables categóricas
codificadas en \textit{one-hot} quedan excluidas aunque tengan mayor importancia
(p.\,ej., \texttt{marital-status\_Married-civ-spouse}, tercera en la ordenación global).}
\end{tabularx}
\end{table}
'''


# ----------------------------------------------------------------------------------------------------------------
# 3. Varianza explicada por el ACPF
# ----------------------------------------------------------------------------------------------------------------

def tabla_varianza_acpf():
    filas = []
    for i, nombre in enumerate(DATASETS):
        filas += cabecera_conjunto(nombre, 7, i == 0)
        res_fpca = R[nombre]['fpca']['res_fpca']
        for var in R[nombre]['variables_top']:
            celdas = []
            for escala in ESCALAS:
                rho = 100 * res_fpca[escala][var]['varianza_explicada']
                celdas += ['%.1f' % rho[0], '%.1f' % rho[1], '%.1f' % rho[:3].sum()]
            filas.append(r'%s & %s \\' % (tt(var), ' & '.join(celdas)))
    return r'''
\begin{table}[htbp]
\centering
\footnotesize
\setlength{\tabcolsep}{5pt}
\caption[Varianza explicada por el ACPF]{Porcentaje de varianza explicada por las
componentes principales funcionales de las curvas \acs{ICE} centradas, en cada escala:
primera ($\rho_1$) y segunda ($\rho_2$) componente, y varianza acumulada por las $M=3$
componentes retenidas ($\rho_1+\rho_2+\rho_3$).}
\label{tab:varianzaACPF}
\begin{tabular}{@{}l ccc ccc@{}}
\toprule
 & \multicolumn{3}{c}{Probabilidad} & \multicolumn{3}{c}{Logit} \\
\cmidrule(lr){2-4} \cmidrule(l){5-7}
Variable & $\rho_1$ & $\rho_2$ & $\sum_{k\le 3}\rho_k$ & $\rho_1$ & $\rho_2$ & $\sum_{k\le 3}\rho_k$ \\
\midrule
''' + '\n'.join(filas) + r'''
\bottomrule
\end{tabular}
\end{table}
'''


# ----------------------------------------------------------------------------------------------------------------
# 4. AUC simétrico de las puntuaciones frente a la clase
# ----------------------------------------------------------------------------------------------------------------

def tabla_auc_componentes():
    filas = []
    for i, nombre in enumerate(DATASETS):
        nota = r' \quad {\scriptsize(\acs{AUC} del modelo: %.3f)}' % auc_modelo_test(nombre)
        filas += cabecera_conjunto(nombre, 7, i == 0, nota)
        df = R[nombre]['fpca']['df_fpca']
        for var in R[nombre]['variables_top']:
            celdas = []
            for escala in ESCALAS:
                auc = df[(df['escala'] == escala) & (df['variable'] == var)].sort_values('componente')['AUC_clase']
                auc = auc.to_numpy()
                celdas += [negrita('%.2f' % a, a == auc.max()) for a in auc]
            filas.append(r'%s & %s \\' % (tt(var), ' & '.join(celdas)))
    return r'''
\begin{table}[htbp]
\centering
\footnotesize
\setlength{\tabcolsep}{5pt}
\caption[Capacidad discriminante de las componentes]{\acs{AUC} simétrico,
$\max(\mathrm{AUC},\, 1-\mathrm{AUC})$, de los \textit{scores} de cada componente principal
funcional frente a la clase real. Un valor de 0.5 indica que la componente no separa las
clases y un valor de 1, que las separa perfectamente. En negrita, la componente con mayor
\acs{AUC} de cada variable y escala.}
\label{tab:aucComponentes}
\begin{tabular}{@{}l ccc ccc@{}}
\toprule
 & \multicolumn{3}{c}{Probabilidad} & \multicolumn{3}{c}{Logit} \\
\cmidrule(lr){2-4} \cmidrule(l){5-7}
Variable & $\xi_1$ & $\xi_2$ & $\xi_3$ & $\xi_1$ & $\xi_2$ & $\xi_3$ \\
\midrule
''' + '\n'.join(filas) + r'''
\bottomrule
\end{tabular}
\end{table}
'''


# ----------------------------------------------------------------------------------------------------------------
# 5. Capacidad discriminante del punto de partida
# ----------------------------------------------------------------------------------------------------------------

def tabla_auc_punto_partida():
    filas = []
    for i, nombre in enumerate(DATASETS):
        d = R[nombre]['df_var']['logit']                 # z0 es el mismo en las dos escalas
        nota = r' \quad {\scriptsize(\acs{AUC} del modelo: %.3f)}' % d['AUC_modelo'].iloc[0]
        filas += cabecera_conjunto(nombre, 4, i == 0, nota)
        for _, r in d.iterrows():
            filas.append(r'%s & %.3f & %.2f & %.2f \\' % (tt(r['variable']), r['AUC_z0'], r['fraccion_capturada_z0'],
                                                          r['corr(z0,logit_real)']))
    return r'''
\begin{table}[htbp]
\centering
\footnotesize
\caption[Capacidad discriminante del punto de partida]{Capacidad discriminante del punto
de partida $z_{0i}$ de cada variable: \acs{AUC} de $z_{0i}$ frente a la clase real,
fracción recuperada de la capacidad discriminante del modelo,
$(\mathrm{AUC}_{z_0}-0.5)/(\mathrm{AUC}_{\text{modelo}}-0.5)$, y correlación de Pearson
entre $z_{0i}$ y el logit $\hat\eta_i$ predicho por el modelo. El \acs{AUC} del modelo se
calcula sobre la misma muestra de instancias empleada para construir las curvas \acs{ICE}.}
\label{tab:aucPuntoPartida}
\begin{tabular}{@{}l ccc@{}}
\toprule
Variable & $\mathrm{AUC}_{z_0}$ & Fracción recuperada & $\mathrm{corr}(z_0, \hat\eta)$ \\
\midrule
''' + '\n'.join(filas) + r'''
\bottomrule
\end{tabular}
\end{table}
'''


# ----------------------------------------------------------------------------------------------------------------
# 6. AUC bruto, AUC residual e intervalo bootstrap (dos primeras componentes, una tabla por escala)
# ----------------------------------------------------------------------------------------------------------------

def tabla_auc_residual_escala(escala, etiqueta):
    filas = []
    for i, nombre in enumerate(DATASETS):
        filas += cabecera_conjunto(nombre, 7, i == 0)
        d = R[nombre]['df_comp'][escala]
        d = d[d['componente'] <= 2]
        for var, dv in d.groupby('variable', sort=False):
            celdas = []
            for _, r in dv.sort_values('componente').iterrows():
                celdas += ['%.2f' % r['AUC_bruto'], negrita('%.2f' % r['AUC_residual'], r['residual_significativo']),
                           r'{\scriptsize[%.2f, %.2f]}' % (r['IC_residual_lo'], r['IC_residual_hi'])]
            filas.append(r'%s & %s \\' % (tt(var), ' & '.join(celdas)))
    return r'''
\begin{table}[htbp]
\centering
\footnotesize
\setlength{\tabcolsep}{4pt}
\caption[\acs{AUC} bruto y residual de las componentes (%s)]{\acs{AUC} simétrico de las
puntuaciones de las dos primeras componentes frente a la clase real (B) y de su residuo tras
descontar la dependencia lineal con el punto de partida (R), junto con el intervalo de confianza
\textit{bootstrap} al 95\,\%% de este último, en escala de %s. En negrita, los residuos cuyo
intervalo queda por encima de 0.5.}
\label{%s}
\begin{tabular}{@{}l ccc ccc@{}}
\toprule
& \multicolumn{3}{c}{Componente 1} & \multicolumn{3}{c}{Componente 2} \\
\cmidrule(lr){2-4} \cmidrule(l){5-7}
Variable & B & R & IC 95\,\%% & B & R & IC 95\,\%% \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
''' % (escala, escala, etiqueta, '\n'.join(filas))


def tabla_auc_residual():
    return (tabla_auc_residual_escala('probabilidad', 'tab:aucResidualProb') + '\n'
            + tabla_auc_residual_escala('logit', 'tab:aucResidualLogit'))


# ----------------------------------------------------------------------------------------------------------------
# 7. Test incremental
# ----------------------------------------------------------------------------------------------------------------

def tabla_test_incremental():
    def celdas(r):
        significativo = r['p_perm'] < 0.05
        delta = (r'$\mathbf{%+.3f}$' if significativo else '$%+.3f$') % r['delta_AUC']
        return ['%.3f' % r['AUC_cv_z0+forma'], delta, negrita('%.3f' % r['p_perm'], significativo)]

    filas = []
    for i, nombre in enumerate(DATASETS):
        filas += cabecera_conjunto(nombre, 8, i == 0)
        dp = R[nombre]['df_var']['probabilidad'].set_index('variable')
        dl = R[nombre]['df_var']['logit'].set_index('variable')
        for var in dp.index:
            fila = [tt(var), '%.3f' % dp.loc[var, 'AUC_cv_z0']] + celdas(dp.loc[var]) + celdas(dl.loc[var])
            filas.append(' & '.join(fila) + r' \\')
    return r'''
\begin{table}[htbp]
\centering
\footnotesize
\setlength{\tabcolsep}{3pt}
\caption[Aporte incremental de las componentes]{\acs{AUC} por validación cruzada
estratificada ($5\times5$) de la regresión logística que utiliza únicamente el punto de
partida (M0) y de la que incorpora además las puntuaciones de las tres componentes (M2),
incremento $\Delta\mathrm{AUC} = \mathrm{AUC}_{M2} - \mathrm{AUC}_{M0}$ y p-valor del
contraste de permutación (200 permutaciones; valor mínimo alcanzable, 0.005). El modelo M0
es común a ambas escalas. En negrita, los incrementos significativos ($p < 0.05$).}
\label{tab:testIncremental}
\begin{tabular}{@{}l c ccc ccc@{}}
\toprule
& & \multicolumn{3}{c}{Probabilidad} & \multicolumn{3}{c}{Logit} \\
\cmidrule(lr){3-5} \cmidrule(l){6-8}
Variable & $\mathrm{AUC}_{M0}$ & $\mathrm{AUC}_{M2}$ & $\Delta\mathrm{AUC}$ & $p$
& $\mathrm{AUC}_{M2}$ & $\Delta\mathrm{AUC}$ & $p$ \\
\midrule
''' + '\n'.join(filas) + r'''
\bottomrule
\end{tabular}
\end{table}
'''


# ----------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    os.makedirs(SALIDA, exist_ok=True)
    print('Resultados leídos de:', RAIZ)
    guardar('tabla_metricas.tex', 'Rendimiento y calibración de los modelos (conjunto de test).', tabla_metricas())
    guardar('tabla_variables_shap.tex', 'Variables seleccionadas mediante SHAP.', tabla_variables_shap())
    guardar('tabla_varianza_acpf.tex', 'Varianza explicada por el ACPF (%).', tabla_varianza_acpf())
    guardar('tabla_auc_componentes.tex', 'AUC simétrico de las puntuaciones frente a la clase.',
            tabla_auc_componentes())
    guardar('tabla_auc_punto_partida.tex', 'Capacidad discriminante del punto de partida.', tabla_auc_punto_partida())
    guardar('tabla_auc_residual.tex', 'AUC bruto, AUC residual e IC bootstrap (dos primeras componentes).',
            tabla_auc_residual())
    guardar('tabla_test_incremental.tex', 'Test incremental.', tabla_test_incremental())
