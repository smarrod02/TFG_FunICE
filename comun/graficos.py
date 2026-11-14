"""Figuras reutilizadas por los scripts de todos los conjuntos de datos."""
import numpy as np
import matplotlib.pyplot as plt

# Colores fijos por clase (0, 1) y por escala, iguales en todos los scripts
COLOR_CLASE = ('#2a78d6', '#eb6834')
COLOR_ESCALA = {'probabilidad': '#2a78d6', 'logit': '#eb6834'}
GRIS = '#52514e'


def curvas_ice(ice, variables_top, y, clases, escala='probabilidad', centradas=None, pdp=True, X_eje=None):
    """Una fila de paneles (una por variable) con las curvas ICE coloreadas por clase real.

    `centradas`: dict var -> matriz ya centrada (si se pasa, se dibuja en lugar de la curva cruda).
    `X_eje`: dict var -> valores observados en unidades originales, para marcar dónde hay datos (rug).
    """
    y = np.asarray(y).reshape(-1).astype(int)
    fig, axes = plt.subplots(1, len(variables_top), figsize=(4.6 * len(variables_top), 4), squeeze=False)
    for ax, var in zip(axes[0], variables_top):
        x = ice[var]['eje_x']
        M = centradas[var] if centradas is not None else ice[var][escala]
        for c in (0, 1):
            ax.plot(x, M[y == c].T, color=COLOR_CLASE[c], alpha=0.18, linewidth=0.6)
            ax.plot([], [], color=COLOR_CLASE[c], label=clases[c])
        if pdp:
            ax.plot(x, M.mean(axis=0), color='black', linewidth=2, label='PDP')
        if X_eje is not None:
            lo = ax.get_ylim()[0]
            ax.plot(X_eje[var], np.full(len(X_eje[var]), lo), '|', color=GRIS, alpha=0.5)
        ax.set_title(var)
        ax.set_xlabel(var)
        ax.grid(alpha=0.3)
    etiqueta = {'probabilidad': 'probabilidad predicha', 'logit': 'logit predicho'}[escala]
    axes[0][0].set_ylabel(etiqueta + (' (centrado)' if centradas is not None else ''))
    axes[0][0].legend(fontsize=8)
    fig.suptitle(f'Curvas ICE — escala {escala}' + (' — centradas en el mínimo' if centradas is not None else ''))
    plt.tight_layout()
    return fig


def componentes_libro(fd, res_fpca, var, escala, n_componentes=3, n_puntos=300):
    """Media ± sqrt(lambda_k)·phi_k para cada componente (representación clásica de Ramsay y Silverman)."""
    t0, t1 = fd.domain_range[0]
    t = np.linspace(t0, t1, n_puntos)
    mu = fd.mean()(t)[0, :, 0]
    fig, axes = plt.subplots(1, n_componentes, figsize=(5 * n_componentes, 3.8), sharey=True)
    for k, ax in enumerate(axes):
        phi = res_fpca['fpca'].components_[k](t)[0, :, 0]
        a = np.sqrt(res_fpca['autovalores'][k])
        ax.plot(t, mu, color='black', linewidth=2, label='media')
        ax.plot(t, mu + a * phi, color='#e34948', linewidth=1.5, label='+sqrt(λ)·φ')
        ax.plot(t, mu - a * phi, color='#2a78d6', linewidth=1.5, label='−sqrt(λ)·φ')
        ax.set_title(f'Componente {k + 1} ({res_fpca["varianza_explicada"][k]:.1%})')
        ax.set_xlabel(var)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(f'cambio en {escala}')
    axes[0].legend(fontsize=8)
    fig.suptitle(f'FPCA — {var} — escala {escala}')
    plt.tight_layout()
    return fig


def scores_por_clase(res_fpca_escala, variables_top, y, clases, escala):
    """Dispersión score1 vs score2 por clase, una columna por variable."""
    y = np.asarray(y).reshape(-1).astype(int)
    fig, axes = plt.subplots(1, len(variables_top), figsize=(4.4 * len(variables_top), 3.9))
    for ax, var in zip(axes, variables_top):
        S = res_fpca_escala[var]['scores']
        for c in (0, 1):
            ax.scatter(S[y == c, 0], S[y == c, 1], s=14, alpha=0.6, color=COLOR_CLASE[c], label=clases[c])
        ax.set_title(var)
        ax.set_xlabel('score 1')
        ax.grid(alpha=0.3)
    axes[0].set_ylabel('score 2')
    axes[0].legend(fontsize=8)
    fig.suptitle(f'Scores FPCA por clase — escala {escala}')
    plt.tight_layout()
    return fig


def clusters(curvas_escala, res_cl_escala, variables_top, K, escala):
    """Curvas de cada clúster y su media local."""
    colores = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
    fig, axes = plt.subplots(1, len(variables_top), figsize=(4.6 * len(variables_top), 4))
    for ax, var in zip(axes, variables_top):
        fd = curvas_escala[var]
        t, D = fd.grid_points[0], fd.data_matrix[:, :, 0]
        r = res_cl_escala[var][K]
        for c in range(K):
            ax.plot(t, D[r['labels'] == c].T, color=colores[c], alpha=0.15, linewidth=0.5)
            if r['info_clusters'][c]['mu'] is not None:
                ax.plot(t, r['info_clusters'][c]['mu'], color=colores[c], linewidth=2.5,
                        label=f'clúster {c} (n={r["info_clusters"][c]["n_curvas"]})')
        ax.set_title(f'{var} (ARI clase={r["ARI_vs_clase"]:.2f})')
        ax.set_xlabel(var)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle(f'Clustering funcional (K={K}) — escala {escala}')
    plt.tight_layout()
    return fig


def auc_residual(df_comp, variables_top, escala):
    """AUC bruto y residual (sin z0) de cada score, con IC95% bootstrap del residual."""
    fig, axes = plt.subplots(1, len(variables_top), figsize=(3.8 * len(variables_top), 3.8), sharey=True)
    for ax, var in zip(axes, variables_top):
        d = df_comp[df_comp['variable'] == var]
        k = d['componente'].to_numpy()
        ax.scatter(k - 0.1, d['AUC_bruto'], color=GRIS, marker='o', s=40, label='score bruto')
        ax.errorbar(k + 0.1, d['AUC_residual'],
                    yerr=[d['AUC_residual'] - d['IC_residual_lo'], d['IC_residual_hi'] - d['AUC_residual']],
                    fmt='s', color=COLOR_ESCALA[escala], capsize=4, label='residual (sin z0)')
        ax.axhline(0.5, color=GRIS, linewidth=0.8, linestyle='--')
        ax.set_xticks([1, 2, 3])
        ax.set_xlabel('componente')
        ax.set_title(var)
        ax.set_ylim(0.3, 1.0)
        ax.grid(axis='y', alpha=0.3)
    axes[0].set_ylabel('AUC')
    axes[0].legend(fontsize=8)
    fig.suptitle(f'Señal de clase en la forma tras quitar z0 — escala {escala}')
    plt.tight_layout()
    return fig
