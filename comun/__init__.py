"""Implementación de FunICE, común a los cuatro conjuntos de datos.

Módulos:
    modelo        -- Etapa 1: ExplicableMLP, entrenamiento calibrado (validación + L2 + parada temprana),
                     métricas y selección de variables con SHAP
    ice           -- Etapas 2 y 3: curvas ICE (probabilidad y logit), centrado y suavizado B-spline con GCV
    fpca          -- Etapa 4: ACPF por variable y escala
    clustering    -- Etapa 4: k-centros funcionales proyectados (Dabo-Niang et al. 2008; Chiou y Li 2007)
    punto_partida -- diagnóstico de la dependencia del punto de partida (z0) de las curvas ICE
    escalas       -- comparación de las escalas de probabilidad y logit
    graficos      -- figuras reutilizadas por los scripts de cada dataset

Los scripts de cada carpeta de dataset importan este paquete y la configuración propia del dataset con:
    sys.path.insert(0, '..'); import config as cfg; from comun import ...
"""

ESCALAS = ('probabilidad', 'logit')
