#!/usr/bin/env bash
# Ejecuta en orden los scripts 01-07 de un conjunto de datos (entorno tfg-skfda).
#
#   ./ejecutar.sh breast_cancer          # 01 -> 07
#   ./ejecutar.sh higgs 03               # a partir del script 03
#   ./ejecutar.sh comparativa            # comparativa entre conjuntos y figuras de la memoria
#
# Si se ejecutan varios conjuntos a la vez, conviene limitar los hilos de cada uno:
#     HILOS=2 ./ejecutar.sh higgs &   HILOS=2 ./ejecutar.sh magic_telescope &
set -e
if [[ -n "$HILOS" ]]; then
    export OMP_NUM_THREADS=$HILOS MKL_NUM_THREADS=$HILOS OPENBLAS_NUM_THREADS=$HILOS LOKY_MAX_CPU_COUNT=$HILOS
fi
RAIZ="$(cd "$(dirname "$0")" && pwd)"

if [[ "$1" == "comparativa" ]]; then
    for s in comparativa_datasets figura_fiabilidad figura_auc_residual figura_mecanismo_z0; do
        python "$RAIZ/comparativa/$s.py"
    done
    for s in figura_ice_2x2 figura_componentes_2x3 figura_clusters_1x2 figura_scores_1x2; do
        python "$RAIZ/breast_cancer/$s.py"
    done
    exit 0
fi

desde=${2:-01}
for script in "$RAIZ/$1"/0*.py; do
    nombre=$(basename "$script")
    [[ "${nombre:0:2}" < "$desde" ]] && continue
    inicio=$(date +%s)
    python "$script"
    echo "$1/$nombre  ok  ($(( $(date +%s) - inicio )) s)"
done
