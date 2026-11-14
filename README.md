<<<<<<< HEAD
# FunICE: análisis funcional de curvas ICE

Código del Trabajo Fin de Grado *Explicabilidad de MLP mediante Análisis de Datos Funcionales* (Sara Martín
Rodríguez, Doble Grado en Ingeniería Informática y Matemáticas, Universidad de Granada).

FunICE explica un modelo de clasificación binaria de caja negra tratando sus curvas ICE como datos funcionales:
las centra, las representa mediante B-splines penalizados y extrae sus principales patrones mediante el análisis de
componentes principales funcional (ACPF) y el clustering funcional. Se aplica a un perceptrón multicapa calibrado
sobre cuatro conjuntos de datos, y se analiza la dependencia de los patrones respecto del punto de partida de las
curvas y el efecto de la escala (probabilidad o logit).

## Estructura

```
FunICE/
├── comun/                       Implementación de FunICE, común a los cuatro conjuntos de datos
│   ├── modelo.py                Etapa 1: ExplicableMLP, entrenamiento calibrado, métricas, SHAP
│   ├── ice.py                   Etapas 2 y 3: curvas ICE, centrado y suavizado B-spline con GCV
│   ├── fpca.py                  Etapa 4: ACPF
│   ├── clustering.py            Etapa 4: k-centros funcionales proyectados
│   ├── punto_partida.py         Dependencia del punto de partida: AUC de z0, AUC residual, test incremental
│   ├── escalas.py               Comparación de las escalas de probabilidad y logit
│   └── graficos.py              Figuras comunes
├── breast_cancer/               Breast Cancer Wisconsin
├── magic_telescope/             MAGIC Gamma Telescope
├── adult_income/                Adult
├── higgs/                       HIGGS
│   ├── config.py                Parámetros del conjunto de datos
│   ├── 01_datos.py              Carga, análisis exploratorio, partición y estandarización (específico)
│   ├── 02_mlp_calibrado.py      Etapa 1: entrenamiento, evaluación, calibración y selección de variables
│   ├── 03_curvas_ice.py         Etapas 2 y 3: curvas ICE y representación funcional
│   ├── 04_fpca.py               Etapa 4: ACPF
│   ├── 05_clustering.py         Etapa 4: clustering funcional
│   ├── 06_punto_partida.py      Dependencia del punto de partida
│   ├── 07_logit_vs_probabilidad.py   Comparación de escalas
│   └── figuras/
├── comparativa/                 Resultados conjuntos de los cuatro conjuntos de datos
│   ├── comparativa_datasets.py  Tablas comparativas; genera comparativa.pkl
│   ├── figura_*.py              Figuras de la memoria (fiabilidad, AUC residual, mecanismo de z0)
│   ├── tablas_memoria.py        Tablas LaTeX de resultados de la memoria
│   ├── figuras/
│   └── tablas/
├── ejecutar.sh
└── environment.yml
```

Los scripts `02`-`07` son idénticos en las cuatro carpetas: lo específico de cada conjunto de datos está en
`config.py` y `01_datos.py`. Los scripts `figura_*.py` de `breast_cancer/` generan las figuras del ejemplo de
`worst texture` que aparecen en la memoria.

## Instalación

```bash
conda env create -f environment.yml
conda activate tfg-skfda
```

Los conjuntos de datos se descargan automáticamente al ejecutar `01_datos.py` (Breast Cancer, desde
`scikit-learn`; MAGIC, Adult y HIGGS, desde OpenML).

## Ejecución

```bash
./ejecutar.sh breast_cancer      # scripts 01 -> 07 de un conjunto de datos
./ejecutar.sh higgs 03           # a partir del script 03
./ejecutar.sh comparativa        # comparativa y figuras de la memoria (tras ejecutar los cuatro conjuntos)
```

Las tablas de resultados de la memoria se generan, tras ejecutar los cuatro conjuntos de datos, con:

```bash
python comparativa/tablas_memoria.py
```

Cada script se puede ejecutar también por separado (`python higgs/04_fpca.py`), siempre que se hayan ejecutado
antes los anteriores. Cada uno guarda sus resultados en `resultados/` y el siguiente los lee, sin repetir los
cálculos. Los resultados no se incluyen en el repositorio: se regeneran con la ejecución completa.

Tiempos aproximados en CPU (Intel Core i7-1165G7): Breast Cancer, unos 5 minutos en total. En MAGIC, Adult y
HIGGS, el script `02` (búsqueda de hiperparámetros, 24 entrenamientos) tarda entre 10 y 20 minutos, y el resto
entre 1 y 6 minutos cada uno. Si se ejecuta un único conjunto, puede aumentarse `HILOS_TORCH` en `config.py`.

## Ficheros de resultados (`resultados/`)

| Fichero | Lo escribe | Contenido |
|---|---|---|
| `datos_procesados.pt`, `scaler.pkl` | 01 | Particiones estandarizadas, nombres de variables, variables continuas, media y desviación de cada variable, clases |
| `mlp_calibrado.pth`, `variables_top.pt`, `informe_mlp.pkl` | 02 | Pesos del modelo, variables seleccionadas, métricas, búsqueda de hiperparámetros, importancia SHAP |
| `curvas_ice.pkl` | 03 | Muestra de test, punto de partida `z0`, logit del modelo, curvas ICE en ambas escalas, curvas suavizadas e hiperparámetros del suavizado |
| `fpca.pkl` | 04 | Puntuaciones, varianza explicada y autofunciones por variable y escala |
| `clustering.pkl` | 05 | Particiones, medias locales, error de reconstrucción y ARI para cada número de clústeres |
| `punto_partida.pkl` | 06 | AUC de `z0`, AUC residual con intervalos bootstrap, test incremental y relación de los clústeres con `z0` |
| `logit_vs_probabilidad.pkl` | 07 | Saturación, suavizado, ACPF y clustering comparados entre escalas |

## Reproducibilidad

Todos los procesos aleatorios usan semillas fijas (definidas en `config.py` y en `comun/`), y las versiones de las
librerías están fijadas en `environment.yml`, por lo que la ejecución completa reproduce los resultados de la
memoria.
=======
# TFG_FunICE
Functional data analysis (ICE curves, FPCA, functional clustering) to explain a black-box MLP — Bachelor’s thesis.
>>>>>>> fa674d4cea23186d9a779200f337213093e2160d
