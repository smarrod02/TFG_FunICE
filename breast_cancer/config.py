"""Parámetros de Breast Cancer Wisconsin para los scripts 01-07. Todo lo específico del dataset está aquí y en 01_datos.py."""
NOMBRE = 'Breast Cancer Wisconsin'
SEMILLA = 42
HILOS_TORCH = 2                 # hilos de PyTorch en el entrenamiento (subir si se ejecuta un solo dataset)

# Etapa 1: MLP calibrado y selección de variables (SHAP)
BATCH_SIZE = 32
N_VARIABLES_TOP = 5             # variables con curva ICE: las de mayor |SHAP| entre las continuas

# Etapas 2 y 3: curvas ICE y representación funcional
N_SUB = None                # instancias de test para las curvas ICE (submuestra estratificada; None = todo el test)
N_PUNTOS_MALLA = 1000

# Etapa 4: ACPF y clustering funcional
N_COMPONENTES = 3
VALORES_K = (2, 3, 4)
K_ELEGIDO = 2

# Análisis de la dependencia del punto de partida
N_BOOTSTRAP = 1000
N_PERMUTACIONES = 200
