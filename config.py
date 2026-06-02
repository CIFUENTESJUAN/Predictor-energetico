"""
Configuracion global del proyecto SAPCER v2.

Este modulo centraliza todos los parametros del sistema para que sean
facilmente ajustables sin modificar el codigo de los agentes.
"""

from pathlib import Path

# =============================================================================
# RUTAS
# =============================================================================

DIR_BASE = Path(__file__).parent
DIR_DATA = DIR_BASE / "data"
DIR_OUTPUTS = DIR_BASE / "outputs"
DIR_PREDICCIONES = DIR_OUTPUTS / "predicciones"
DIR_NOTICIAS = DIR_OUTPUTS / "noticias"
DIR_MODELOS = DIR_OUTPUTS / "modelos"

# Crear directorios si no existen
for d in [DIR_OUTPUTS, DIR_PREDICCIONES, DIR_NOTICIAS, DIR_MODELOS]:
    d.mkdir(parents=True, exist_ok=True)

ARCHIVO_DATASET = DIR_DATA / "Historico_energetico_completo_COMPLETO.xlsx"

# =============================================================================
# REPRODUCIBILIDAD
# =============================================================================

# Semilla fija para que los resultados sean reproducibles entre ejecuciones.
# Practica estandar en ML academico.
RANDOM_STATE = 42

# =============================================================================
# DIVISION TEMPORAL TRAIN / TEST
# =============================================================================

# El test set son los ultimos 5 anios reales (2020-2024), simulando que en
# 2019 hubieramos entrenado un modelo y queremos ver que tan bien predice
# los anios siguientes. Esta es la prueba mas honesta de la calidad del modelo.
ANIO_FIN_TRAIN = 2019    # Ultimo anio del conjunto de entrenamiento
ANIO_INICIO_TEST = 2020  # Primer anio del conjunto de prueba
ANIO_FIN_TEST = 2024     # Ultimo anio del conjunto de prueba

# =============================================================================
# HORIZONTE DE PREDICCION
# =============================================================================

# Anio limite para las predicciones futuras (despues del test set)
ANIO_HORIZONTE = 2050

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

# Lags y medias moviles para construir features temporales.
# Estos valores se eligen para balancear captura de patron y perdida de filas.
LAGS = [1, 2, 3]                # Valores anteriores a usar como feature
VENTANAS_MEDIA_MOVIL = [3, 5]   # Anios para promediar
N_FEATURES_TIEMPO = 10          # Total de features generadas

# =============================================================================
# HIPERPARAMETROS PARA GRIDSEARCH
# =============================================================================

# Cada modelo tiene su grid de busqueda. La busqueda se hace con
# TimeSeriesSplit para respetar el orden temporal de los datos.
#
# Los grids estan calibrados para balancear rigor academico con tiempo
# de ejecucion razonable (~5-10 minutos para las 28 series completas).
# Si quieres una busqueda mas exhaustiva, amplia estos grids.

GRID_RIDGE = {
    "regressor__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]
}

GRID_RANDOM_FOREST = {
    "regressor__n_estimators": [100, 200],
    "regressor__max_depth": [3, 5, None],
    "regressor__min_samples_leaf": [2, 5]
}

GRID_XGBOOST = {
    "regressor__n_estimators": [100, 200],
    "regressor__max_depth": [2, 3, 4],
    "regressor__learning_rate": [0.05, 0.1],
    "regressor__reg_lambda": [0.1, 1.0]
}

GRID_MLP = {
    "regressor__hidden_layer_sizes": [(16,), (32, 16)],
    "regressor__alpha": [0.01, 0.1],
}

# Numero de folds para TimeSeriesSplit en el GridSearch.
# Con ~45 puntos de entrenamiento, 5 folds dan ventanas razonables.
N_SPLITS_CV = 5

# =============================================================================
# AGENTE DE CONTEXTO
# =============================================================================

# Queries que el agente de contexto usa para buscar noticias relevantes.
QUERIES_NOTICIAS = [
    "consumo energetico Colombia 2024",
    "transicion energetica Colombia",
    "precio petroleo Colombia 2024",
    "energia renovable Colombia inversion",
    "gas natural Colombia regulacion",
    "crisis energetica Colombia",
    "politica energetica Colombia ministerio minas",
    "fenomeno El Nino energia hidroelectrica Colombia",
]

# Maximo de noticias a procesar
MAX_NOTICIAS = 25

# Antiguedad maxima en dias para considerar una noticia "reciente"
DIAS_ANTIGUEDAD_MAX = 180

# Categorias y keywords asociados para la clasificacion automatica
CATEGORIAS_NOTICIAS = {
    "Politicas energeticas": {
        "keywords": ["regulacion", "ley", "decreto", "ministerio", "creg",
                     "subsidio", "reforma", "politica", "gobierno"],
        "label": "POL"
    },
    "Crisis / Escasez": {
        "keywords": ["crisis", "apagon", "racionamiento", "escasez",
                     "emergencia", "sequia", "deficit", "alerta"],
        "label": "CRI"
    },
    "Economia y Precios": {
        "keywords": ["precio", "inversion", "pib", "exportacion", "opep",
                     "dolar", "inflacion", "mercado", "tarifa"],
        "label": "ECO"
    },
    "Transicion Verde": {
        "keywords": ["renovable", "solar", "eolica", "hidrogeno",
                     "descarbonizacion", "cop", "limpia", "verde"],
        "label": "VER"
    },
    "Infraestructura": {
        "keywords": ["planta", "proyecto", "construccion", "linea",
                     "transmision", "expansion", "obra", "infraestructura"],
        "label": "INF"
    },
}

# Palabras para analisis de sentimiento simple
PALABRAS_POSITIVAS = [
    "crecimiento", "inversion", "mejora", "record", "avance", "expansion",
    "exito", "aumenta", "positivo", "innovacion", "desarrollo", "logro",
    "acuerdo", "fortalece", "impulsa", "beneficia", "supera"
]

PALABRAS_NEGATIVAS = [
    "crisis", "caida", "riesgo", "recorte", "problema", "deficit", "apagon",
    "escasez", "alerta", "preocupacion", "conflicto", "pierde", "reduce",
    "afecta", "negativo", "deteriora", "fracaso", "emergencia"
]

# =============================================================================
# DASHBOARD STREAMLIT
# =============================================================================

# Paleta de colores profesional tipo McKinsey
COLOR_PRIMARIO = "#003A70"      # Azul corporativo oscuro
COLOR_SECUNDARIO = "#00A0DC"    # Azul claro
COLOR_ACENTO = "#FFA500"        # Naranja para destacar
COLOR_EXITO = "#00875A"         # Verde
COLOR_ALERTA = "#DE350B"        # Rojo
COLOR_NEUTRO = "#5E6C84"        # Gris

# Colores para los 4 modelos en las graficas
COLOR_RIDGE = "#003A70"
COLOR_RANDOM_FOREST = "#00875A"
COLOR_XGBOOST = "#DE350B"
COLOR_MLP = "#7B61FF"
COLOR_REAL = "#172B4D"

COLORES_MODELOS = {
    "ridge": COLOR_RIDGE,
    "random_forest": COLOR_RANDOM_FOREST,
    "xgboost": COLOR_XGBOOST,
    "mlp": COLOR_MLP,
}

NOMBRES_MODELOS = {
    "ridge": "Ridge Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "mlp": "MLP (Red Neuronal)",
}

# =============================================================================
# CONFIGURACION DE LOGS
# =============================================================================

VERBOSE = True  # Imprimir progreso en consola
