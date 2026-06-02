"""
Agente 3 - Predictor Multi-Modelo.

Responsabilidad: entrenar 4 modelos de ML (Ridge, Random Forest, XGBoost, MLP)
sobre cada serie temporal del dataset, generar predicciones hacia el futuro
y calcular metricas de rendimiento sobre el periodo de test (2020-2024).

Estrategia:
- Train: 1975-2019 (~45 anios)
- Test:  2020-2024 (5 anios fuera de muestra)
- Validacion durante el entrenamiento: TimeSeriesSplit con GridSearchCV
- Prediccion final: extender hasta el ANIO_HORIZONTE (2050) en modo autorregresivo

Entrada: DataFrame limpio y metadata (salida del Agente 1)
Salida:  diccionario con predicciones y metricas por serie y por modelo
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

from config import (
    ANIO_FIN_TRAIN, ANIO_INICIO_TEST, ANIO_FIN_TEST, ANIO_HORIZONTE,
    LAGS, VENTANAS_MEDIA_MOVIL, RANDOM_STATE,
    GRID_RIDGE, GRID_RANDOM_FOREST, GRID_XGBOOST, GRID_MLP, N_SPLITS_CV,
)

# Suprimir warnings de convergencia del MLP cuando los hiperparametros no son optimos
warnings.filterwarnings("ignore")


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def construir_features(serie: pd.Series, anios: pd.Series,
                        anio_min: int, anio_max: int) -> pd.DataFrame:
    """
    Construye la matriz de features temporales para una serie.

    Las features capturan:
    - Tendencia temporal: anio normalizado y su cuadrado
    - Inercia reciente: lags y medias moviles
    - Velocidad de cambio: diferencias absolutas y porcentuales

    Args:
        serie: valores de la serie temporal (incluyendo los anios futuros como NaN)
        anios: valores del anio correspondientes
        anio_min, anio_max: rango para normalizar el anio

    Returns:
        DataFrame con todas las features. Las primeras filas tienen NaN
        porque dependen de lags que aun no existen.
    """
    df_features = pd.DataFrame({"anio": anios.values})
    df_features["anio_norm"] = (df_features["anio"] - anio_min) / (anio_max - anio_min)
    df_features["anio_cuadrado"] = df_features["anio_norm"] ** 2

    valores = pd.Series(serie.values)

    # Lags
    for lag in LAGS:
        df_features[f"lag_{lag}"] = valores.shift(lag).values

    # Medias moviles
    for v in VENTANAS_MEDIA_MOVIL:
        df_features[f"media_movil_{v}"] = valores.shift(1).rolling(window=v).mean().values

    # Diferencias
    df_features["diff_1"] = valores.shift(1).diff(1).values

    # Diferencia porcentual: protegida contra division por cero
    val_anterior = valores.shift(1)
    val_dos_atras = valores.shift(2)
    diff_pct = ((val_anterior - val_dos_atras) / val_dos_atras.replace(0, np.nan))
    df_features["diff_pct_1"] = diff_pct.values

    return df_features


# =============================================================================
# CONSTRUCCION DE PIPELINES POR MODELO
# =============================================================================

def construir_pipeline(nombre_modelo: str) -> Pipeline:
    """
    Construye un Pipeline de sklearn con escalado + modelo.

    Ridge y MLP requieren features escaladas (son sensibles a la escala).
    RF y XGBoost no, pero por consistencia se escala en todos.
    """
    if nombre_modelo == "ridge":
        modelo = Ridge(random_state=RANDOM_STATE)
    elif nombre_modelo == "random_forest":
        modelo = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
    elif nombre_modelo == "xgboost":
        modelo = xgb.XGBRegressor(
            random_state=RANDOM_STATE,
            objective="reg:squarederror",
            verbosity=0,
            n_jobs=-1,
        )
    elif nombre_modelo == "mlp":
        modelo = MLPRegressor(
            random_state=RANDOM_STATE,
            solver="adam",
            max_iter=500,  # menor para acelerar; con 500 ya converge en este dataset
            learning_rate_init=0.01,
            early_stopping=False,
        )
    else:
        raise ValueError(f"Modelo desconocido: {nombre_modelo}")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", modelo),
    ])
    return pipeline


def obtener_grid(nombre_modelo: str) -> dict:
    """Devuelve el grid de hiperparametros para GridSearchCV."""
    grids = {
        "ridge": GRID_RIDGE,
        "random_forest": GRID_RANDOM_FOREST,
        "xgboost": GRID_XGBOOST,
        "mlp": GRID_MLP,
    }
    return grids[nombre_modelo]


# =============================================================================
# ENTRENAMIENTO Y PREDICCION DE UNA SERIE
# =============================================================================

def entrenar_y_predecir_serie(df: pd.DataFrame, columna_serie: str,
                                nombre_modelo: str, verbose: bool = False) -> dict:
    """
    Entrena un modelo sobre una serie y genera predicciones hasta el horizonte.

    Flujo:
    1. Construir features con feature engineering
    2. Separar train (hasta ANIO_FIN_TRAIN) y test (ANIO_INICIO_TEST a ANIO_FIN_TEST)
    3. Buscar mejores hiperparametros con GridSearchCV + TimeSeriesSplit
    4. Entrenar modelo final con los mejores hiperparametros
    5. Predecir el periodo de test
    6. Predecir el futuro (test_fin+1 hasta ANIO_HORIZONTE) en modo autorregresivo

    Returns:
        dict con:
            - historico: dict {anio: valor_real}
            - test_real: dict {anio: valor_real_test}
            - test_pred: dict {anio: valor_predicho_test}
            - futuro_pred: dict {anio: valor_predicho_futuro}
            - metricas: dict con RMSE, MAE, MAPE, R2 sobre el test
            - mejores_params: hiperparametros ganadores del GridSearchCV
            - feature_importance: importance scores (solo RF y XGBoost)
    """
    # Crear DataFrame extendido hasta el anio horizonte con NaN para los anios futuros
    anios_historicos = df["anio"].values
    anios_futuros = np.arange(anios_historicos.max() + 1, ANIO_HORIZONTE + 1)
    anios_todos = np.concatenate([anios_historicos, anios_futuros])

    valores_extendidos = np.concatenate([
        df[columna_serie].values,
        np.full(len(anios_futuros), np.nan)
    ])
    serie_extendida = pd.Series(valores_extendidos)
    anios_serie = pd.Series(anios_todos)

    # Construir features sobre toda la serie extendida
    features = construir_features(serie_extendida, anios_serie,
                                    anio_min=anios_historicos.min(),
                                    anio_max=ANIO_HORIZONTE)
    features["target"] = valores_extendidos
    features["anio_original"] = anios_todos

    # Eliminar filas con NaN en features de entrenamiento
    # (las primeras filas que no tienen suficientes lags)
    n_dropear = max(max(LAGS), max(VENTANAS_MEDIA_MOVIL))
    features_validas = features.iloc[n_dropear:].copy()

    # Separar train, test y futuro
    mask_train = features_validas["anio_original"] <= ANIO_FIN_TRAIN
    mask_test = (features_validas["anio_original"] >= ANIO_INICIO_TEST) & \
                (features_validas["anio_original"] <= ANIO_FIN_TEST)

    feature_cols = [c for c in features_validas.columns
                    if c not in ("target", "anio_original", "anio")]

    X_train = features_validas.loc[mask_train, feature_cols].values
    y_train = features_validas.loc[mask_train, "target"].values
    X_test = features_validas.loc[mask_test, feature_cols].values
    y_test = features_validas.loc[mask_test, "target"].values

    # GridSearchCV con TimeSeriesSplit para respetar el orden temporal
    pipeline = construir_pipeline(nombre_modelo)
    grid = obtener_grid(nombre_modelo)
    cv = TimeSeriesSplit(n_splits=N_SPLITS_CV)

    grid_search = GridSearchCV(
        pipeline, grid, cv=cv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1, verbose=0,
    )
    grid_search.fit(X_train, y_train)
    mejor_modelo = grid_search.best_estimator_

    # Prediccion sobre el test set
    y_pred_test = mejor_modelo.predict(X_test)

    # Metricas sobre el test
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
    mae = float(mean_absolute_error(y_test, y_pred_test))
    # MAPE protegido contra valores cero
    y_test_seguro = np.where(np.abs(y_test) < 1e-9, 1e-9, y_test)
    mape = float(np.mean(np.abs((y_test - y_pred_test) / y_test_seguro)) * 100)
    r2 = float(r2_score(y_test, y_pred_test))

    # Prediccion autorregresiva del futuro:
    # Para predecir el anio T se necesitan lags de T-1, T-2, T-3
    # Cuando T es lejano al test, esos lags ya no existen como valores reales,
    # entonces usamos las predicciones anteriores como pseudo-historico.
    serie_completa = serie_extendida.copy()

    # Llenar el test con valores reales (para no afectar los lags futuros)
    # NOTA: usamos los valores reales del test para construir los lags del futuro,
    # asumiendo que en uso real estos tambien estarian disponibles.
    for i, anio in enumerate(anios_historicos):
        if ANIO_INICIO_TEST <= anio <= ANIO_FIN_TEST:
            serie_completa.iloc[i] = df.loc[df["anio"] == anio, columna_serie].values[0]

    # Empezar prediccion autorregresiva desde ANIO_FIN_TEST + 1
    inicio_idx = len(anios_historicos)  # primer indice futuro
    for idx in range(inicio_idx, len(anios_todos)):
        # Reconstruir features para este anio usando la serie acumulada
        features_punto = construir_features(serie_completa, anios_serie,
                                              anio_min=anios_historicos.min(),
                                              anio_max=ANIO_HORIZONTE)
        x_punto = features_punto.iloc[idx][feature_cols].values.reshape(1, -1)

        if np.isnan(x_punto).any():
            # Si todavia hay NaN (caso borde) usamos la prediccion anterior
            pred = serie_completa.iloc[idx - 1]
        else:
            pred = mejor_modelo.predict(x_punto)[0]

        serie_completa.iloc[idx] = pred

    # Extraer las predicciones futuras
    futuro_pred = {
        int(anios_todos[idx]): float(serie_completa.iloc[idx])
        for idx in range(inicio_idx, len(anios_todos))
    }

    # Feature importance (solo aplica a RF y XGBoost)
    feature_importance = None
    if nombre_modelo in ("random_forest", "xgboost"):
        regressor = mejor_modelo.named_steps["regressor"]
        if hasattr(regressor, "feature_importances_"):
            importances = regressor.feature_importances_
            feature_importance = dict(zip(feature_cols, [float(v) for v in importances]))

    # Historico real (todo el periodo conocido)
    historico = {
        int(df["anio"].iloc[i]): float(df[columna_serie].iloc[i])
        for i in range(len(df))
        if df["anio"].iloc[i] <= ANIO_FIN_TRAIN
    }
    test_real = {
        int(df["anio"].iloc[i]): float(df[columna_serie].iloc[i])
        for i in range(len(df))
        if ANIO_INICIO_TEST <= df["anio"].iloc[i] <= ANIO_FIN_TEST
    }
    test_pred = {
        int(features_validas.loc[mask_test, "anio_original"].iloc[i]): float(y_pred_test[i])
        for i in range(len(y_pred_test))
    }

    return {
        "historico": historico,
        "test_real": test_real,
        "test_pred": test_pred,
        "futuro_pred": futuro_pred,
        "metricas": {
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
            "r2": r2,
        },
        "mejores_params": {k.replace("regressor__", ""): v
                             for k, v in grid_search.best_params_.items()},
        "feature_importance": feature_importance,
    }


# =============================================================================
# AGENTE PRINCIPAL: PROCESA TODAS LAS SERIES Y MODELOS
# =============================================================================

MODELOS = ["ridge", "random_forest", "xgboost", "mlp"]


def _entrenar_modelos_para_serie(args):
    """
    Helper para paralelizacion: entrena los 4 modelos para una serie.
    """
    df, serie = args
    resultados = {}
    for modelo in MODELOS:
        try:
            resultados[modelo] = entrenar_y_predecir_serie(df, serie, modelo, verbose=False)
        except Exception as e:
            resultados[modelo] = None
    return serie, resultados


def ejecutar(resultado_ingesta: dict, verbose: bool = True,
              n_jobs_series: int = 1) -> dict:
    """
    Punto de entrada del agente. Entrena los 4 modelos sobre todas las series.

    Args:
        resultado_ingesta: salida del agente de ingesta
        verbose: imprimir progreso
        n_jobs_series: numero de series a procesar en paralelo
                       (cuidado: cada serie ya usa todos los cores
                       internamente via GridSearchCV n_jobs=-1)

    Returns:
        dict con:
            - "predicciones": {serie: {modelo: resultado_de_serie}}
            - "ranking_global": DataFrame con metricas promedio por modelo
            - "mejor_modelo_por_serie": {serie: nombre_modelo_ganador}
    """
    df = resultado_ingesta["df_clean"]
    meta = resultado_ingesta["metadata"]
    series = meta["todas_las_series"]

    if verbose:
        print(f"[Agente Predictor] Entrenando 4 modelos sobre {len(series)} series...")
        print(f"[Agente Predictor] Train: 1975-{ANIO_FIN_TRAIN} | Test: {ANIO_INICIO_TEST}-{ANIO_FIN_TEST} | Horizonte: {ANIO_HORIZONTE}")

    predicciones = {}
    for i, serie in enumerate(series):
        if verbose:
            print(f"[Agente Predictor] ({i+1}/{len(series)}) {serie}")
        predicciones[serie] = {}
        for modelo in MODELOS:
            try:
                resultado = entrenar_y_predecir_serie(df, serie, modelo, verbose=False)
                predicciones[serie][modelo] = resultado
            except Exception as e:
                if verbose:
                    print(f"    ERROR en {modelo}: {e}")
                predicciones[serie][modelo] = None

    # Construir ranking global
    if verbose:
        print("[Agente Predictor] Construyendo ranking global...")

    filas_ranking = []
    for serie, resultados in predicciones.items():
        for modelo, res in resultados.items():
            if res is None:
                continue
            filas_ranking.append({
                "serie": serie,
                "modelo": modelo,
                "rmse": res["metricas"]["rmse"],
                "mae": res["metricas"]["mae"],
                "mape": res["metricas"]["mape"],
                "r2": res["metricas"]["r2"],
            })

    df_ranking = pd.DataFrame(filas_ranking)
    ranking_global = df_ranking.groupby("modelo").agg({
        "rmse": "mean",
        "mae": "mean",
        "mape": "mean",
        "r2": "mean",
    }).sort_values("mape")

    # Mejor modelo por serie (segun MAPE)
    mejor_modelo_por_serie = {}
    for serie in predicciones:
        mejor_mape = float("inf")
        mejor = None
        for modelo, res in predicciones[serie].items():
            if res is None:
                continue
            if res["metricas"]["mape"] < mejor_mape:
                mejor_mape = res["metricas"]["mape"]
                mejor = modelo
        mejor_modelo_por_serie[serie] = mejor

    if verbose:
        print("[Agente Predictor] Ranking global de modelos (por MAPE promedio):")
        print(ranking_global.round(2).to_string())
        print("[Agente Predictor] Listo")

    return {
        "predicciones": predicciones,
        "ranking_global": ranking_global,
        "ranking_por_serie": df_ranking,
        "mejor_modelo_por_serie": mejor_modelo_por_serie,
    }


if __name__ == "__main__":
    from agentes import ingesta
    res_ingesta = ingesta.ejecutar()
    res_pred = ejecutar(res_ingesta)
