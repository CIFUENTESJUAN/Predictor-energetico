"""
Agente 3 - Predictor Multi-Modelo.

Responsabilidad: entrenar 4 modelos de ML (Ridge, Random Forest, XGBoost, MLP)
sobre cada serie temporal del dataset, generar predicciones hacia el futuro
y calcular metricas de rendimiento sobre el periodo de test (2020-2024).

Estrategia:
- Cada serie se entrena desde su primer anio con dato real distinto de cero.
  Esto es correcto porque series como Biodiesel o Alcohol Carburante no
  existian en 1975; inventar datos para esos anios seria metodologicamente
  incorrecto. El modelo aprende desde cuando realmente existe la fuente.
- Train: primer_anio_valido hasta ANIO_FIN_TRAIN
- Test:  ANIO_INICIO_TEST a ANIO_FIN_TEST (5 anios fuera de muestra)
- Validacion durante entrenamiento: TimeSeriesSplit con GridSearchCV
- Prediccion final: autorregresiva hasta ANIO_HORIZONTE

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

warnings.filterwarnings("ignore")


# =============================================================================
# DETECCION DEL PRIMER ANIO VALIDO DE UNA SERIE
# =============================================================================

def detectar_primer_anio_valido(df: pd.DataFrame, columna: str) -> int:
    """
    Detecta el primer anio en que la serie tiene un dato real distinto de cero
    y distinto de NaN.

    Justificacion: series como Biodiesel, Alcohol Carburante o Autogeneracion
    no existian en 1975. Sus primeros anios contienen ceros que no representan
    datos reales sino ausencia de la fuente energetica. Entrenar desde esos
    ceros distorsionaria el modelo porque aprenderia una tendencia de "aparicion
    desde cero" que no es representativa del comportamiento real de la serie.

    Se exige ademas que haya al menos N_SPLITS_CV + max(LAGS) + max(VENTANAS)
    anios de datos validos antes de ANIO_FIN_TRAIN para que el TimeSeriesSplit
    tenga suficiente historia.

    Returns:
        Primer anio con dato real. Si la serie no tiene suficientes datos,
        devuelve el anio minimo del dataset.
    """
    minimo_anios_necesarios = N_SPLITS_CV + max(LAGS) + max(VENTANAS_MEDIA_MOVIL) + 2

    # Buscar primer valor distinto de cero y no nulo
    serie = df[["anio", columna]].copy()
    serie_valida = serie[serie[columna].notna() & (serie[columna] != 0)]

    if serie_valida.empty:
        return int(df["anio"].min())

    primer_anio = int(serie_valida["anio"].iloc[0])

    # Verificar que hay suficientes datos hasta ANIO_FIN_TRAIN
    n_datos_train = len(serie_valida[serie_valida["anio"] <= ANIO_FIN_TRAIN])
    if n_datos_train < minimo_anios_necesarios:
        # Si no hay suficientes datos, intentar desde el inicio del dataset
        primer_anio = int(df["anio"].min())

    return primer_anio


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def construir_features(serie: pd.Series, anios: pd.Series,
                        anio_min: int, anio_max: int) -> pd.DataFrame:
    """
    Construye la matriz de features temporales para una serie.

    Las features capturan:
    - Tendencia temporal: anio normalizado y su cuadrado (captura tendencias
      lineales y no lineales en el tiempo)
    - Inercia reciente: lags 1, 2, 3 (el valor del anio anterior, hace 2 y 3
      anios) y medias moviles de 3 y 5 anios
    - Velocidad de cambio: diferencia absoluta y porcentual respecto al anio
      anterior (captura la aceleracion o desaceleracion de la serie)

    Args:
        serie:    valores de la serie (puede incluir NaN para anios futuros)
        anios:    anios correspondientes a cada valor
        anio_min: anio minimo para normalizar (del subset de entrenamiento)
        anio_max: anio maximo del horizonte de prediccion

    Returns:
        DataFrame con todas las features. Las primeras filas tendran NaN en
        las columnas de lag porque no tienen historia previa suficiente.
        Esas filas se eliminan antes del entrenamiento.
    """
    df_f = pd.DataFrame({"anio": anios.values})
    df_f["anio_norm"] = (df_f["anio"] - anio_min) / max(anio_max - anio_min, 1)
    df_f["anio_cuadrado"] = df_f["anio_norm"] ** 2

    v = pd.Series(serie.values)

    for lag in LAGS:
        df_f[f"lag_{lag}"] = v.shift(lag).values

    for ventana in VENTANAS_MEDIA_MOVIL:
        df_f[f"media_movil_{ventana}"] = v.shift(1).rolling(window=ventana).mean().values

    df_f["diff_1"] = v.shift(1).diff(1).values

    # NOTA: diff_pct_1 (diferencia porcentual) fue descartada deliberadamente.
    # Algunas series energeticas tienen ceros intermedios (hidroenergia en
    # anios de sequia, series que arrancan tarde), lo que provoca division por
    # cero y NaN que invalidan Ridge y MLP. La diferencia absoluta diff_1
    # captura la misma informacion de velocidad de cambio sin este problema.

    return df_f


# =============================================================================
# CONSTRUCCION DE PIPELINES POR MODELO
# =============================================================================

def construir_pipeline(nombre_modelo: str) -> Pipeline:
    """
    Construye un Pipeline de sklearn: StandardScaler + modelo.

    Ridge y MLP son sensibles a la escala de las features, por lo que el
    escalado es obligatorio para ellos. Para RF y XGBoost no es necesario
    pero se aplica por consistencia en el pipeline.

    No se usa SimpleImputer porque los NaN se eliminan antes de entrenar
    al recortar la serie desde su primer anio con dato real.
    """
    if nombre_modelo == "ridge":
        modelo = Ridge(random_state=RANDOM_STATE)

    elif nombre_modelo == "random_forest":
        modelo = RandomForestRegressor(
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

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
            max_iter=500,
            learning_rate_init=0.01,
            early_stopping=False,
        )

    else:
        raise ValueError(f"Modelo desconocido: {nombre_modelo}")

    return Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", modelo),
    ])


def obtener_grid(nombre_modelo: str) -> dict:
    """Devuelve el grid de hiperparametros para GridSearchCV."""
    return {
        "ridge": GRID_RIDGE,
        "random_forest": GRID_RANDOM_FOREST,
        "xgboost": GRID_XGBOOST,
        "mlp": GRID_MLP,
    }[nombre_modelo]


# =============================================================================
# CALCULO DE METRICAS
# =============================================================================

def calcular_metricas(y_test: np.ndarray, y_pred: np.ndarray,
                       y_train: np.ndarray) -> dict:
    """
    Calcula RMSE, MAE, MAPE y R2 sobre el conjunto de test.

    MAPE requiere manejo especial cuando el test contiene valores cercanos
    a cero (series que recien empezaban en 2020-2024). En ese caso se usa
    solo los valores suficientemente grandes para el calculo, o se recurre
    al MAE relativo a la media del train como alternativa.

    Args:
        y_test:  valores reales del periodo de test
        y_pred:  valores predichos del periodo de test
        y_train: valores del periodo de entrenamiento (para fallback de MAPE)

    Returns:
        dict con rmse, mae, mape, r2
    """
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    # MAPE: usar solo observaciones donde el valor real es mayor al 1%
    # de la media del test para evitar division por valores casi cero.
    umbral = max(np.abs(y_test).mean() * 0.01, 1e-6)
    mask_validos = np.abs(y_test) > umbral

    if mask_validos.sum() >= 2:
        mape = float(
            np.mean(np.abs(
                (y_test[mask_validos] - y_pred[mask_validos]) / y_test[mask_validos]
            )) * 100
        )
    else:
        # Fallback: MAE como porcentaje de la media del train
        media_train = float(np.abs(y_train).mean())
        mape = float(mae / media_train * 100) if media_train > 0 else float("nan")

    return {"rmse": rmse, "mae": mae, "mape": mape, "r2": r2}


# =============================================================================
# ENTRENAMIENTO Y PREDICCION DE UNA SERIE
# =============================================================================

def entrenar_y_predecir_serie(df: pd.DataFrame, columna_serie: str,
                                nombre_modelo: str,
                                verbose: bool = False) -> dict:
    """
    Entrena un modelo sobre una serie y genera predicciones hasta el horizonte.

    Flujo detallado:
    1. Detectar el primer anio con dato real de la serie
    2. Recortar el DataFrame desde ese anio (evita ceros artificiales iniciales)
    3. Construir features temporales con feature engineering
    4. Separar train / test eliminando filas con NaN en features
    5. GridSearchCV + TimeSeriesSplit para encontrar mejores hiperparametros
    6. Predecir el periodo de test y calcular metricas
    7. Prediccion autorregresiva del futuro hasta ANIO_HORIZONTE

    Returns:
        dict con:
            - anio_inicio_serie: primer anio con dato real usado en entrenamiento
            - historico:    {anio: valor_real} periodo de train
            - test_real:    {anio: valor_real} periodo de test
            - test_pred:    {anio: valor_predicho} periodo de test
            - futuro_pred:  {anio: valor_predicho} desde 2025 hasta horizonte
            - metricas:     {rmse, mae, mape, r2}
            - mejores_params: hiperparametros ganadores del GridSearchCV
            - feature_importance: dict de importancias (solo RF y XGBoost)
    """
    # --- Paso 1: detectar inicio real de la serie y recortar ---
    anio_inicio_serie = detectar_primer_anio_valido(df, columna_serie)
    df_serie = df[df["anio"] >= anio_inicio_serie].copy().reset_index(drop=True)

    anio_min_serie = int(df_serie["anio"].min())
    anios_historicos = df_serie["anio"].values

    # --- Paso 2: extender la serie hasta el horizonte con NaN ---
    anios_futuros = np.arange(anios_historicos.max() + 1, ANIO_HORIZONTE + 1)
    anios_todos = np.concatenate([anios_historicos, anios_futuros])

    valores_extendidos = np.concatenate([
        df_serie[columna_serie].values,
        np.full(len(anios_futuros), np.nan)
    ])
    serie_extendida = pd.Series(valores_extendidos)
    anios_serie_ext = pd.Series(anios_todos)

    # --- Paso 3: construir features ---
    features = construir_features(
        serie_extendida, anios_serie_ext,
        anio_min=anio_min_serie,
        anio_max=ANIO_HORIZONTE
    )
    features["target"] = valores_extendidos
    features["anio_original"] = anios_todos

    # Eliminar filas donde los lags aun no tienen suficiente historia
    n_dropear = max(max(LAGS), max(VENTANAS_MEDIA_MOVIL))
    features_validas = features.iloc[n_dropear:].copy()

    # --- Paso 4: separar train y test ---
    mask_train = features_validas["anio_original"] <= ANIO_FIN_TRAIN
    mask_test = (
        (features_validas["anio_original"] >= ANIO_INICIO_TEST) &
        (features_validas["anio_original"] <= ANIO_FIN_TEST)
    )

    feature_cols = [
        c for c in features_validas.columns
        if c not in ("target", "anio_original", "anio")
    ]

    X_train = features_validas.loc[mask_train, feature_cols].values
    y_train = features_validas.loc[mask_train, "target"].values
    X_test  = features_validas.loc[mask_test,  feature_cols].values
    y_test  = features_validas.loc[mask_test,  "target"].values

    # Verificacion: si alguna feature de train aun tiene NaN, hay un problema
    # de datos en la serie que no se resolvio con el recorte.
    if np.isnan(X_train).any():
        nan_cols = [feature_cols[i] for i in range(len(feature_cols))
                    if np.isnan(X_train[:, i]).any()]
        raise ValueError(
            f"Serie '{columna_serie}': NaN en features de train tras recorte "
            f"desde {anio_inicio_serie}. Columnas afectadas: {nan_cols}"
        )

    if len(X_train) < N_SPLITS_CV + 2:
        raise ValueError(
            f"Serie '{columna_serie}': insuficientes datos de train "
            f"({len(X_train)} filas) para {N_SPLITS_CV} folds de CV."
        )

    # --- Paso 5: GridSearchCV con TimeSeriesSplit ---
    pipeline = construir_pipeline(nombre_modelo)
    grid = obtener_grid(nombre_modelo)
    cv = TimeSeriesSplit(n_splits=N_SPLITS_CV)

    grid_search = GridSearchCV(
        pipeline, grid,
        cv=cv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        verbose=0,
    )
    grid_search.fit(X_train, y_train)
    mejor_modelo = grid_search.best_estimator_

    # --- Paso 6: prediccion sobre el test y metricas ---
    y_pred_test = mejor_modelo.predict(X_test)
    metricas = calcular_metricas(y_test, y_pred_test, y_train)

    # --- Paso 7: prediccion autorregresiva del futuro ---
    # Construimos una copia de la serie donde vamos llenando anio a anio
    # con las predicciones, de modo que cada prediccion use como lag
    # la prediccion del anio anterior (cuando ya no hay datos reales).
    serie_completa = serie_extendida.copy()

    # Rellenar el test con valores reales para que los lags del futuro
    # sean lo mas precisos posible
    for i, anio in enumerate(anios_historicos):
        if ANIO_INICIO_TEST <= anio <= ANIO_FIN_TEST:
            val = df_serie.loc[df_serie["anio"] == anio, columna_serie]
            if not val.empty:
                serie_completa.iloc[i] = float(val.values[0])

    inicio_idx = len(anios_historicos)
    for idx in range(inicio_idx, len(anios_todos)):
        features_punto = construir_features(
            serie_completa, anios_serie_ext,
            anio_min=anio_min_serie,
            anio_max=ANIO_HORIZONTE
        )
        x_punto = features_punto.iloc[idx][feature_cols].values.reshape(1, -1)

        if np.isnan(x_punto).any():
            # Caso borde: propagar el ultimo valor conocido
            pred = float(serie_completa.dropna().iloc[-1])
        else:
            pred = float(mejor_modelo.predict(x_punto)[0])

        serie_completa.iloc[idx] = pred

    futuro_pred = {
        int(anios_todos[idx]): float(serie_completa.iloc[idx])
        for idx in range(inicio_idx, len(anios_todos))
    }

    # --- Feature importance (solo RF y XGBoost) ---
    feature_importance = None
    if nombre_modelo in ("random_forest", "xgboost"):
        reg = mejor_modelo.named_steps["regressor"]
        if hasattr(reg, "feature_importances_"):
            feature_importance = dict(
                zip(feature_cols, [float(v) for v in reg.feature_importances_])
            )

    # --- Historico y test reales para el dashboard ---
    historico = {
        int(row["anio"]): float(row[columna_serie])
        for _, row in df_serie.iterrows()
        if row["anio"] <= ANIO_FIN_TRAIN
    }
    test_real = {
        int(row["anio"]): float(row[columna_serie])
        for _, row in df_serie.iterrows()
        if ANIO_INICIO_TEST <= row["anio"] <= ANIO_FIN_TEST
    }
    test_pred = {
        int(features_validas.loc[mask_test, "anio_original"].iloc[i]): float(y_pred_test[i])
        for i in range(len(y_pred_test))
    }

    mejores_params = {
        k.replace("regressor__", ""): v
        for k, v in grid_search.best_params_.items()
    }

    return {
        "anio_inicio_serie": anio_inicio_serie,
        "historico": historico,
        "test_real": test_real,
        "test_pred": test_pred,
        "futuro_pred": futuro_pred,
        "metricas": metricas,
        "mejores_params": mejores_params,
        "feature_importance": feature_importance,
    }


# =============================================================================
# AGENTE PRINCIPAL
# =============================================================================

MODELOS = ["ridge", "random_forest", "xgboost", "mlp"]


def ejecutar(resultado_ingesta: dict, verbose: bool = True) -> dict:
    """
    Punto de entrada del agente. Entrena los 4 modelos sobre todas las series.

    Para cada serie:
    - Detecta el primer anio con dato real y recorta desde ahi
    - Entrena los 4 modelos con GridSearchCV + TimeSeriesSplit
    - Calcula metricas sobre el test (2020-2024)
    - Genera predicciones hasta ANIO_HORIZONTE

    Returns:
        dict con:
            - "predicciones":         {serie: {modelo: resultado}}
            - "ranking_global":       DataFrame con metricas promedio por modelo
            - "ranking_por_serie":    DataFrame con metricas por serie y modelo
            - "mejor_modelo_por_serie": {serie: nombre_modelo_ganador_por_MAPE}
    """
    df = resultado_ingesta["df_clean"]
    meta = resultado_ingesta["metadata"]
    series = meta["todas_las_series"]

    if verbose:
        print(f"[Agente Predictor] Entrenando 4 modelos sobre {len(series)} series...")
        print(f"[Agente Predictor] Train: hasta {ANIO_FIN_TRAIN} | "
              f"Test: {ANIO_INICIO_TEST}-{ANIO_FIN_TEST} | "
              f"Horizonte: {ANIO_HORIZONTE}")
        print(f"[Agente Predictor] Cada serie usa datos desde su primer anio con valor real")

    predicciones = {}
    for i, serie in enumerate(series):
        if verbose:
            anio_inicio = detectar_primer_anio_valido(df, serie)
            print(f"[Agente Predictor] ({i+1}/{len(series)}) {serie} "
                  f"[desde {anio_inicio}]")
        predicciones[serie] = {}
        for modelo in MODELOS:
            try:
                resultado = entrenar_y_predecir_serie(df, serie, modelo)
                predicciones[serie][modelo] = resultado
            except Exception as e:
                if verbose:
                    print(f"    ERROR en {modelo}: {e}")
                predicciones[serie][modelo] = None

    # --- Ranking global ---
    if verbose:
        print("[Agente Predictor] Construyendo ranking global...")

    filas = []
    for serie, mods in predicciones.items():
        for modelo, res in mods.items():
            if res is None:
                continue
            filas.append({
                "serie":  serie,
                "modelo": modelo,
                "rmse":   res["metricas"]["rmse"],
                "mae":    res["metricas"]["mae"],
                "mape":   res["metricas"]["mape"],
                "r2":     res["metricas"]["r2"],
            })

    df_ranking = pd.DataFrame(filas)

    # Ranking global: promedio de metricas por modelo
    # Se usa nanmean para ignorar series donde MAPE fue nan
    ranking_global = (
        df_ranking.groupby("modelo")
        .agg({"rmse": "mean", "mae": "mean",
              "mape": lambda x: float(np.nanmean(x)),
              "r2": "mean"})
        .sort_values("mape")
    )

    # Mejor modelo por serie segun MAPE (ignorando nan)
    mejor_por_serie = {}
    for serie, mods in predicciones.items():
        mejor_mape = float("inf")
        mejor = None
        for modelo, res in mods.items():
            if res is None:
                continue
            mape = res["metricas"]["mape"]
            if not np.isnan(mape) and mape < mejor_mape:
                mejor_mape = mape
                mejor = modelo
        mejor_por_serie[serie] = mejor

    if verbose:
        print("[Agente Predictor] Ranking global de modelos (MAPE promedio):")
        print(ranking_global.round(2).to_string())
        print("[Agente Predictor] Listo")

    return {
        "predicciones": predicciones,
        "ranking_global": ranking_global,
        "ranking_por_serie": df_ranking,
        "mejor_modelo_por_serie": mejor_por_serie,
    }


if __name__ == "__main__":
    from agentes import ingesta
    res_ingesta = ingesta.ejecutar()
    res_pred = ejecutar(res_ingesta)