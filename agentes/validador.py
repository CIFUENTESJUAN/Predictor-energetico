"""
Agente 5 - Validador y Comparador.

Responsabilidad: recibir los resultados del Agente Predictor y construir
visualizaciones comparativas + un ranking consolidado de modelos.

Entrada: resultado del Agente Predictor (predicciones y metricas)
Salida:  diccionario con figuras Plotly y resumenes textuales
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import (
    COLORES_MODELOS, NOMBRES_MODELOS, COLOR_REAL,
    ANIO_INICIO_TEST, ANIO_FIN_TEST,
)


# =============================================================================
# GRAFICOS COMPARATIVOS
# =============================================================================

def grafico_predicciones_vs_real(predicciones_serie: dict, columna_serie: str) -> go.Figure:
    """
    Genera un grafico que muestra:
    - Linea negra: serie historica completa (1975 hasta el ultimo anio de train)
    - Puntos negros con marcador especial: valores reales del test (2020-2024)
    - 4 lineas de colores: predicciones de los 4 modelos durante test + futuro

    Permite visualmente comparar cual modelo se ajusta mejor al test real
    y como difieren las proyecciones a futuro.
    """
    fig = go.Figure()

    # Tomar el primer modelo valido para extraer el historico
    primer_modelo_valido = next(
        (m for m in predicciones_serie.values() if m is not None),
        None
    )
    if primer_modelo_valido is None:
        return fig

    historico = primer_modelo_valido["historico"]
    test_real = primer_modelo_valido["test_real"]

    # Serie historica de entrenamiento
    anios_train = sorted(historico.keys())
    valores_train = [historico[a] for a in anios_train]
    fig.add_trace(go.Scatter(
        x=anios_train, y=valores_train,
        mode="lines",
        name="Historico real (train)",
        line=dict(color=COLOR_REAL, width=2.5),
    ))

    # Test real (puntos destacados)
    anios_test = sorted(test_real.keys())
    valores_test = [test_real[a] for a in anios_test]
    fig.add_trace(go.Scatter(
        x=anios_test, y=valores_test,
        mode="lines+markers",
        name=f"Real test {ANIO_INICIO_TEST}-{ANIO_FIN_TEST}",
        line=dict(color=COLOR_REAL, width=2.5, dash="solid"),
        marker=dict(size=10, symbol="circle", color=COLOR_REAL,
                     line=dict(width=2, color="white")),
    ))

    # Predicciones de cada modelo (test + futuro)
    for nombre_modelo, resultado in predicciones_serie.items():
        if resultado is None:
            continue

        color = COLORES_MODELOS[nombre_modelo]
        nombre_display = NOMBRES_MODELOS[nombre_modelo]

        # Concatenar test_pred + futuro_pred para una linea continua
        test_pred = resultado["test_pred"]
        futuro = resultado["futuro_pred"]

        anios_pred = sorted(test_pred.keys()) + sorted(futuro.keys())
        valores_pred = [test_pred[a] for a in sorted(test_pred.keys())] + \
                        [futuro[a] for a in sorted(futuro.keys())]

        fig.add_trace(go.Scatter(
            x=anios_pred, y=valores_pred,
            mode="lines",
            name=nombre_display,
            line=dict(color=color, width=2, dash="dash"),
        ))

    # Banda vertical destacando el periodo de test
    fig.add_vrect(
        x0=ANIO_INICIO_TEST - 0.5, x1=ANIO_FIN_TEST + 0.5,
        fillcolor="lightgray", opacity=0.2,
        layer="below", line_width=0,
        annotation_text="Test", annotation_position="top left",
    )

    fig.update_layout(
        title=f"Predicciones vs Real: {columna_serie}",
        xaxis_title="Anio",
        yaxis_title="Terajoules (TJ)",
        template="plotly_white",
        height=500,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                     xanchor="right", x=1),
    )
    return fig


def grafico_barras_metricas(ranking_serie: pd.DataFrame, metrica: str = "mape") -> go.Figure:
    """
    Grafico de barras comparando una metrica entre los 4 modelos para
    una serie especifica.
    """
    fig = go.Figure()

    df_orden = ranking_serie.sort_values(metrica, ascending=(metrica != "r2"))

    fig.add_trace(go.Bar(
        x=[NOMBRES_MODELOS[m] for m in df_orden["modelo"]],
        y=df_orden[metrica],
        marker_color=[COLORES_MODELOS[m] for m in df_orden["modelo"]],
        text=df_orden[metrica].round(3),
        textposition="outside",
    ))

    nombre_metrica = {
        "rmse": "RMSE (TJ)",
        "mae": "MAE (TJ)",
        "mape": "MAPE (%)",
        "r2": "R^2",
    }.get(metrica, metrica)

    fig.update_layout(
        title=f"Comparativa de {nombre_metrica}",
        xaxis_title="Modelo",
        yaxis_title=nombre_metrica,
        template="plotly_white",
        height=400,
        showlegend=False,
    )
    return fig


def grafico_feature_importance(predicciones_serie: dict, top_n: int = 10) -> go.Figure:
    """
    Compara feature importance de Random Forest vs XGBoost.

    Solo aplica a estos dos modelos basados en arboles.
    """
    fig = go.Figure()

    for modelo in ("random_forest", "xgboost"):
        if modelo not in predicciones_serie or predicciones_serie[modelo] is None:
            continue
        fi = predicciones_serie[modelo].get("feature_importance")
        if fi is None:
            continue

        # Top N features
        fi_items = sorted(fi.items(), key=lambda x: -x[1])[:top_n]
        features = [f for f, _ in fi_items]
        valores = [v for _, v in fi_items]

        fig.add_trace(go.Bar(
            x=valores, y=features,
            orientation="h",
            name=NOMBRES_MODELOS[modelo],
            marker_color=COLORES_MODELOS[modelo],
        ))

    fig.update_layout(
        title="Feature Importance: Random Forest vs XGBoost",
        xaxis_title="Importancia relativa",
        yaxis_title="Feature",
        template="plotly_white",
        height=400,
        barmode="group",
        yaxis=dict(autorange="reversed"),
    )
    return fig


def grafico_heatmap_modelo_vs_serie(ranking_por_serie: pd.DataFrame) -> go.Figure:
    """
    Heatmap donde cada celda es el MAPE de un modelo en una serie.

    Permite ver de un vistazo en que series cada modelo es bueno o malo.
    """
    # Pivot: filas = series, columnas = modelos
    pivot = ranking_por_serie.pivot(
        index="serie", columns="modelo", values="mape"
    )

    # Renombrar columnas con nombres legibles
    pivot = pivot.rename(columns=NOMBRES_MODELOS)

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale="RdYlGn_r",  # Rojo = mal, Verde = bien
        text=np.round(pivot.values, 1),
        texttemplate="%{text}",
        textfont={"size": 9},
        colorbar=dict(title="MAPE (%)"),
        zmin=0,
        zmax=min(50, pivot.values.max()),  # Truncar para evitar outliers
    ))

    fig.update_layout(
        title="MAPE por Modelo y Serie (menor = mejor)",
        xaxis_title="Modelo",
        yaxis_title="Serie",
        template="plotly_white",
        height=800,
    )
    return fig


def grafico_radar_modelos(ranking_global: pd.DataFrame) -> go.Figure:
    """
    Grafico de radar comparando los 4 modelos en multiples metricas
    normalizadas a [0, 1] (donde 1 es mejor).
    """
    # Normalizar: para metricas de error, invertir el orden (menor = mejor)
    df_norm = pd.DataFrame(index=ranking_global.index)

    # RMSE, MAE, MAPE: menor es mejor -> invertir
    for col in ["rmse", "mae", "mape"]:
        max_val = ranking_global[col].max()
        min_val = ranking_global[col].min()
        if max_val == min_val:
            df_norm[col] = 1.0
        else:
            df_norm[col] = 1 - (ranking_global[col] - min_val) / (max_val - min_val)

    # R2: mayor es mejor
    max_val = ranking_global["r2"].max()
    min_val = ranking_global["r2"].min()
    if max_val == min_val:
        df_norm["r2"] = 1.0
    else:
        df_norm["r2"] = (ranking_global["r2"] - min_val) / (max_val - min_val)

    fig = go.Figure()
    for modelo in df_norm.index:
        fig.add_trace(go.Scatterpolar(
            r=[df_norm.loc[modelo, "rmse"], df_norm.loc[modelo, "mae"],
               df_norm.loc[modelo, "mape"], df_norm.loc[modelo, "r2"]],
            theta=["RMSE (inv)", "MAE (inv)", "MAPE (inv)", "R^2"],
            fill="toself",
            name=NOMBRES_MODELOS[modelo],
            line_color=COLORES_MODELOS[modelo],
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Comparativa global de modelos (1 = mejor)",
        template="plotly_white",
        height=500,
    )
    return fig


# =============================================================================
# RESUMEN TEXTUAL
# =============================================================================

def generar_resumen_textual(resultado_predictor: dict) -> str:
    """
    Genera un parrafo en lenguaje natural describiendo los resultados
    de la comparativa de modelos.
    """
    ranking = resultado_predictor["ranking_global"]
    mejor_por_serie = resultado_predictor["mejor_modelo_por_serie"]

    # Modelo ganador global
    mejor_global = ranking.index[0]
    mejor_global_nombre = NOMBRES_MODELOS[mejor_global]

    # Conteo de victorias por modelo
    victorias = {}
    for serie, modelo in mejor_por_serie.items():
        if modelo is None:
            continue
        victorias[modelo] = victorias.get(modelo, 0) + 1

    victorias_ordenadas = sorted(victorias.items(), key=lambda x: -x[1])

    texto = f"**Modelo ganador global:** {mejor_global_nombre} "
    texto += f"con MAPE promedio de {ranking.loc[mejor_global, 'mape']:.2f}% "
    texto += f"sobre el periodo de test {ANIO_INICIO_TEST}-{ANIO_FIN_TEST}.\n\n"

    texto += "**Victorias por modelo (cantidad de series donde gana):**\n"
    for modelo, n in victorias_ordenadas:
        texto += f"- {NOMBRES_MODELOS[modelo]}: {n} series\n"

    texto += f"\n**Ranking global por MAPE promedio:**\n"
    for i, (modelo, fila) in enumerate(ranking.iterrows(), 1):
        texto += (f"{i}. {NOMBRES_MODELOS[modelo]}: "
                  f"MAPE={fila['mape']:.2f}% | "
                  f"RMSE={fila['rmse']:.0f} | "
                  f"MAE={fila['mae']:.0f} | "
                  f"R^2={fila['r2']:.3f}\n")

    return texto


# =============================================================================
# AGENTE PRINCIPAL
# =============================================================================

def ejecutar(resultado_predictor: dict, verbose: bool = True) -> dict:
    """
    Punto de entrada del agente. Genera todas las visualizaciones comparativas.
    """
    if verbose:
        print("[Agente Validador] Generando visualizaciones comparativas...")

    ranking_global = resultado_predictor["ranking_global"]
    ranking_por_serie = resultado_predictor["ranking_por_serie"]

    graficos = {
        "heatmap_modelo_serie": grafico_heatmap_modelo_vs_serie(ranking_por_serie),
        "radar_modelos": grafico_radar_modelos(ranking_global),
    }

    resumen = generar_resumen_textual(resultado_predictor)

    if verbose:
        print("[Agente Validador] Listo")

    return {
        "graficos": graficos,
        "resumen_textual": resumen,
        "ranking_global": ranking_global,
        "ranking_por_serie": ranking_por_serie,
    }


if __name__ == "__main__":
    print("Este agente recibe resultados del Predictor.")
    print("Ejecutalo via el orquestador o pasale resultados manualmente.")
