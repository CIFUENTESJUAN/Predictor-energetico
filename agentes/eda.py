"""
Agente 2 - Analisis Exploratorio (EDA).

Responsabilidad: analizar el dataset limpio para descubrir tendencias,
correlaciones, composicion de la mezcla energetica y anomalias.

Genera estadisticas y figuras Plotly que se usan en el dashboard.

Entrada:  diccionario con DataFrame limpio y metadata (salida del Agente 1)
Salida:   diccionario con resultados de los analisis y figuras
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from config import COLOR_PRIMARIO, COLOR_SECUNDARIO, COLOR_ALERTA


def analizar_tendencias(df: pd.DataFrame, series: list) -> dict:
    """
    Calcula pendiente lineal, CAGR y crecimiento total para cada serie.

    CAGR (Compound Annual Growth Rate) es la tasa de crecimiento anual
    equivalente: cuanto deberia crecer la serie cada anio para llegar
    desde el valor inicial al final en n anios.
    """
    resultados = {}
    anios = df["anio"].values

    for col in series:
        valores = df[col].values
        # Eliminar NaN para los calculos
        mask = ~np.isnan(valores)
        if mask.sum() < 5:
            continue

        x = anios[mask].astype(float)
        y = valores[mask].astype(float)

        # Tendencia lineal: pendiente de la recta de regresion
        pendiente, intercepto = np.polyfit(x, y, 1)

        # CAGR: solo si los valores inicial y final son positivos
        v_inicio = y[0]
        v_fin = y[-1]
        n_anios = len(y) - 1
        if v_inicio > 0 and v_fin > 0 and n_anios > 0:
            cagr = (v_fin / v_inicio) ** (1 / n_anios) - 1
        else:
            cagr = None

        # Crecimiento total absoluto y porcentual
        crecimiento_abs = v_fin - v_inicio
        crecimiento_pct = (v_fin / v_inicio - 1) * 100 if v_inicio > 0 else None

        # Clasificacion cualitativa
        if pendiente > 0 and (cagr is None or cagr > 0):
            etiqueta = "creciente"
        elif pendiente < 0 and (cagr is None or cagr < 0):
            etiqueta = "decreciente"
        else:
            etiqueta = "estable"

        resultados[col] = {
            "pendiente": float(pendiente),
            "intercepto": float(intercepto),
            "cagr": float(cagr) if cagr is not None else None,
            "crecimiento_absoluto": float(crecimiento_abs),
            "crecimiento_porcentual": float(crecimiento_pct) if crecimiento_pct is not None else None,
            "valor_inicio": float(v_inicio),
            "valor_fin": float(v_fin),
            "tendencia": etiqueta,
        }
    return resultados


def calcular_correlaciones(df: pd.DataFrame, series: list) -> pd.DataFrame:
    """
    Matriz de correlacion de Pearson entre todas las series.

    Una correlacion cercana a 1 indica que dos series se mueven juntas;
    cercana a -1 indica que se mueven en direcciones opuestas.
    """
    return df[series].corr(method="pearson")


def analizar_composicion(df: pd.DataFrame, series_grupo: list,
                         columna_total: str) -> pd.DataFrame:
    """
    Calcula la participacion porcentual de cada serie en el total del grupo.

    Util para ver, por ejemplo, como ha cambiado la mezcla energetica:
    cuanto representaba el petroleo del total primario en 1975 vs 2024.
    """
    composicion = pd.DataFrame({"anio": df["anio"]})
    total = df[columna_total]
    for col in series_grupo:
        composicion[col] = (df[col] / total * 100).round(2)
    return composicion


def detectar_anomalias(df: pd.DataFrame, series: list) -> dict:
    """
    Detecta valores atipicos usando el criterio del rango intercuartilico.

    Un valor se considera anomalia si esta fuera del intervalo:
        [Q1 - 1.5 * IQR, Q3 + 1.5 * IQR]
    donde IQR = Q3 - Q1.

    Este es el criterio estandar de Tukey para detectar outliers.
    """
    anomalias = {}
    for col in series:
        valores = df[col].dropna()
        if len(valores) < 4:
            continue

        q1 = valores.quantile(0.25)
        q3 = valores.quantile(0.75)
        iqr = q3 - q1
        limite_inf = q1 - 1.5 * iqr
        limite_sup = q3 + 1.5 * iqr

        mask_anom = (df[col] < limite_inf) | (df[col] > limite_sup)
        anios_anom = df.loc[mask_anom, "anio"].tolist()

        if anios_anom:
            anomalias[col] = [
                {
                    "anio": int(a),
                    "valor": float(df.loc[df["anio"] == a, col].values[0]),
                    "tipo": "alto" if df.loc[df["anio"] == a, col].values[0] > limite_sup else "bajo"
                }
                for a in anios_anom
            ]
    return anomalias


def grafico_serie_historica(df: pd.DataFrame, columna: str,
                             titulo: str = None) -> go.Figure:
    """Grafico de linea simple para una serie temporal."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["anio"], y=df[columna],
        mode="lines+markers",
        line=dict(color=COLOR_PRIMARIO, width=2),
        marker=dict(size=5),
        name=columna,
    ))
    fig.update_layout(
        title=titulo or columna,
        xaxis_title="Anio",
        yaxis_title="Terajoules (TJ)",
        template="plotly_white",
        height=400,
        showlegend=False,
    )
    return fig


def grafico_matriz_correlacion(matriz: pd.DataFrame) -> go.Figure:
    """Heatmap interactivo de la matriz de correlacion."""
    fig = go.Figure(data=go.Heatmap(
        z=matriz.values,
        x=matriz.columns,
        y=matriz.index,
        colorscale="RdBu",
        zmid=0,
        zmin=-1,
        zmax=1,
        colorbar=dict(title="Correlacion"),
    ))
    fig.update_layout(
        title="Matriz de Correlacion de Pearson",
        template="plotly_white",
        height=700,
        xaxis=dict(tickangle=-45),
    )
    return fig


def grafico_composicion_apilada(composicion: pd.DataFrame,
                                  titulo: str = "Composicion") -> go.Figure:
    """Grafico de areas apiladas mostrando la evolucion de la mezcla."""
    fig = go.Figure()
    columnas_series = [c for c in composicion.columns if c != "anio"]
    colores = px.colors.qualitative.Set3

    for i, col in enumerate(columnas_series):
        fig.add_trace(go.Scatter(
            x=composicion["anio"],
            y=composicion[col],
            mode="lines",
            stackgroup="one",
            name=col.replace("primario_", "").replace("secundario_", "")
                    .replace("consumo_", "").replace("_", " ").title(),
            fillcolor=colores[i % len(colores)],
            line=dict(width=0),
        ))

    fig.update_layout(
        title=titulo,
        xaxis_title="Anio",
        yaxis_title="Participacion (%)",
        template="plotly_white",
        height=500,
        hovermode="x unified",
    )
    return fig


def ejecutar(resultado_ingesta: dict, verbose: bool = True) -> dict:
    """
    Punto de entrada del agente. Ejecuta todos los analisis exploratorios.

    Returns:
        dict con keys:
            - "tendencias": dict de tendencias por serie
            - "correlaciones": DataFrame de matriz de correlacion
            - "composicion_primaria": DataFrame de %
            - "composicion_secundaria": DataFrame de %
            - "composicion_consumo": DataFrame de %
            - "anomalias": dict de anomalias por serie
            - "graficos": dict de figuras Plotly
    """
    df = resultado_ingesta["df_clean"]
    meta = resultado_ingesta["metadata"]

    if verbose:
        print("[Agente EDA] Calculando tendencias...")
    todas_series = meta["todas_las_series"]
    tendencias = analizar_tendencias(df, todas_series)

    if verbose:
        print("[Agente EDA] Calculando correlaciones...")
    correlaciones = calcular_correlaciones(df, todas_series)

    if verbose:
        print("[Agente EDA] Analizando composicion energetica...")
    composicion_primaria = analizar_composicion(
        df, meta["series_primarias"], "primario_total_primarios"
    )
    composicion_secundaria = analizar_composicion(
        df, meta["series_secundarias"], "secundario_total_secundarios"
    )
    composicion_consumo = analizar_composicion(
        df, meta["series_consumo"], "consumo_consumo_total"
    )

    if verbose:
        print("[Agente EDA] Detectando anomalias...")
    anomalias = detectar_anomalias(df, todas_series)

    if verbose:
        print("[Agente EDA] Generando figuras...")
    graficos = {
        "matriz_correlacion": grafico_matriz_correlacion(correlaciones),
        "composicion_primaria": grafico_composicion_apilada(
            composicion_primaria, "Composicion de la Oferta Primaria"
        ),
        "composicion_secundaria": grafico_composicion_apilada(
            composicion_secundaria, "Composicion de la Oferta Secundaria"
        ),
        "composicion_consumo": grafico_composicion_apilada(
            composicion_consumo, "Composicion del Consumo Final por Sector"
        ),
    }

    if verbose:
        n_anom = sum(len(v) for v in anomalias.values())
        print(f"[Agente EDA] {n_anom} anomalias detectadas en {len(anomalias)} series")
        print("[Agente EDA] Listo")

    return {
        "tendencias": tendencias,
        "correlaciones": correlaciones,
        "composicion_primaria": composicion_primaria,
        "composicion_secundaria": composicion_secundaria,
        "composicion_consumo": composicion_consumo,
        "anomalias": anomalias,
        "graficos": graficos,
    }


if __name__ == "__main__":
    from agentes import ingesta
    res_ingesta = ingesta.ejecutar()
    res_eda = ejecutar(res_ingesta)
    print("\nTendencias principales:")
    for serie, info in list(res_eda["tendencias"].items())[:5]:
        print(f"  {serie}: {info['tendencia']} (CAGR={info['cagr']})")
