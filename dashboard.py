"""
Dashboard interactivo del sistema SAPCER v2.

Interfaz visual profesional tipo McKinsey con paleta clara y elementos
informativos para que el usuario explore datos historicos, compare
modelos de Machine Learning y consulte el contexto noticioso del sector.

Ejecutar con:
    streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import orquestador
from config import (
    COLOR_PRIMARIO, COLOR_SECUNDARIO, COLOR_ACENTO, COLOR_EXITO,
    COLOR_ALERTA, COLOR_NEUTRO, COLORES_MODELOS, NOMBRES_MODELOS,
    COLOR_REAL, ANIO_INICIO_TEST, ANIO_FIN_TEST, ANIO_HORIZONTE,
    CATEGORIAS_NOTICIAS,
)


# =============================================================================
# CONFIGURACION DE LA PAGINA
# =============================================================================

st.set_page_config(
    page_title="SAPCER v2 - Prediccion Energetica Colombia",
    page_icon="E",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS personalizado para look profesional tipo McKinsey
st.markdown(f"""
<style>
    /* Tipografia y colores generales */
    .main {{
        background-color: #FAFBFC;
    }}

    /* Headers */
    h1 {{
        color: {COLOR_PRIMARIO} !important;
        font-weight: 600 !important;
        border-bottom: 3px solid {COLOR_PRIMARIO};
        padding-bottom: 0.5rem;
        margin-bottom: 1.5rem;
    }}
    h2 {{
        color: {COLOR_PRIMARIO} !important;
        font-weight: 500 !important;
        margin-top: 1.5rem;
    }}
    h3 {{
        color: #1B2A4E !important;
        font-weight: 500 !important;
    }}

    /* Cards de metricas */
    div[data-testid="metric-container"] {{
        background-color: white;
        border: 1px solid #E1E4E8;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }}
    div[data-testid="metric-container"] label {{
        color: {COLOR_NEUTRO} !important;
        font-size: 0.875rem !important;
        font-weight: 500 !important;
    }}
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {{
        color: {COLOR_PRIMARIO} !important;
        font-size: 1.75rem !important;
        font-weight: 600 !important;
    }}

    /* Sidebar */
    [data-testid="stSidebar"] {{
        background-color: #F4F5F7;
        border-right: 1px solid #DFE1E6;
    }}
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {{
        color: {COLOR_PRIMARIO} !important;
    }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background-color: white;
        padding: 4px;
        border-radius: 8px;
        border: 1px solid #E1E4E8;
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: transparent;
        border-radius: 6px;
        padding: 8px 16px;
        color: {COLOR_NEUTRO};
        font-weight: 500;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {COLOR_PRIMARIO} !important;
        color: white !important;
    }}

    /* Alertas */
    .stAlert {{
        border-radius: 6px;
    }}

    /* Tablas */
    .dataframe {{
        border: none !important;
    }}

    /* Boton primario */
    .stButton > button {{
        background-color: {COLOR_PRIMARIO};
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        transition: background-color 0.2s;
    }}
    .stButton > button:hover {{
        background-color: {COLOR_SECUNDARIO};
    }}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# CARGA DEL PIPELINE (con cache de Streamlit)
# =============================================================================

@st.cache_resource(show_spinner=False)
def cargar_pipeline():
    """
    Carga (o ejecuta si no hay cache) el pipeline completo.
    El decorador @st.cache_resource asegura que se ejecute una sola vez
    por sesion de Streamlit.
    """
    return orquestador.ejecutar(usar_cache=True, refrescar_noticias=False, verbose=False)


def refrescar_noticias():
    """Refresca las noticias sin re-entrenar los modelos."""
    return orquestador.ejecutar(usar_cache=True, refrescar_noticias=True, verbose=False)


# =============================================================================
# UTILIDADES DE FORMATO
# =============================================================================

def formatear_numero(valor: float) -> str:
    """Formatea numeros grandes con separadores de miles."""
    if abs(valor) >= 1e6:
        return f"{valor/1e6:.2f}M"
    if abs(valor) >= 1e3:
        return f"{valor/1e3:.1f}K"
    return f"{valor:.0f}"


def nombre_legible_serie(col: str) -> str:
    """Convierte 'consumo_industrial' -> 'Industrial' (segun tipo)."""
    if col.startswith("primario_"):
        return col.replace("primario_", "").replace("_", " ").title()
    if col.startswith("secundario_"):
        return col.replace("secundario_", "").replace("_", " ").title()
    if col.startswith("consumo_"):
        return col.replace("consumo_", "").replace("_", " ").title()
    if col.startswith("total_"):
        return col.replace("total_", "Total ").replace("_", " ").title()
    return col.replace("_", " ").title()


def grupo_de_serie(col: str) -> str:
    """Devuelve el grupo al que pertenece una serie."""
    if col.startswith("primario_"):
        return "Energia Primaria"
    if col.startswith("secundario_"):
        return "Energia Secundaria"
    if col.startswith("consumo_"):
        return "Consumo por Sector"
    if col.startswith("total_"):
        return "Totales"
    return "Otros"


# =============================================================================
# HEADER PRINCIPAL
# =============================================================================

def render_header():
    col1, col2 = st.columns([5, 1])
    with col1:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {COLOR_PRIMARIO} 0%, {COLOR_SECUNDARIO} 100%);
                    padding: 1.5rem 2rem; border-radius: 10px; color: white;
                    margin-bottom: 1.5rem;">
            <h1 style="color: white !important; border: none !important;
                       margin: 0 !important; padding: 0 !important;
                       font-size: 1.8rem !important;">
                SAPCER v2
            </h1>
            <p style="color: rgba(255,255,255,0.9); margin: 0.3rem 0 0 0;
                      font-size: 0.95rem;">
                Sistema Agentico de Prediccion de Consumo Energetico
                con Reconocimiento de Contexto - Colombia 1975-{ANIO_HORIZONTE}
            </p>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# SIDEBAR - CONTROLES GLOBALES
# =============================================================================

def render_sidebar(pipeline):
    st.sidebar.markdown(f"""
    <div style="padding: 0.5rem 0 1rem 0; border-bottom: 1px solid #DFE1E6;
                margin-bottom: 1rem;">
        <h3 style="margin: 0; color: {COLOR_PRIMARIO};">Panel de Control</h3>
    </div>
    """, unsafe_allow_html=True)

    meta = pipeline["ingesta"]["metadata"]
    series_primarias = meta["series_primarias"]
    series_secundarias = meta["series_secundarias"]
    series_consumo = meta["series_consumo"]
    totales = ["primario_total_primarios", "secundario_total_secundarios",
               "consumo_consumo_total", "total_total_general"]
    totales = [t for t in totales if t in meta["todas_las_series"]]

    st.sidebar.markdown("### Filtros de visualizacion")

    grupos_disponibles = st.sidebar.multiselect(
        "Grupos de series",
        options=["Totales", "Consumo por Sector", "Energia Primaria", "Energia Secundaria"],
        default=["Totales", "Consumo por Sector"],
        help="Selecciona los grupos de series que quieres visualizar"
    )

    series_filtradas = []
    if "Totales" in grupos_disponibles:
        series_filtradas.extend(totales)
    if "Consumo por Sector" in grupos_disponibles:
        series_filtradas.extend(series_consumo)
    if "Energia Primaria" in grupos_disponibles:
        series_filtradas.extend(series_primarias)
    if "Energia Secundaria" in grupos_disponibles:
        series_filtradas.extend(series_secundarias)

    # Seleccion de serie focal
    st.sidebar.markdown("### Serie focal")
    if series_filtradas:
        opciones_display = {s: nombre_legible_serie(s) for s in series_filtradas}
        # Default a consumo_consumo_total si esta disponible, sino primera
        default_idx = 0
        if "consumo_consumo_total" in series_filtradas:
            default_idx = series_filtradas.index("consumo_consumo_total")

        serie_focal = st.sidebar.selectbox(
            "Selecciona una serie para el analisis detallado",
            options=series_filtradas,
            format_func=lambda x: f"[{grupo_de_serie(x)}] {opciones_display[x]}",
            index=default_idx,
        )
    else:
        serie_focal = None
        st.sidebar.warning("Selecciona al menos un grupo")

    # Modelo focal
    st.sidebar.markdown("### Modelo focal")
    modelo_focal = st.sidebar.selectbox(
        "Modelo a destacar",
        options=["todos", "ridge", "random_forest", "xgboost", "mlp"],
        format_func=lambda x: "Todos los modelos" if x == "todos" else NOMBRES_MODELOS[x],
        help="Selecciona un modelo especifico para destacar o 'Todos' para ver la comparativa completa"
    )

    # Boton para refrescar noticias
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Datos de contexto")
    if st.sidebar.button("Refrescar noticias", use_container_width=True):
        with st.spinner("Buscando noticias actuales..."):
            cargar_pipeline.clear()
            refrescar_noticias()
        st.rerun()

    # Info del pipeline
    st.sidebar.markdown("---")
    ts = pipeline["metadata"]["timestamp"][:19].replace("T", " ")
    dur = pipeline["metadata"]["duracion_segundos"]
    st.sidebar.markdown(f"""
    <div style="font-size: 0.8rem; color: {COLOR_NEUTRO}; padding: 0.5rem 0;">
        <p style="margin: 0;"><b>Ultima ejecucion:</b><br>{ts}</p>
        <p style="margin: 0.3rem 0 0 0;"><b>Duracion pipeline:</b> {dur:.1f}s</p>
    </div>
    """, unsafe_allow_html=True)

    return {
        "series_filtradas": series_filtradas,
        "serie_focal": serie_focal,
        "modelo_focal": modelo_focal,
        "grupos": grupos_disponibles,
    }


# =============================================================================
# PESTANIA 1: RESUMEN EJECUTIVO
# =============================================================================

def render_pestania_resumen(pipeline, controles):
    st.markdown("## Resumen Ejecutivo")

    df = pipeline["ingesta"]["df_clean"]
    meta = pipeline["ingesta"]["metadata"]
    predicciones = pipeline["predictor"]["predicciones"]
    ranking = pipeline["predictor"]["ranking_global"]
    noticias = pipeline["contexto"]

    # Metricas principales en tarjetas
    consumo_actual = df["consumo_consumo_total"].iloc[-1]
    consumo_inicial = df["consumo_consumo_total"].iloc[0]
    crecimiento_total = (consumo_actual / consumo_inicial - 1) * 100
    anios_data = len(df)
    cagr = ((consumo_actual / consumo_inicial) ** (1 / (anios_data - 1)) - 1) * 100

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Consumo total 2024",
            f"{formatear_numero(consumo_actual)} TJ",
            help="Consumo final de energia en el ultimo anio del dataset"
        )

    with col2:
        st.metric(
            "Crecimiento 1975-2024",
            f"+{crecimiento_total:.1f}%",
            f"CAGR {cagr:.2f}%/anio",
            help="Crecimiento acumulado y tasa anual compuesta"
        )

    with col3:
        mejor_modelo = ranking.index[0]
        mejor_mape = ranking.iloc[0]["mape"]
        st.metric(
            "Mejor modelo ML",
            NOMBRES_MODELOS[mejor_modelo],
            f"MAPE {mejor_mape:.2f}%",
            help=f"Modelo con menor error promedio en el test {ANIO_INICIO_TEST}-{ANIO_FIN_TEST}"
        )

    with col4:
        n_noticias = noticias["resumen"]["total_noticias"]
        sentimiento = noticias["resumen"]["sentimiento_label"]
        st.metric(
            "Noticias analizadas",
            n_noticias,
            f"Sentimiento {sentimiento}",
            help="Noticias recopiladas y clasificadas por el agente de contexto"
        )

    st.markdown("---")

    # Grafico principal: consumo total + prediccion del mejor modelo
    st.markdown("### Consumo Total Nacional: Historico y Proyeccion")
    st.caption(f"Linea solida: historico real | Linea punteada: proyeccion del mejor modelo ({NOMBRES_MODELOS[mejor_modelo]})")

    serie_principal = "consumo_consumo_total"
    if serie_principal in predicciones and predicciones[serie_principal][mejor_modelo] is not None:
        res = predicciones[serie_principal][mejor_modelo]

        fig = go.Figure()

        # Historico
        anios_h = sorted(res["historico"].keys())
        valores_h = [res["historico"][a] for a in anios_h]
        fig.add_trace(go.Scatter(
            x=anios_h, y=valores_h, mode="lines",
            name="Historico (1975-2019)",
            line=dict(color=COLOR_REAL, width=2.5),
        ))

        # Test real
        anios_test = sorted(res["test_real"].keys())
        valores_test_real = [res["test_real"][a] for a in anios_test]
        fig.add_trace(go.Scatter(
            x=anios_test, y=valores_test_real, mode="lines+markers",
            name=f"Real test ({ANIO_INICIO_TEST}-{ANIO_FIN_TEST})",
            line=dict(color=COLOR_REAL, width=2.5),
            marker=dict(size=8, color=COLOR_REAL),
        ))

        # Futuro
        anios_f = sorted(res["futuro_pred"].keys())
        valores_f = [res["futuro_pred"][a] for a in anios_f]
        fig.add_trace(go.Scatter(
            x=anios_f, y=valores_f, mode="lines",
            name=f"Proyeccion ({anios_f[0]}-{ANIO_HORIZONTE})",
            line=dict(color=COLOR_ACENTO, width=2.5, dash="dash"),
        ))

        fig.update_layout(
            xaxis_title="Anio",
            yaxis_title="Terajoules (TJ)",
            template="plotly_white",
            height=450,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                         xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Resumen del contexto
    st.markdown("### Contexto Energetico Reciente")
    col_a, col_b = st.columns(2)
    with col_a:
        if noticias["resumen"]["distribucion_categorias"]:
            dist = noticias["resumen"]["distribucion_categorias"]
            fig_pie = go.Figure(data=[go.Pie(
                labels=list(dist.keys()),
                values=list(dist.values()),
                hole=0.4,
                marker=dict(colors=[COLOR_PRIMARIO, COLOR_SECUNDARIO,
                                     COLOR_ACENTO, COLOR_EXITO, COLOR_ALERTA]),
            )])
            fig_pie.update_layout(
                title="Distribucion de noticias por categoria",
                template="plotly_white",
                height=350,
                margin=dict(t=50, b=20),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        st.markdown("**Alertas del agente de contexto**")
        if noticias["resumen"]["alertas"]:
            for alerta in noticias["resumen"]["alertas"]:
                st.warning(alerta)
        else:
            st.info("No se detectaron alertas en las noticias recientes.")

        st.markdown("**Ultimas noticias**")
        for n in noticias["noticias"][:3]:
            st.markdown(
                f"<div style='padding: 0.5rem; background: white; "
                f"border-left: 3px solid {COLOR_PRIMARIO}; margin-bottom: 0.5rem;'>"
                f"<b style='font-size: 0.9rem;'>{n['titulo']}</b><br>"
                f"<span style='color: {COLOR_NEUTRO}; font-size: 0.8rem;'>"
                f"{n['fuente']} - {n['categoria']}</span>"
                f"</div>",
                unsafe_allow_html=True
            )


# =============================================================================
# PESTANIA 2: DATOS HISTORICOS
# =============================================================================

def render_pestania_historicos(pipeline, controles):
    st.markdown("## Exploracion de Datos Historicos")
    st.caption("Visualizacion interactiva de las series energeticas de Colombia 1975-2024")

    df = pipeline["ingesta"]["df_clean"]
    eda = pipeline["eda"]

    # Grafico de la serie focal seleccionada
    serie = controles["serie_focal"]
    if serie:
        st.markdown(f"### {nombre_legible_serie(serie)}")

        tendencia_info = eda["tendencias"].get(serie, {})

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Valor inicial (1975)",
                       f"{formatear_numero(tendencia_info.get('valor_inicio', 0))} TJ")
        with col2:
            st.metric("Valor final (2024)",
                       f"{formatear_numero(tendencia_info.get('valor_fin', 0))} TJ")
        with col3:
            cagr = tendencia_info.get("cagr")
            if cagr is not None:
                st.metric("CAGR", f"{cagr*100:.2f}%")
            else:
                st.metric("CAGR", "N/A")
        with col4:
            st.metric("Tendencia",
                       tendencia_info.get("tendencia", "N/A").title())

        # Grafico de la serie
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["anio"], y=df[serie],
            mode="lines+markers",
            line=dict(color=COLOR_PRIMARIO, width=2.5),
            marker=dict(size=5),
            fill="tozeroy",
            fillcolor="rgba(0, 58, 112, 0.1)",
        ))

        # Marcar anomalias si las hay
        if serie in eda["anomalias"]:
            anom = eda["anomalias"][serie]
            fig.add_trace(go.Scatter(
                x=[a["anio"] for a in anom],
                y=[a["valor"] for a in anom],
                mode="markers",
                marker=dict(size=12, color=COLOR_ALERTA, symbol="x"),
                name="Anomalia",
            ))

        fig.update_layout(
            xaxis_title="Anio",
            yaxis_title="Terajoules (TJ)",
            template="plotly_white",
            height=400,
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Comparativa multi-serie
    st.markdown("### Comparativa de Multiples Series")
    series_a_comparar = controles["series_filtradas"][:8]  # Limitar para legibilidad

    if series_a_comparar:
        fig_comp = go.Figure()
        import plotly.express as px
        colores = px.colors.qualitative.Bold

        for i, s in enumerate(series_a_comparar):
            fig_comp.add_trace(go.Scatter(
                x=df["anio"], y=df[s],
                mode="lines",
                name=nombre_legible_serie(s),
                line=dict(width=2, color=colores[i % len(colores)]),
            ))

        fig_comp.update_layout(
            xaxis_title="Anio",
            yaxis_title="Terajoules (TJ)",
            template="plotly_white",
            height=500,
            hovermode="x unified",
        )
        st.plotly_chart(fig_comp, use_container_width=True)

    # Composicion energetica
    st.markdown("### Composicion de la Mezcla Energetica")
    sub_tabs = st.tabs(["Energia Primaria", "Energia Secundaria", "Consumo por Sector"])

    with sub_tabs[0]:
        st.plotly_chart(eda["graficos"]["composicion_primaria"], use_container_width=True)
    with sub_tabs[1]:
        st.plotly_chart(eda["graficos"]["composicion_secundaria"], use_container_width=True)
    with sub_tabs[2]:
        st.plotly_chart(eda["graficos"]["composicion_consumo"], use_container_width=True)

    # Correlaciones
    st.markdown("### Matriz de Correlacion")
    st.caption("Identifica que series se mueven juntas (azul = correlacion positiva, rojo = negativa)")
    st.plotly_chart(eda["graficos"]["matriz_correlacion"], use_container_width=True)


# =============================================================================
# PESTANIA 3: MODELOS Y PREDICCIONES
# =============================================================================

def render_pestania_modelos(pipeline, controles):
    st.markdown("## Modelos de Machine Learning")
    st.caption("Comparativa de 4 modelos entrenados con busqueda de hiperparametros via GridSearchCV + TimeSeriesSplit")

    serie = controles["serie_focal"]
    if serie is None:
        st.warning("Selecciona una serie en el panel lateral")
        return

    predicciones = pipeline["predictor"]["predicciones"]
    if serie not in predicciones:
        st.error(f"No hay predicciones disponibles para {serie}")
        return

    pred_serie = predicciones[serie]

    st.markdown(f"### Serie analizada: {nombre_legible_serie(serie)}")

    # Grafico principal: real vs predicciones de los 4 modelos
    from agentes.validador import grafico_predicciones_vs_real
    fig = grafico_predicciones_vs_real(pred_serie, nombre_legible_serie(serie))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Metricas de error en el periodo de test (2020-2024)")

    # Tabla de metricas
    filas = []
    for modelo, res in pred_serie.items():
        if res is None:
            continue
        m = res["metricas"]
        filas.append({
            "Modelo": NOMBRES_MODELOS[modelo],
            "RMSE (TJ)": f"{m['rmse']:.0f}",
            "MAE (TJ)": f"{m['mae']:.0f}",
            "MAPE (%)": f"{m['mape']:.2f}",
            "R^2": f"{m['r2']:.3f}",
        })
    df_metricas = pd.DataFrame(filas)

    # Resaltar el mejor por MAPE
    mejor_idx = df_metricas["MAPE (%)"].astype(float).idxmin()

    def highlight_mejor(row):
        if row.name == mejor_idx:
            return [f"background-color: {COLOR_EXITO}; color: white;"] * len(row)
        return [""] * len(row)

    st.dataframe(
        df_metricas.style.apply(highlight_mejor, axis=1),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        f"**RMSE:** error en TJ (raiz del error cuadratico medio). "
        f"**MAE:** error absoluto promedio. "
        f"**MAPE:** error porcentual promedio. "
        f"**R^2:** proporcion de varianza explicada (mas cercano a 1 es mejor)."
    )

    # Grafico de barras comparativo
    st.markdown("### Comparativa visual de metricas")
    metrica_seleccionada = st.radio(
        "Metrica a visualizar",
        options=["mape", "rmse", "mae", "r2"],
        format_func=lambda x: {
            "mape": "MAPE (%)", "rmse": "RMSE (TJ)",
            "mae": "MAE (TJ)", "r2": "R^2"
        }[x],
        horizontal=True,
    )

    from agentes.validador import grafico_barras_metricas
    ranking_serie = pipeline["predictor"]["ranking_por_serie"]
    ranking_filtrado = ranking_serie[ranking_serie["serie"] == serie]
    fig_barras = grafico_barras_metricas(ranking_filtrado, metrica_seleccionada)
    st.plotly_chart(fig_barras, use_container_width=True)

    # Feature importance
    tiene_fi = any(
        res is not None and res.get("feature_importance")
        for modelo, res in pred_serie.items() if modelo in ("random_forest", "xgboost")
    )
    if tiene_fi:
        st.markdown("### Importancia de Features (Random Forest vs XGBoost)")
        st.caption("Que variables temporales son mas importantes para predecir esta serie")
        from agentes.validador import grafico_feature_importance
        fig_fi = grafico_feature_importance(pred_serie)
        st.plotly_chart(fig_fi, use_container_width=True)

    # Hiperparametros ganadores
    st.markdown("### Hiperparametros optimos (encontrados por GridSearchCV)")
    cols = st.columns(len(pred_serie))
    for i, (modelo, res) in enumerate(pred_serie.items()):
        with cols[i]:
            st.markdown(f"**{NOMBRES_MODELOS[modelo]}**")
            if res is None:
                st.write("No disponible")
                continue
            params = res["mejores_params"]
            for k, v in params.items():
                st.markdown(
                    f"<span style='font-size: 0.85rem;'>"
                    f"<b>{k}:</b> {v}</span>",
                    unsafe_allow_html=True
                )


# =============================================================================
# PESTANIA 4: CONTEXTO Y NOTICIAS
# =============================================================================

def render_pestania_contexto(pipeline, controles):
    st.markdown("## Contexto Energetico - Noticias del Sector")
    st.caption(
        "El agente de contexto busca noticias relevantes del sector energetico "
        "y las clasifica automaticamente. Esta informacion complementa las "
        "predicciones numericas con factores cualitativos del mundo real."
    )

    contexto = pipeline["contexto"]
    resumen = contexto["resumen"]

    if resumen["total_noticias"] == 0:
        st.warning("No hay noticias disponibles. Intenta refrescar desde el panel lateral.")
        return

    # Metricas
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total de noticias", resumen["total_noticias"])
    with col2:
        st.metric("Sentimiento global",
                   resumen["sentimiento_label"].title(),
                   f"Score: {resumen['sentimiento_promedio']:.2f}")
    with col3:
        n_cat = len(resumen["distribucion_categorias"])
        st.metric("Categorias detectadas", n_cat)

    # Alertas
    if resumen["alertas"]:
        for alerta in resumen["alertas"]:
            st.warning(alerta)

    st.markdown("---")

    # Distribucion de categorias y sentimiento
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Distribucion por categoria")
        dist = resumen["distribucion_categorias"]
        colores_cats = [COLOR_PRIMARIO, COLOR_SECUNDARIO, COLOR_ACENTO,
                         COLOR_EXITO, COLOR_ALERTA, COLOR_NEUTRO]
        fig_pie = go.Figure(data=[go.Pie(
            labels=list(dist.keys()),
            values=list(dist.values()),
            hole=0.45,
            marker=dict(colors=colores_cats[:len(dist)]),
            textinfo="label+percent",
            textfont=dict(size=11),
        )])
        fig_pie.update_layout(
            template="plotly_white", height=350,
            margin=dict(t=20, b=20),
            showlegend=False,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        st.markdown("#### Sentimiento por noticia")
        # Conteo de sentimientos
        sentimientos = {}
        for n in contexto["noticias"]:
            s = n["sentimiento_label"]
            sentimientos[s] = sentimientos.get(s, 0) + 1

        colores_sent = {
            "positivo": COLOR_EXITO,
            "neutro": COLOR_NEUTRO,
            "negativo": COLOR_ALERTA,
        }
        fig_bar = go.Figure(data=[go.Bar(
            x=list(sentimientos.keys()),
            y=list(sentimientos.values()),
            marker_color=[colores_sent[s] for s in sentimientos.keys()],
            text=list(sentimientos.values()),
            textposition="outside",
        )])
        fig_bar.update_layout(
            xaxis_title="Sentimiento",
            yaxis_title="Cantidad de noticias",
            template="plotly_white",
            height=350,
            margin=dict(t=20, b=20),
            showlegend=False,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")

    # Listado de noticias
    st.markdown("### Detalle de las noticias")

    # Filtros
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        categorias_disponibles = list(set(n["categoria"] for n in contexto["noticias"]))
        cat_filtro = st.multiselect(
            "Filtrar por categoria",
            options=categorias_disponibles,
            default=categorias_disponibles,
        )
    with col_f2:
        sentimientos_disponibles = list(set(n["sentimiento_label"] for n in contexto["noticias"]))
        sent_filtro = st.multiselect(
            "Filtrar por sentimiento",
            options=sentimientos_disponibles,
            default=sentimientos_disponibles,
        )

    # Mostrar noticias filtradas
    noticias_filtradas = [
        n for n in contexto["noticias"]
        if n["categoria"] in cat_filtro and n["sentimiento_label"] in sent_filtro
    ]

    st.markdown(f"**{len(noticias_filtradas)} noticias** coinciden con los filtros")

    for n in noticias_filtradas:
        # Color de borde segun sentimiento
        color_borde = colores_sent.get(n["sentimiento_label"], COLOR_NEUTRO)

        # Fecha legible
        fecha_str = n["fecha"][:10] if n["fecha"] else "Fecha desconocida"

        st.markdown(
            f"""
            <div style='background: white; padding: 1rem 1.2rem;
                        border-left: 4px solid {color_borde};
                        border-radius: 4px; margin-bottom: 0.8rem;
                        box-shadow: 0 1px 2px rgba(0,0,0,0.04);'>
                <div style='display: flex; justify-content: space-between;
                            align-items: start; margin-bottom: 0.3rem;'>
                    <h4 style='margin: 0; font-size: 1rem; color: {COLOR_PRIMARIO};
                                line-height: 1.4;'>{n['titulo']}</h4>
                </div>
                <div style='display: flex; gap: 1rem; font-size: 0.82rem;
                            color: {COLOR_NEUTRO}; margin-top: 0.3rem;'>
                    <span><b>Fuente:</b> {n['fuente']}</span>
                    <span><b>Fecha:</b> {fecha_str}</span>
                    <span style='background: {color_borde}; color: white;
                                  padding: 1px 8px; border-radius: 10px;'>
                        {n['categoria']}
                    </span>
                    <span><b>Sentimiento:</b> {n['sentimiento_label']}</span>
                </div>
                <div style='margin-top: 0.5rem;'>
                    <a href='{n['link']}' target='_blank' style='font-size: 0.85rem;
                       color: {COLOR_SECUNDARIO}; text-decoration: none;'>
                       Ver noticia completa &rarr;
                    </a>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


# =============================================================================
# PESTANIA 5: COMPARATIVA GLOBAL DE MODELOS
# =============================================================================

def render_pestania_comparativa(pipeline, controles):
    st.markdown("## Comparativa Global de Modelos")
    st.caption(
        f"Ranking consolidado a traves de las {len(pipeline['ingesta']['metadata']['todas_las_series'])} series temporales. "
        f"Todos los modelos se evaluaron sobre el mismo periodo de test ({ANIO_INICIO_TEST}-{ANIO_FIN_TEST})."
    )

    validador = pipeline["validador"]
    ranking_global = pipeline["predictor"]["ranking_global"]

    # Ranking principal
    st.markdown("### Ranking global por MAPE promedio")

    # Construir tabla con medallas (texto)
    ranking_display = ranking_global.copy()
    ranking_display = ranking_display.reset_index()
    ranking_display.columns = ["modelo", "RMSE promedio", "MAE promedio",
                                "MAPE promedio (%)", "R^2 promedio"]
    ranking_display["Posicion"] = range(1, len(ranking_display) + 1)
    ranking_display["modelo"] = ranking_display["modelo"].map(NOMBRES_MODELOS)
    ranking_display = ranking_display[["Posicion", "modelo", "MAPE promedio (%)",
                                          "RMSE promedio", "MAE promedio", "R^2 promedio"]]
    ranking_display.columns = ["Posicion", "Modelo", "MAPE (%)", "RMSE (TJ)",
                                 "MAE (TJ)", "R^2"]

    # Formatear numeros
    for col in ["MAPE (%)", "RMSE (TJ)", "MAE (TJ)"]:
        ranking_display[col] = ranking_display[col].apply(lambda x: f"{x:,.2f}")
    ranking_display["R^2"] = ranking_display["R^2"].apply(lambda x: f"{x:.3f}")

    def color_pos(row):
        if row["Posicion"] == 1:
            return ["background-color: #FFD700;"] * len(row)
        if row["Posicion"] == 2:
            return ["background-color: #C0C0C0;"] * len(row)
        if row["Posicion"] == 3:
            return ["background-color: #CD7F32; color: white;"] * len(row)
        return [""] * len(row)

    st.dataframe(
        ranking_display.style.apply(color_pos, axis=1),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Resumen textual")
    st.markdown(validador["resumen_textual"])

    st.markdown("---")

    # Heatmap modelo vs serie
    st.markdown("### Rendimiento por serie")
    st.caption(
        "Cada celda muestra el MAPE de un modelo en una serie. "
        "Verde = bajo error (bueno) | Rojo = alto error (malo)"
    )
    st.plotly_chart(validador["graficos"]["heatmap_modelo_serie"],
                     use_container_width=True)

    # Radar global
    st.markdown("### Comparativa multidimensional")
    st.caption(
        "Cada eje es una metrica normalizada donde 1 es el mejor. "
        "El area cubierta indica que tan bien se comporta el modelo globalmente."
    )
    st.plotly_chart(validador["graficos"]["radar_modelos"],
                     use_container_width=True)


# =============================================================================
# APLICACION PRINCIPAL
# =============================================================================

def main():
    # Cargar pipeline (con spinner mientras se ejecuta)
    with st.spinner("Cargando pipeline de agentes... (la primera vez puede tardar 5-15 minutos)"):
        pipeline = cargar_pipeline()

    render_header()

    controles = render_sidebar(pipeline)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Resumen Ejecutivo",
        "Datos Historicos",
        "Modelos y Predicciones",
        "Contexto y Noticias",
        "Comparativa Global",
    ])

    with tab1:
        render_pestania_resumen(pipeline, controles)
    with tab2:
        render_pestania_historicos(pipeline, controles)
    with tab3:
        render_pestania_modelos(pipeline, controles)
    with tab4:
        render_pestania_contexto(pipeline, controles)
    with tab5:
        render_pestania_comparativa(pipeline, controles)

    # Footer
    st.markdown("---")
    st.markdown(
        f"<div style='text-align: center; color: {COLOR_NEUTRO}; "
        "font-size: 0.85rem; padding: 1rem 0;'>"
        "SAPCER v2 - Sistema multi-agente de prediccion energetica<br>"
        "Asignatura de Machine Learning - Aprendizaje Automatico"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
