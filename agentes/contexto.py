"""
Agente 4 - Contexto y Noticias.

Responsabilidad: buscar noticias recientes sobre el sector energetico,
clasificarlas por tematica, analizar sentimiento basico y producir un
resumen que aporte contexto cualitativo al usuario del dashboard.

IMPORTANTE: este agente NO modifica las predicciones numericas del Agente 3.
Su rol es informativo: provee al usuario el contexto del mundo real para
que pueda interpretar las predicciones con criterio.

Entrada: opcional (queries personalizadas), por defecto usa config
Salida:  diccionario con noticias clasificadas y resumen agregado
"""

import json
import re
import time
import unicodedata
import urllib.parse
from datetime import datetime, timedelta
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from config import (
    DIR_NOTICIAS, QUERIES_NOTICIAS, MAX_NOTICIAS, DIAS_ANTIGUEDAD_MAX,
    CATEGORIAS_NOTICIAS, PALABRAS_POSITIVAS, PALABRAS_NEGATIVAS,
)

# Headers para simular un navegador en las peticiones HTTP
HEADERS_BROWSER = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
}


# =============================================================================
# UTILIDADES DE TEXTO
# =============================================================================

def _normalizar_texto(texto: str) -> str:
    """Convierte a minusculas sin acentos para comparar keywords."""
    if not isinstance(texto, str):
        return ""
    texto = unicodedata.normalize("NFD", texto)
    texto = texto.encode("ascii", "ignore").decode("ascii")
    return texto.lower()


# =============================================================================
# BUSQUEDA DE NOTICIAS (via Google News RSS)
# =============================================================================

def buscar_noticias_google_news(query: str, max_resultados: int = 10) -> list:
    """
    Usa el feed RSS publico de Google News para buscar noticias.

    Esta es una alternativa gratuita y sin API key. Google News expone
    un endpoint RSS estable que devuelve los resultados de busqueda en
    formato XML.

    Args:
        query: texto de busqueda
        max_resultados: cuantas noticias devolver como maximo

    Returns:
        lista de dicts con keys: titulo, link, fecha_publicacion, fuente
    """
    query_encoded = urllib.parse.quote_plus(query)
    url = f"https://news.google.com/rss/search?q={query_encoded}&hl=es-419&gl=CO&ceid=CO:es-419"

    try:
        response = requests.get(url, headers=HEADERS_BROWSER, timeout=15)
        response.raise_for_status()
    except requests.RequestException:
        return []

    soup = BeautifulSoup(response.content, "xml")
    items = soup.find_all("item")[:max_resultados]

    noticias = []
    for item in items:
        titulo = item.title.text if item.title else ""
        link = item.link.text if item.link else ""
        pub_date_str = item.pubDate.text if item.pubDate else ""
        fuente = item.source.text if item.source else "Desconocido"

        # Parsear fecha (formato RFC 822 de RSS)
        try:
            fecha = datetime.strptime(pub_date_str, "%a, %d %b %Y %H:%M:%S %Z")
        except (ValueError, TypeError):
            fecha = None

        noticias.append({
            "titulo": titulo.strip(),
            "link": link.strip(),
            "fecha": fecha.isoformat() if fecha else None,
            "fecha_obj": fecha,
            "fuente": fuente.strip(),
            "query_origen": query,
        })

    return noticias


# =============================================================================
# CLASIFICACION POR CATEGORIA
# =============================================================================

def clasificar_categoria(titulo: str) -> tuple:
    """
    Asigna la categoria mas probable a una noticia segun keywords en el titulo.

    Returns:
        tupla (nombre_categoria, label_corto)
        ("Sin categoria", "") si no hay coincidencias claras
    """
    titulo_norm = _normalizar_texto(titulo)

    # Contar coincidencias de keywords por categoria
    puntajes = {}
    for cat, info in CATEGORIAS_NOTICIAS.items():
        n_matches = sum(1 for kw in info["keywords"] if kw in titulo_norm)
        if n_matches > 0:
            puntajes[cat] = n_matches

    if not puntajes:
        return ("Sin categoria", "")

    # Si hay empate, "Crisis / Escasez" tiene prioridad por ser informacion critica
    max_puntaje = max(puntajes.values())
    candidatas = [c for c, p in puntajes.items() if p == max_puntaje]
    if "Crisis / Escasez" in candidatas:
        categoria = "Crisis / Escasez"
    else:
        categoria = candidatas[0]

    return (categoria, CATEGORIAS_NOTICIAS[categoria]["label"])


# =============================================================================
# ANALISIS DE SENTIMIENTO
# =============================================================================

def analizar_sentimiento(titulo: str) -> tuple:
    """
    Calcula un score de sentimiento basico contando palabras positivas
    y negativas en el titulo.

    Returns:
        tupla (score, label)
        score: float entre -1 y 1
        label: "positivo" | "negativo" | "neutro"
    """
    titulo_norm = _normalizar_texto(titulo)

    n_pos = sum(1 for w in PALABRAS_POSITIVAS if w in titulo_norm)
    n_neg = sum(1 for w in PALABRAS_NEGATIVAS if w in titulo_norm)
    total = n_pos + n_neg

    if total == 0:
        score = 0.0
    else:
        score = (n_pos - n_neg) / total

    if score > 0.2:
        label = "positivo"
    elif score < -0.2:
        label = "negativo"
    else:
        label = "neutro"

    return (float(score), label)


# =============================================================================
# AGREGACION Y RESUMEN
# =============================================================================

def generar_resumen(noticias_procesadas: list) -> dict:
    """
    Construye un resumen estadistico agregado del conjunto de noticias.
    """
    total = len(noticias_procesadas)
    if total == 0:
        return {
            "total_noticias": 0,
            "distribucion_categorias": {},
            "sentimiento_promedio": 0.0,
            "sentimiento_label": "sin datos",
            "alertas": [],
        }

    # Distribucion por categoria
    distribucion = {}
    for n in noticias_procesadas:
        cat = n["categoria"]
        distribucion[cat] = distribucion.get(cat, 0) + 1

    # Sentimiento promedio
    sent_avg = sum(n["sentimiento_score"] for n in noticias_procesadas) / total
    if sent_avg > 0.1:
        sent_label = "positivo"
    elif sent_avg < -0.1:
        sent_label = "negativo"
    else:
        sent_label = "neutro"

    # Alertas: si hay un numero significativo de noticias de crisis o negativas
    alertas = []
    n_crisis = distribucion.get("Crisis / Escasez", 0)
    if n_crisis >= 2:
        alertas.append(
            f"Se detectaron {n_crisis} noticias clasificadas como crisis o escasez. "
            "Considerar este contexto al interpretar las predicciones."
        )

    n_negativas = sum(1 for n in noticias_procesadas if n["sentimiento_label"] == "negativo")
    if n_negativas >= total / 2 and total >= 4:
        alertas.append(
            f"El sentimiento predominante de las noticias es negativo "
            f"({n_negativas}/{total}). Puede haber tension en el sector."
        )

    return {
        "total_noticias": total,
        "distribucion_categorias": distribucion,
        "sentimiento_promedio": float(sent_avg),
        "sentimiento_label": sent_label,
        "alertas": alertas,
    }


# =============================================================================
# CACHE
# =============================================================================

CACHE_PATH = DIR_NOTICIAS / "cache_noticias.json"
CACHE_MAX_HORAS = 6


def cargar_cache() -> dict:
    """Carga el cache si existe y es reciente."""
    if not CACHE_PATH.exists():
        return None
    try:
        with open(CACHE_PATH, encoding="utf-8") as f:
            data = json.load(f)
        # Verificar antiguedad
        ts = datetime.fromisoformat(data["timestamp"])
        if datetime.now() - ts > timedelta(hours=CACHE_MAX_HORAS):
            return None
        return data
    except (json.JSONDecodeError, KeyError, ValueError):
        return None


def guardar_cache(noticias: list, resumen: dict):
    """Guarda los resultados en cache."""
    data = {
        "timestamp": datetime.now().isoformat(),
        "noticias": noticias,
        "resumen": resumen,
    }
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


# =============================================================================
# AGENTE PRINCIPAL
# =============================================================================

def _noticias_demo() -> list:
    """
    Devuelve un conjunto de noticias de demostracion cuando no hay acceso
    a internet o cuando Google News no devuelve resultados.

    Esto permite que el dashboard siga siendo funcional en entornos
    sin red y sirve como ejemplo del tipo de noticias esperadas.
    """
    base_fecha = datetime.now()
    return [
        {
            "titulo": "Colombia acelera la transicion energetica con nuevos proyectos solares",
            "link": "https://example.com/noticia1",
            "fecha": (base_fecha - timedelta(days=3)).isoformat(),
            "fecha_obj": base_fecha - timedelta(days=3),
            "fuente": "El Tiempo (demo)",
            "query_origen": "transicion energetica Colombia",
        },
        {
            "titulo": "Precio del petroleo cae y afecta las exportaciones colombianas",
            "link": "https://example.com/noticia2",
            "fecha": (base_fecha - timedelta(days=7)).isoformat(),
            "fecha_obj": base_fecha - timedelta(days=7),
            "fuente": "Portafolio (demo)",
            "query_origen": "precio petroleo Colombia",
        },
        {
            "titulo": "Fenomeno El Nino genera alerta por posible crisis energetica",
            "link": "https://example.com/noticia3",
            "fecha": (base_fecha - timedelta(days=15)).isoformat(),
            "fecha_obj": base_fecha - timedelta(days=15),
            "fuente": "Semana (demo)",
            "query_origen": "crisis energetica Colombia",
        },
        {
            "titulo": "Ministerio de Minas anuncia nueva regulacion sobre gas natural",
            "link": "https://example.com/noticia4",
            "fecha": (base_fecha - timedelta(days=22)).isoformat(),
            "fecha_obj": base_fecha - timedelta(days=22),
            "fuente": "La Republica (demo)",
            "query_origen": "politica energetica Colombia",
        },
        {
            "titulo": "Inversion en energia eolica supera el record historico en Colombia",
            "link": "https://example.com/noticia5",
            "fecha": (base_fecha - timedelta(days=30)).isoformat(),
            "fecha_obj": base_fecha - timedelta(days=30),
            "fuente": "Dinero (demo)",
            "query_origen": "energia renovable Colombia",
        },
        {
            "titulo": "Construccion de nueva planta de hidrogeno verde inicia en La Guajira",
            "link": "https://example.com/noticia6",
            "fecha": (base_fecha - timedelta(days=40)).isoformat(),
            "fecha_obj": base_fecha - timedelta(days=40),
            "fuente": "El Espectador (demo)",
            "query_origen": "infraestructura energetica",
        },
        {
            "titulo": "Sequia afecta generacion hidroelectrica y aumenta el riesgo de racionamiento",
            "link": "https://example.com/noticia7",
            "fecha": (base_fecha - timedelta(days=50)).isoformat(),
            "fecha_obj": base_fecha - timedelta(days=50),
            "fuente": "Caracol Radio (demo)",
            "query_origen": "crisis energetica Colombia",
        },
        {
            "titulo": "CREG aprueba nuevo decreto sobre tarifas de electricidad",
            "link": "https://example.com/noticia8",
            "fecha": (base_fecha - timedelta(days=60)).isoformat(),
            "fecha_obj": base_fecha - timedelta(days=60),
            "fuente": "Bluradio (demo)",
            "query_origen": "regulacion energia",
        },
    ]


def ejecutar(forzar_refresh: bool = False, verbose: bool = True) -> dict:
    """
    Punto de entrada del agente.

    Args:
        forzar_refresh: si True, ignora el cache y vuelve a buscar
        verbose: si True, imprime progreso

    Returns:
        dict con:
            - "noticias": lista de noticias procesadas (titulo, link, categoria, sentimiento, etc.)
            - "resumen": estadisticas agregadas
            - "desde_cache": True si los resultados vienen del cache
    """
    # Intentar usar cache
    if not forzar_refresh:
        cache = cargar_cache()
        if cache is not None:
            if verbose:
                ts = cache["timestamp"][:19].replace("T", " ")
                print(f"[Agente Contexto] Usando cache de noticias ({ts})")
            return {
                "noticias": cache["noticias"],
                "resumen": cache["resumen"],
                "desde_cache": True,
            }

    if verbose:
        print(f"[Agente Contexto] Buscando noticias en Google News...")
        print(f"[Agente Contexto] {len(QUERIES_NOTICIAS)} queries a ejecutar")

    # Buscar noticias en multiples queries
    todas_noticias = []
    for i, query in enumerate(QUERIES_NOTICIAS):
        if verbose:
            print(f"[Agente Contexto] ({i+1}/{len(QUERIES_NOTICIAS)}) {query}")
        nuevas = buscar_noticias_google_news(query, max_resultados=5)
        todas_noticias.extend(nuevas)
        # Pausa pequenia para no saturar el servicio
        time.sleep(0.5)

    # Si no se obtuvo nada (probablemente sin acceso a internet),
    # usar noticias demo para mantener funcional el dashboard
    if len(todas_noticias) == 0:
        if verbose:
            print("[Agente Contexto] No se obtuvieron noticias online, usando demo")
        todas_noticias = _noticias_demo()

    if verbose:
        print(f"[Agente Contexto] {len(todas_noticias)} noticias en total")

    # Deduplicar por titulo
    titulos_vistos = set()
    noticias_unicas = []
    for n in todas_noticias:
        titulo_clean = _normalizar_texto(n["titulo"])
        if titulo_clean and titulo_clean not in titulos_vistos:
            titulos_vistos.add(titulo_clean)
            noticias_unicas.append(n)

    # Filtrar por antiguedad
    limite_fecha = datetime.now() - timedelta(days=DIAS_ANTIGUEDAD_MAX)
    noticias_recientes = []
    for n in noticias_unicas:
        if n["fecha_obj"] is None:
            # Si no tiene fecha, la conservamos
            noticias_recientes.append(n)
        elif n["fecha_obj"] >= limite_fecha:
            noticias_recientes.append(n)

    # Ordenar por fecha descendente y limitar
    noticias_recientes.sort(
        key=lambda n: n["fecha_obj"] if n["fecha_obj"] else datetime.min,
        reverse=True
    )
    noticias_recientes = noticias_recientes[:MAX_NOTICIAS]

    if verbose:
        print(f"[Agente Contexto] {len(noticias_recientes)} noticias recientes y unicas")
        print(f"[Agente Contexto] Clasificando y analizando sentimiento...")

    # Clasificar y analizar sentimiento
    noticias_procesadas = []
    for n in noticias_recientes:
        categoria, label_cat = clasificar_categoria(n["titulo"])
        sent_score, sent_label = analizar_sentimiento(n["titulo"])

        noticias_procesadas.append({
            "titulo": n["titulo"],
            "link": n["link"],
            "fecha": n["fecha"],
            "fuente": n["fuente"],
            "categoria": categoria,
            "categoria_label": label_cat,
            "sentimiento_score": sent_score,
            "sentimiento_label": sent_label,
            "query_origen": n["query_origen"],
        })

    # Generar resumen
    resumen = generar_resumen(noticias_procesadas)

    # Guardar en cache
    guardar_cache(noticias_procesadas, resumen)

    if verbose:
        print(f"[Agente Contexto] Resumen:")
        print(f"  Total: {resumen['total_noticias']}")
        print(f"  Sentimiento promedio: {resumen['sentimiento_label']} ({resumen['sentimiento_promedio']:.2f})")
        print(f"  Distribucion: {resumen['distribucion_categorias']}")
        if resumen["alertas"]:
            for a in resumen["alertas"]:
                print(f"  ALERTA: {a}")
        print("[Agente Contexto] Listo")

    return {
        "noticias": noticias_procesadas,
        "resumen": resumen,
        "desde_cache": False,
    }


if __name__ == "__main__":
    resultado = ejecutar(forzar_refresh=True)
    print(f"\nTotal de noticias: {len(resultado['noticias'])}")
    if resultado["noticias"]:
        print(f"\nEjemplo:")
        n = resultado["noticias"][0]
        print(f"  Titulo: {n['titulo']}")
        print(f"  Categoria: {n['categoria']}")
        print(f"  Sentimiento: {n['sentimiento_label']}")
