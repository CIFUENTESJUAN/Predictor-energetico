"""
Orquestador del sistema SAPCER v2.

Coordina la ejecucion secuencial de los 5 agentes y gestiona el cache
del pipeline completo para que el dashboard no tenga que re-entrenar
los modelos en cada recarga.
"""

import pickle
import time
from datetime import datetime
from pathlib import Path

from agentes import ingesta, eda, predictor, contexto, validador
from config import DIR_OUTPUTS, VERBOSE

CACHE_PIPELINE = DIR_OUTPUTS / "pipeline_cache.pkl"


def cargar_cache_pipeline() -> dict:
    """Carga los resultados del pipeline si existen en cache."""
    if not CACHE_PIPELINE.exists():
        return None
    try:
        with open(CACHE_PIPELINE, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def guardar_cache_pipeline(resultados: dict):
    """Guarda los resultados del pipeline en cache."""
    with open(CACHE_PIPELINE, "wb") as f:
        pickle.dump(resultados, f)


def ejecutar(usar_cache: bool = True,
              refrescar_noticias: bool = False,
              verbose: bool = VERBOSE) -> dict:
    """
    Ejecuta el pipeline completo del sistema.

    Args:
        usar_cache: si True, intenta cargar resultados de ejecuciones anteriores.
                    El cache se invalida si se borra el archivo manualmente.
        refrescar_noticias: si True, fuerza al agente de contexto a buscar
                             noticias frescas (ignorando su cache propio).
        verbose: imprimir progreso en consola.

    Returns:
        dict con keys:
            - "ingesta": resultado del Agente 1
            - "eda": resultado del Agente 2
            - "predictor": resultado del Agente 3
            - "contexto": resultado del Agente 4
            - "validador": resultado del Agente 5
            - "metadata": informacion de la ejecucion
    """
    inicio = time.time()

    if usar_cache:
        cache = cargar_cache_pipeline()
        if cache is not None:
            if verbose:
                print("[Orquestador] Usando cache del pipeline completo")
                print(f"[Orquestador] Cache creado: {cache['metadata']['timestamp']}")
            # Actualizar solo las noticias si se solicita
            if refrescar_noticias:
                if verbose:
                    print("[Orquestador] Refrescando solo las noticias...")
                cache["contexto"] = contexto.ejecutar(forzar_refresh=True, verbose=verbose)
            return cache

    if verbose:
        print("=" * 70)
        print("SAPCER v2 - Iniciando pipeline completo")
        print("=" * 70)

    # Agente 1: Ingesta
    res_ingesta = ingesta.ejecutar(verbose=verbose)

    # Agente 2: EDA
    res_eda = eda.ejecutar(res_ingesta, verbose=verbose)

    # Agente 4: Contexto (se ejecuta antes que el predictor porque es independiente)
    res_contexto = contexto.ejecutar(forzar_refresh=refrescar_noticias, verbose=verbose)

    # Agente 3: Predictor (el mas largo)
    res_predictor = predictor.ejecutar(res_ingesta, verbose=verbose)

    # Agente 5: Validador
    res_validador = validador.ejecutar(res_predictor, verbose=verbose)

    duracion = time.time() - inicio

    resultados = {
        "ingesta": res_ingesta,
        "eda": res_eda,
        "predictor": res_predictor,
        "contexto": res_contexto,
        "validador": res_validador,
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "duracion_segundos": duracion,
        }
    }

    if verbose:
        print("=" * 70)
        print(f"[Orquestador] Pipeline completado en {duracion:.1f} segundos")
        print("=" * 70)

    # Guardar cache
    guardar_cache_pipeline(resultados)

    return resultados


if __name__ == "__main__":
    res = ejecutar(usar_cache=False)
    print("\nResumen final:")
    print(f"Series predichas: {len(res['predictor']['predicciones'])}")
    print(f"Noticias procesadas: {res['contexto']['resumen']['total_noticias']}")
    print(f"\nRanking global de modelos:")
    print(res['predictor']['ranking_global'].round(2))
