"""
Script para ejecutar el pipeline completo desde la linea de comando.

Util para generar el cache la primera vez sin tener que abrir Streamlit.
Tras ejecutarlo, el dashboard cargara instantaneamente desde el cache.

Uso:
    python entrenar.py
"""

from orquestador import ejecutar

if __name__ == "__main__":
    print("Iniciando entrenamiento completo del pipeline SAPCER v2...")
    print("Este proceso puede tardar entre 5 y 15 minutos.")
    print("Tras finalizar, el dashboard cargara instantaneamente desde el cache.\n")

    resultados = ejecutar(usar_cache=False, refrescar_noticias=False, verbose=True)

    print("\n" + "=" * 70)
    print("RESUMEN DE RESULTADOS")
    print("=" * 70)
    print(f"\nSeries predichas: {len(resultados['predictor']['predicciones'])}")
    print(f"Noticias procesadas: {resultados['contexto']['resumen']['total_noticias']}")
    print(f"\nRanking global de modelos:")
    print(resultados['predictor']['ranking_global'].round(2))

    print(f"\nMejores modelos por tipo de serie:")
    mejor_por_serie = resultados['predictor']['mejor_modelo_por_serie']
    conteo = {}
    for serie, modelo in mejor_por_serie.items():
        if modelo:
            conteo[modelo] = conteo.get(modelo, 0) + 1
    for modelo, n in sorted(conteo.items(), key=lambda x: -x[1]):
        print(f"  {modelo}: gana en {n} series")

    print(f"\nDuracion total: {resultados['metadata']['duracion_segundos']:.1f} segundos")
    print(f"\nCache guardado en: outputs/pipeline_cache.pkl")
    print("Ahora puedes ejecutar 'streamlit run dashboard.py'")
