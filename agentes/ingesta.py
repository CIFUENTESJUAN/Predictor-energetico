"""
Agente 1 - Ingesta y Preprocesamiento.

Responsabilidad: leer el archivo Excel crudo con encabezados MultiIndex,
limpiarlo, renombrar columnas y dejarlo listo para los siguientes agentes.

Entrada:  ruta al archivo Excel
Salida:   diccionario con DataFrame limpio y metadata del dataset
"""

from pathlib import Path
import pandas as pd
import numpy as np
import unicodedata

from config import ARCHIVO_DATASET


def _normalizar_nombre(texto: str) -> str:
    """
    Convierte un texto a un nombre de columna valido (snake_case sin acentos).

    Ejemplo: "Carbon Mineral" -> "carbon_mineral"
             "Comercial y Publico" -> "comercial_y_publico"
             "Ano" -> "anio" (caso especial para que sea claro)
    """
    # Quitar acentos
    texto_sin_acentos = unicodedata.normalize("NFD", texto)
    texto_sin_acentos = texto_sin_acentos.encode("ascii", "ignore").decode("ascii")
    # A minusculas
    texto_norm = texto_sin_acentos.lower()
    # Reemplazar caracteres especiales por guion bajo
    texto_norm = (texto_norm.replace(" ", "_")
                            .replace("/", "_")
                            .replace(".", "")
                            .replace("(", "")
                            .replace(")", "")
                            .replace(",", ""))
    # Eliminar guiones bajos duplicados
    while "__" in texto_norm:
        texto_norm = texto_norm.replace("__", "_")
    texto_norm = texto_norm.strip("_")

    # Caso especial: la columna "Ano" original es "Anio" (con tilde en N)
    # tras quitar acentos queda como "ano" que es confuso, lo renombramos.
    if texto_norm == "ano":
        texto_norm = "anio"

    return texto_norm


def _mapear_columna_multiindex(grupo: str, subgrupo: str) -> str:
    """
    Construye el nombre final de columna a partir del MultiIndex original.

    Convierte por ejemplo:
        ("Oferta Interna ... - PRIMARIOS (TJ)", "Gas Natural") -> "primario_gas_natural"
        ("Consumo Final por Sector (TJ)", "Industrial")        -> "consumo_industrial"
        ("Unnamed: 0_level_0", "Anio")                         -> "anio"
    """
    sub_norm = _normalizar_nombre(subgrupo)

    if "Unnamed" in grupo or grupo.strip() == "":
        return sub_norm  # Es la columna del anio
    if "PRIMARIOS" in grupo:
        return f"primario_{sub_norm}"
    if "SECUNDARIOS" in grupo:
        return f"secundario_{sub_norm}"
    if "Consumo Final por Sector" in grupo:
        return f"consumo_{sub_norm}"
    if "TOTAL" in grupo:
        return f"total_{sub_norm}"
    return sub_norm


def cargar_dataset_crudo(ruta: Path = ARCHIVO_DATASET) -> pd.DataFrame:
    """
    Lee el Excel original con header de dos niveles y devuelve el DataFrame crudo.
    """
    return pd.read_excel(ruta, sheet_name=0, header=[0, 1])


def limpiar_dataset(df_crudo: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica todas las transformaciones de limpieza al DataFrame crudo.

    Pasos:
    1. Renombrar columnas usando el mapeo del MultiIndex
    2. Eliminar columnas duplicadas o totalmente vacias
    3. Convertir todas las columnas (excepto 'anio') a numerico
    4. Reordenar las columnas por grupos logicos
    5. Verificar integridad temporal (sin huecos de anios)
    """
    # 1. Renombrar columnas
    nuevas_columnas = []
    for grupo, subgrupo in df_crudo.columns:
        nuevas_columnas.append(_mapear_columna_multiindex(str(grupo), str(subgrupo)))
    df = df_crudo.copy()
    df.columns = nuevas_columnas

    # 2. Eliminar columnas vacias o duplicadas (los "Total General.1" y ".2")
    columnas_a_eliminar = []
    for col in df.columns:
        if df[col].isna().all():
            columnas_a_eliminar.append(col)
        elif col.startswith("total_total_general") and col != "total_total_general":
            columnas_a_eliminar.append(col)
    df = df.drop(columns=columnas_a_eliminar)

    # 3. Convertir a numerico
    for col in df.columns:
        if col != "anio":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["anio"] = df["anio"].astype(int)

    # 4. Ordenar por anio para garantizar cronologia
    df = df.sort_values("anio").reset_index(drop=True)

    # 5. Verificacion de integridad temporal
    anios = df["anio"].values
    if not np.array_equal(anios, np.arange(anios.min(), anios.max() + 1)):
        raise ValueError("Hay huecos en la serie temporal de anios")

    return df


def generar_metadata(df: pd.DataFrame) -> dict:
    """
    Construye el diccionario de metadata con informacion estructural del dataset.

    Esta metadata se usa en los demas agentes para saber que series existen
    y como agruparlas.
    """
    cols = df.columns.tolist()

    series_primarias = [c for c in cols if c.startswith("primario_")
                        and c != "primario_total_primarios"]
    series_secundarias = [c for c in cols if c.startswith("secundario_")
                          and c != "secundario_total_secundarios"]
    series_consumo = [c for c in cols if c.startswith("consumo_")
                      and c != "consumo_consumo_total"]
    totales = [c for c in cols if c.startswith("total_")
               or c in ("primario_total_primarios", "secundario_total_secundarios",
                        "consumo_consumo_total")]

    metadata = {
        "anio_inicio": int(df["anio"].min()),
        "anio_fin": int(df["anio"].max()),
        "n_filas": len(df),
        "n_columnas": len(cols),
        "series_primarias": series_primarias,
        "series_secundarias": series_secundarias,
        "series_consumo": series_consumo,
        "series_totales": totales,
        "todas_las_series": [c for c in cols if c != "anio"],
        "columnas_con_nulos": {
            c: int(df[c].isna().sum())
            for c in cols if df[c].isna().any()
        },
    }
    return metadata


def ejecutar(ruta: Path = ARCHIVO_DATASET, verbose: bool = True) -> dict:
    """
    Punto de entrada del agente. Ejecuta el flujo completo de ingesta.

    Returns:
        dict con keys:
            - "df_clean": DataFrame limpio
            - "metadata": diccionario con metadata del dataset
    """
    if verbose:
        print("[Agente Ingesta] Cargando dataset crudo...")
    df_crudo = cargar_dataset_crudo(ruta)

    if verbose:
        print(f"[Agente Ingesta] Dataset crudo: {df_crudo.shape}")
        print("[Agente Ingesta] Limpiando y normalizando columnas...")
    df = limpiar_dataset(df_crudo)

    if verbose:
        print(f"[Agente Ingesta] Dataset limpio: {df.shape}")
        print("[Agente Ingesta] Generando metadata...")
    metadata = generar_metadata(df)

    if verbose:
        print(f"[Agente Ingesta] Anios: {metadata['anio_inicio']}-{metadata['anio_fin']}")
        print(f"[Agente Ingesta] {len(metadata['todas_las_series'])} series temporales detectadas")
        if metadata["columnas_con_nulos"]:
            print(f"[Agente Ingesta] Aviso: {len(metadata['columnas_con_nulos'])} columnas tienen valores nulos")
        print("[Agente Ingesta] Listo")

    return {"df_clean": df, "metadata": metadata}


if __name__ == "__main__":
    resultado = ejecutar()
    print("\nPrimeras filas del dataset limpio:")
    print(resultado["df_clean"].head())
    print("\nColumnas:")
    for c in resultado["df_clean"].columns:
        print(f"  - {c}")
