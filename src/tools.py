"""Herramientas (tools) del agente farmacéutico."""

import pandas as pd
from langchain.tools import tool

DATA_PATH = "data/medicamentos.csv"


def cargar_datos() -> pd.DataFrame:
    """Carga el dataset de medicamentos. Fail fast si no existe."""
    try:
        return pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Dataset no encontrado en {DATA_PATH}. "
            "Verifica que el archivo data/medicamentos.csv existe."
        )


df = cargar_datos()


@tool
def buscar_por_principio_activo(principio: str) -> str:
    """Busca medicamentos por principio activo. Ej: 'ACICLOVIR', 'IBUPROFENO'"""
    resultado = df[df["principioactivo"].str.contains(principio, case=False, na=False)]
    if resultado.empty:
        return f"Sin resultados para principio activo '{principio}'"

    lineas = [
        f"{r['producto']} | {r['titular']} | {r['formafarmaceutica']} | {r['estadoregistro']}"
        for _, r in resultado.head(10).iterrows()
    ]
    return f"{len(resultado)} resultados:\n" + "\n".join(lineas)


@tool
def buscar_por_nombre_comercial(nombre: str) -> str:
    """Busca medicamentos por nombre comercial. Ej: 'ADVIL', 'TYLENOL'"""
    resultado = df[df["producto"].str.contains(nombre, case=False, na=False)]
    if resultado.empty:
        return f"Sin resultados para nombre '{nombre}'"

    lineas = [
        f"{r['producto']} | {r['principioactivo']} | {r['titular']}"
        for _, r in resultado.head(10).iterrows()
    ]
    return f"{len(resultado)} resultados:\n" + "\n".join(lineas)


@tool
def consultar_vencidos() -> str:
    """Consulta medicamentos con registro sanitario vencido."""
    vencidos = df[df["estadoregistro"] == "Vencido"]
    if vencidos.empty:
        return "No hay medicamentos vencidos"

    lineas = [
        f"{r['producto']} | {r['titular']} | Venció: {r['fechavencimiento']}"
        for _, r in vencidos.head(10).iterrows()
    ]
    return f"{len(vencidos)} vencidos (muestra 10):\n" + "\n".join(lineas)


@tool
def consultar_por_titular(titular: str) -> str:
    """Consulta medicamentos por laboratorio titular. Ej: 'PFIZER', 'NOVARTIS'"""
    resultado = df[df["titular"].str.contains(titular, case=False, na=False)]
    if resultado.empty:
        return f"Sin resultados para titular '{titular}'"

    lineas = [
        f"{r['producto']} | {r['principioactivo']} | {r['estadoregistro']}"
        for _, r in resultado.head(10).iterrows()
    ]
    return f"{len(resultado)} resultados:\n" + "\n".join(lineas)


@tool
def consultar_por_via(via: str) -> str:
    """Consulta medicamentos por vía de administración. Ej: 'ORAL', 'PARENTERAL'"""
    resultado = df[df["viaadministracion"].str.contains(via, case=False, na=False)]
    if resultado.empty:
        return f"Sin resultados para vía '{via}'"

    lineas = [
        f"{r['producto']} | {r['principioactivo']} | {r['formafarmaceutica']}"
        for _, r in resultado.head(10).iterrows()
    ]
    return f"{len(resultado)} resultados:\n" + "\n".join(lineas)


TOOLS = [
    buscar_por_principio_activo,
    buscar_por_nombre_comercial,
    consultar_vencidos,
    consultar_por_titular,
    consultar_por_via,
]
