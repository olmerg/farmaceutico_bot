"""Evaluador del agente farmacéutico — Fase 1.

Métricas objetivas: Tool Precision, Exactitud, Consistencia, Latencia.
"""

import json
import os
import time

import pandas as pd

from agente import crear_agente, responder

DATA_PATH = "data/medicamentos.csv"
EVAL_PATH = "data/evaluacion.json"
RESULT_PATH = "resultados/metricas.csv"


def contar_medicamentos(respuesta: str) -> int:
    """Cuenta medicamentos mencionados en la respuesta.

    Acepta dos formatos:
    - Líneas que empiezan con '-'
    - Filas de tabla Markdown (empiezan con '|')
    """
    count = 0
    for linea in respuesta.split("\n"):
        linea = linea.strip()
        if linea.startswith("- "):
            count += 1
        elif linea.startswith("|") and "---" not in linea and "Producto" not in linea:
            count += 1
    return count


def evaluar_caso(agente, caso: dict) -> dict:
    """Evalúa un solo caso y retorna métricas."""
    print(f"\n[Caso {caso['id']}] {caso['pregunta']}")

    # Ejecutar agente y medir latencia
    t0 = time.time()
    resultado = agente.invoke(
        {"messages": [("user", caso["pregunta"])]},
        config={"configurable": {"thread_id": f"eval-{caso['id']}"}}
    )
    latencia = time.time() - t0

    respuesta = resultado["messages"][-1].content

    # Tool Precision: ¿llamó a la tool correcta?
    tools_llamadas = []
    for msg in resultado.get("messages", []):
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for call in msg.tool_calls:
                tools_llamadas.append(call.get("name", ""))

    tool_ok = caso["tool_esperada"] in tools_llamadas if tools_llamadas else False

    # Exactitud: ¿cumple el criterio de éxito?
    min_esperado = int(caso["criterio_exito"].split("al menos ")[1].split(" ")[0])
    exactitud_ok = contar_medicamentos(respuesta) >= min_esperado

    print(f"  Tool: {tools_llamadas[0] if tools_llamadas else 'ninguna'} | "
          f"Correcta: {tool_ok} | Exactitud: {exactitud_ok} | "
          f"Latencia: {latencia:.2f}s")

    return {
        "id": caso["id"],
        "pregunta": caso["pregunta"],
        "tool_esperada": caso["tool_esperada"],
        "tool_llamada": tools_llamadas[0] if tools_llamadas else "ninguna",
        "tool_precision": tool_ok,
        "exactitud": exactitud_ok,
        "latencia_seg": round(latencia, 2),
    }


def main():
    """Ejecuta la evaluación completa."""
    casos = json.loads(open(EVAL_PATH, encoding="utf-8").read())
    agente = crear_agente()

    resultados = [evaluar_caso(agente, caso) for caso in casos]

    # Consistencia: 3 ejecuciones del caso 1
    print("\n[Consistencia] Ejecutando caso 1 tres veces...")
    respuestas = [responder(agente, casos[0]["pregunta"]).strip() for _ in range(3)]
    consistencia = len(set(respuestas)) == 1
    for r in resultados:
        if r["id"] == 1:
            r["consistencia"] = consistencia

    # Resumen
    total = len(resultados)
    precision_total = sum(1 for r in resultados if r["tool_precision"]) / total
    exactitud_total = sum(1 for r in resultados if r["exactitud"]) / total
    latencia_promedio = sum(r["latencia_seg"] for r in resultados) / total

    print("\n" + "=" * 50)
    print("RESULTADOS")
    print("=" * 50)
    print(f"Tool Precision: {precision_total:.1%}")
    print(f"Exactitud: {exactitud_total:.1%}")
    print(f"Consistencia: {consistencia}")
    print(f"Latencia promedio: {latencia_promedio:.2f}s")

    # Guardar CSV
    os.makedirs("resultados", exist_ok=True)
    pd.DataFrame(resultados).to_csv(RESULT_PATH, index=False)
    print(f"\nGuardado en {RESULT_PATH}")


if __name__ == "__main__":
    main()
